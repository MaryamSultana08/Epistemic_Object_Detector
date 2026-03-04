import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image, ImageDraw

from retinanet import model
from retinanet import random_set
from retinanet.dataloader import CocoDataset, Normalizer, Resizer


def split_minival_ids(image_ids: List[int], minival_size: int, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    minival_ids = ids[:minival_size]
    trainval_ids = ids[minival_size:]
    return trainval_ids, minival_ids


def build_model(args, num_classes: int, random_set_base_class_names=None):
    kwargs = dict(
        num_classes=num_classes,
        pretrained=False,
        use_dirichlet=True,
        use_random_set=args.model_variant == "dirichlet_randomset",
        random_set_betp_loss=args.random_set_betp_loss,
        random_set_path=args.random_set_path,
        random_set_alpha=args.random_set_alpha,
        random_set_beta=args.random_set_beta,
        random_set_base_class_names=random_set_base_class_names if args.model_variant == "dirichlet_randomset" else None,
        dirichlet_coord_l1_weight=args.dirichlet_coord_l1_weight,
        dirichlet_kl_weight=args.dirichlet_kl_weight,
        dirichlet_delta_clip=args.dirichlet_delta_clip,
        dirichlet_target_concentration=args.dirichlet_target_concentration,
        score_threshold=args.score_threshold,
    )

    if args.depth == 18:
        return model.resnet18(**kwargs)
    if args.depth == 34:
        return model.resnet34(**kwargs)
    if args.depth == 50:
        return model.resnet50(**kwargs)
    if args.depth == 101:
        return model.resnet101(**kwargs)
    if args.depth == 152:
        return model.resnet152(**kwargs)
    raise ValueError("Unsupported model depth. Choose from 18, 34, 50, 101, 152.")


def _try_load_state_dict(net, state_dict: Dict[str, torch.Tensor]):
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def load_checkpoint(net, model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format in {model_path}")

    candidates = [state_dict]
    if any(k.startswith("module.") for k in state_dict.keys()):
        candidates.append({k.replace("module.", "", 1): v for k, v in state_dict.items()})
    else:
        candidates.append({f"module.{k}": v for k, v in state_dict.items()})

    last_err = None
    for cand in candidates:
        try:
            return _try_load_state_dict(net, cand)
        except RuntimeError as err:
            last_err = err
            continue

    raise RuntimeError(f"Failed to load checkpoint {model_path}: {last_err}")


def dirichlet_regression_epistemic(reg_alphas: torch.Tensor) -> torch.Tensor:
    # reg_alphas: [B, A, 12], where each coordinate has 3-bin Dirichlet alphas.
    a = reg_alphas.view(reg_alphas.shape[0], reg_alphas.shape[1], 4, 3)
    strength = torch.clamp(a.sum(dim=-1), min=1e-6)  # [B, A, 4]
    # Subjective uncertainty: K / S (K=3 bins). Clamp to [0,1] for readability.
    u = torch.clamp(3.0 / strength, min=0.0, max=1.0)
    return u.mean(dim=-1)  # [B, A]


def randomset_classification_epistemic(
    beliefs: torch.Tensor,
    rs_mass_coeff: torch.Tensor,
) -> torch.Tensor:
    # beliefs: [B, A, num_sets]
    mass = random_set.belief_to_mass(beliefs, rs_mass_coeff.to(beliefs.device), clamp_negative=True)
    unknown = torch.clamp(1.0 - mass.sum(dim=-1), min=0.0)
    norm = torch.clamp(mass.sum(dim=-1) + unknown, min=1e-6)
    return unknown / norm


def combine_uncertainty(reg_u: float, cls_u: float) -> float:
    vals = []
    if math.isfinite(reg_u):
        vals.append(reg_u)
    if math.isfinite(cls_u):
        vals.append(cls_u)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def run_backbone_with_uncertainty(base_model, img_batch: torch.Tensor):
    x = base_model.conv1(img_batch)
    x = base_model.bn1(x)
    x = base_model.relu(x)
    x = base_model.maxpool(x)

    x1 = base_model.layer1(x)
    x2 = base_model.layer2(x1)
    x3 = base_model.layer3(x2)
    x4 = base_model.layer4(x3)

    features = base_model.fpn([x2, x3, x4])

    regression_raw = torch.cat([base_model.regressionModel(feature) for feature in features], dim=1)
    classification_raw = torch.cat([base_model.classificationModel(feature) for feature in features], dim=1)
    anchors = base_model.anchors(img_batch)

    regression_deltas = base_model.regressionModel.alphas_to_deltas(regression_raw, img_batch.shape[0])
    reg_epistemic = dirichlet_regression_epistemic(regression_raw)

    if base_model.use_random_set:
        classification_scores = base_model.classificationModel.beliefs_to_label_scores(classification_raw)
        cls_epistemic = randomset_classification_epistemic(
            classification_raw,
            base_model.classificationModel.rs_mass_coeff,
        )
    else:
        classification_scores = classification_raw
        cls_epistemic = None

    boxes = base_model.regressBoxes(anchors, regression_deltas)
    boxes = base_model.clipBoxes(boxes, img_batch)

    return classification_scores, boxes, reg_epistemic, cls_epistemic


def decode_detections(
    classification_scores: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    reg_epistemic: torch.Tensor,
    cls_epistemic: Optional[torch.Tensor],
    score_threshold: float,
    nms_iou_threshold: float,
    pre_nms_topk: Optional[int],
    max_dets: int,
) -> List[Dict]:
    num_anchors, num_classes = classification_scores.shape
    anchor_ids = torch.arange(num_anchors, dtype=torch.long)

    detections = []
    for class_idx in range(num_classes):
        class_scores = classification_scores[:, class_idx]
        keep = class_scores >= float(score_threshold)
        if keep.sum().item() == 0:
            continue

        scores_k = class_scores[keep]
        boxes_k = boxes_xyxy[keep]
        anchor_ids_k = anchor_ids[keep]

        if pre_nms_topk is not None and pre_nms_topk > 0 and scores_k.numel() > int(pre_nms_topk):
            top_vals, top_idx = torch.topk(scores_k, k=int(pre_nms_topk), largest=True, sorted=True)
            scores_k = top_vals
            boxes_k = boxes_k[top_idx]
            anchor_ids_k = anchor_ids_k[top_idx]

        keep_idx = nms(boxes_k, scores_k, float(nms_iou_threshold))
        if keep_idx.numel() == 0:
            continue

        for idx in keep_idx.tolist():
            anchor_idx = int(anchor_ids_k[idx].item())
            reg_u = float(reg_epistemic[anchor_idx].item())
            cls_u = float("nan")
            if cls_epistemic is not None:
                cls_u = float(cls_epistemic[anchor_idx].item())

            detections.append(
                {
                    "score": float(scores_k[idx].item()),
                    "label": int(class_idx),
                    "anchor_idx": anchor_idx,
                    "box_xyxy": boxes_k[idx].tolist(),
                    "reg_epistemic": reg_u,
                    "cls_epistemic": cls_u,
                    "total_epistemic": combine_uncertainty(reg_u, cls_u),
                }
            )

    detections.sort(key=lambda d: d["score"], reverse=True)
    if max_dets is not None and max_dets > 0:
        detections = detections[: int(max_dets)]
    return detections


def box_iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area_box = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes[:, 3] - boxes[:, 1],
        a_min=0.0,
        a_max=None,
    )
    union = np.clip(area_box + area_boxes - inter, a_min=1e-8, a_max=None)
    return inter / union


def match_detections_to_gt(detections: List[Dict], gt_annots: np.ndarray, iou_threshold: float):
    if gt_annots.shape[0] == 0:
        for det in detections:
            det["is_tp"] = 0
            det["matched_iou"] = 0.0
        return

    gt_boxes = gt_annots[:, :4].astype(np.float32)
    gt_labels = gt_annots[:, 4].astype(np.int64)
    used = np.zeros((gt_boxes.shape[0],), dtype=bool)

    for det in detections:
        det_label = int(det["label"])
        same_class_idx = np.where((gt_labels == det_label) & (~used))[0]
        if same_class_idx.size == 0:
            det["is_tp"] = 0
            det["matched_iou"] = 0.0
            continue

        ious = box_iou_np(np.array(det["box_xyxy"], dtype=np.float32), gt_boxes[same_class_idx])
        best_local = int(np.argmax(ious)) if ious.size > 0 else -1
        best_iou = float(ious[best_local]) if best_local >= 0 else 0.0

        if best_iou >= float(iou_threshold):
            gt_idx = int(same_class_idx[best_local])
            used[gt_idx] = True
            det["is_tp"] = 1
            det["matched_iou"] = best_iou
        else:
            det["is_tp"] = 0
            det["matched_iou"] = best_iou


def uncertainty_to_rgb(u: float) -> Tuple[int, int, int]:
    if not math.isfinite(u):
        return (255, 255, 255)
    u = max(0.0, min(1.0, u))
    # low uncertainty -> green, high uncertainty -> red
    return (int(255.0 * u), int(255.0 * (1.0 - u)), 0)


def save_visualization(
    dataset: CocoDataset,
    idx: int,
    detections: List[Dict],
    gt_annots: np.ndarray,
    output_path: Path,
    max_draw_dets: int,
):
    img_rgb = dataset.load_image(idx)
    img_vis = (np.clip(img_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img_vis)
    draw = ImageDraw.Draw(pil_img)

    dets = sorted(detections, key=lambda d: d["total_epistemic"], reverse=True)
    dets = dets[: max(1, int(max_draw_dets))]

    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
        color = uncertainty_to_rgb(float(det["total_epistemic"]))
        label_name = dataset.labels.get(int(det["label"]), str(det["label"]))
        caption = (
            f"{label_name} s={det['score']:.2f} "
            f"u={det['total_epistemic']:.2f} "
            f"r={det['reg_epistemic']:.2f}"
        )
        if math.isfinite(float(det["cls_epistemic"])):
            caption += f" c={det['cls_epistemic']:.2f}"
        if "is_tp" in det:
            caption += " TP" if int(det["is_tp"]) == 1 else " FP"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        text_y = max(0, y1 - 14)
        draw.text((x1, text_y), caption, fill=color)

    # Ground-truth boxes in red for direct visual comparison.
    if gt_annots is not None and gt_annots.shape[0] > 0:
        for gt in gt_annots:
            gx1, gy1, gx2, gy2 = [int(v) for v in gt[:4].tolist()]
            glabel = int(gt[4])
            glabel_name = dataset.labels.get(glabel, str(glabel))
            draw.rectangle((gx1, gy1, gx2, gy2), outline=(255, 0, 0), width=2)
            gt_text_y = max(0, gy1 - 14)
            draw.text((gx1, gt_text_y), f"GT:{glabel_name}", fill=(255, 0, 0))

    pil_img.save(output_path)


def finite_values(records: List[Dict], key: str) -> np.ndarray:
    vals = [float(r[key]) for r in records if math.isfinite(float(r[key]))]
    if not vals:
        return np.zeros((0,), dtype=np.float32)
    return np.array(vals, dtype=np.float32)


def summarize_detections(records: List[Dict], num_images: int) -> Dict:
    summary = {
        "images_processed": int(num_images),
        "num_detections": int(len(records)),
    }

    total = finite_values(records, "total_epistemic")
    reg = finite_values(records, "reg_epistemic")
    cls = finite_values(records, "cls_epistemic")

    def add_stats(name: str, values: np.ndarray):
        if values.size == 0:
            summary[f"{name}_count"] = 0
            return
        summary[f"{name}_count"] = int(values.size)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std())
        summary[f"{name}_p50"] = float(np.percentile(values, 50))
        summary[f"{name}_p75"] = float(np.percentile(values, 75))
        summary[f"{name}_p90"] = float(np.percentile(values, 90))
        summary[f"{name}_p95"] = float(np.percentile(values, 95))

    add_stats("total_epistemic", total)
    add_stats("reg_epistemic", reg)
    add_stats("cls_epistemic", cls)

    tp = finite_values([r for r in records if int(r.get("is_tp", 0)) == 1], "total_epistemic")
    fp = finite_values([r for r in records if int(r.get("is_tp", 0)) == 0], "total_epistemic")
    summary["tp_count"] = int(tp.size)
    summary["fp_count"] = int(fp.size)
    if tp.size > 0:
        summary["tp_total_epistemic_mean"] = float(tp.mean())
    if fp.size > 0:
        summary["fp_total_epistemic_mean"] = float(fp.mean())
    if tp.size > 0 and fp.size > 0:
        summary["fp_minus_tp_total_epistemic"] = float(fp.mean() - tp.mean())

    if total.size > 0:
        bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.01], dtype=np.float32)
        hist, _ = np.histogram(total, bins=bins)
        summary["total_epistemic_histogram"] = {
            "0.0-0.2": int(hist[0]),
            "0.2-0.4": int(hist[1]),
            "0.4-0.6": int(hist[2]),
            "0.6-0.8": int(hist[3]),
            "0.8-1.0": int(hist[4]),
        }

    return summary


def write_csv(path: Path, records: List[Dict], fieldnames: List[str]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RetinaNet detections with epistemic uncertainty (quantitative + visual)."
    )
    parser.add_argument("--coco_path", required=True, help="Path to COCO directory")
    parser.add_argument("--model_path", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument(
        "--model_variant",
        choices=["dirichlet_std", "dirichlet_randomset"],
        default="dirichlet_std",
        help="Epistemic evaluator currently supports Dirichlet variants only.",
    )
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5)
    parser.add_argument("--pre_nms_topk", type=int, default=1000)
    parser.add_argument("--max_dets", type=int, default=100)
    parser.add_argument("--match_iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=0, help="0 means process full split")
    parser.add_argument("--output_dir", default="uncertainty_eval_output")
    parser.add_argument("--visualize_images", type=int, default=30)
    parser.add_argument("--max_draw_dets", type=int, default=25)

    parser.add_argument(
        "--random_set_path",
        default="",
        help="Path to random-set class clusters (required for dirichlet_randomset).",
    )
    parser.add_argument("--random_set_alpha", type=float, default=0.001)
    parser.add_argument("--random_set_beta", type=float, default=0.001)
    parser.add_argument("--random_set_betp_loss", action="store_true")

    parser.add_argument("--dirichlet_kl_weight", type=float, default=0.005)
    parser.add_argument("--dirichlet_coord_l1_weight", type=float, default=1.0)
    parser.add_argument("--dirichlet_delta_clip", type=float, default=3.0)
    parser.add_argument("--dirichlet_target_concentration", type=float, default=20.0)

    parser.add_argument("--coco_minival_split", action="store_true")
    parser.add_argument("--minival_size", type=int, default=5000)
    parser.add_argument("--minival_seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_variant == "dirichlet_randomset" and not args.random_set_path:
        raise ValueError(
            "--random_set_path is required when --model_variant dirichlet_randomset."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")

    dataset = CocoDataset(
        args.coco_path,
        set_name="val2017",
        transform=transforms.Compose([Normalizer(), Resizer()]),
        # Keep GT loading on standard COCO annotations for TP/FP matching.
        use_dirichlet=False,
        annotation_path=None,
    )

    if args.coco_minival_split:
        _, minival_ids = split_minival_ids(dataset.image_ids, args.minival_size, args.minival_seed)
        dataset.image_ids = minival_ids
        print(f"Using minival split with {len(dataset.image_ids)} images")

    if args.max_images and args.max_images > 0:
        dataset.image_ids = dataset.image_ids[: int(args.max_images)]
        print(f"Limiting to {len(dataset.image_ids)} images")

    random_set_base_class_names = None
    if args.model_variant == "dirichlet_randomset":
        random_set_base_class_names = [str(dataset.labels[i]) for i in range(dataset.num_classes())]

    net = build_model(
        args,
        num_classes=dataset.num_classes(),
        random_set_base_class_names=random_set_base_class_names,
    )
    missing, unexpected = load_checkpoint(net, args.model_path)
    print(f"Loaded checkpoint: {args.model_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if 0 < len(missing) < 30:
        print("Missing key examples:", missing[:10])
    if 0 < len(unexpected) < 30:
        print("Unexpected key examples:", unexpected[:10])

    net = net.to(device)
    net.eval()
    net.freeze_bn()

    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_records = []
    detection_records = []
    saved_visuals = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            img = data["img"]
            scale = float(data["scale"])

            if isinstance(img, torch.Tensor):
                img_batch = img.permute(2, 0, 1).unsqueeze(0).to(device).float()
            else:
                img_batch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float()

            cls_scores, boxes, reg_u, cls_u = run_backbone_with_uncertainty(net, img_batch)

            cls_scores = cls_scores[0].detach().cpu()
            boxes = (boxes[0].detach().cpu()) / max(scale, 1e-8)
            reg_u = reg_u[0].detach().cpu()
            cls_u = cls_u[0].detach().cpu() if cls_u is not None else None

            detections = decode_detections(
                classification_scores=cls_scores,
                boxes_xyxy=boxes,
                reg_epistemic=reg_u,
                cls_epistemic=cls_u,
                score_threshold=args.score_threshold,
                nms_iou_threshold=args.nms_iou_threshold,
                pre_nms_topk=args.pre_nms_topk,
                max_dets=args.max_dets,
            )

            gt_annots = dataset.load_annotations(idx)
            match_detections_to_gt(detections, gt_annots, iou_threshold=args.match_iou_threshold)

            image_id = int(dataset.image_ids[idx])
            img_info = dataset.coco.loadImgs(image_id)[0]
            file_name = img_info["file_name"]

            tp_count = int(sum(int(d.get("is_tp", 0)) for d in detections))
            fp_count = int(len(detections) - tp_count)

            image_total_u = [
                float(d["total_epistemic"])
                for d in detections
                if math.isfinite(float(d["total_epistemic"]))
            ]
            image_reg_u = [
                float(d["reg_epistemic"])
                for d in detections
                if math.isfinite(float(d["reg_epistemic"]))
            ]
            image_cls_u = [
                float(d["cls_epistemic"])
                for d in detections
                if math.isfinite(float(d["cls_epistemic"]))
            ]

            image_records.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "num_detections": len(detections),
                    "tp_count": tp_count,
                    "fp_count": fp_count,
                    "total_epistemic_mean": float(np.mean(image_total_u)) if image_total_u else float("nan"),
                    "reg_epistemic_mean": float(np.mean(image_reg_u)) if image_reg_u else float("nan"),
                    "cls_epistemic_mean": float(np.mean(image_cls_u)) if image_cls_u else float("nan"),
                }
            )

            for det in detections:
                label = int(det["label"])
                detection_records.append(
                    {
                        "image_id": image_id,
                        "file_name": file_name,
                        "label": label,
                        "label_name": dataset.labels.get(label, str(label)),
                        "score": float(det["score"]),
                        "reg_epistemic": float(det["reg_epistemic"]),
                        "cls_epistemic": float(det["cls_epistemic"]),
                        "total_epistemic": float(det["total_epistemic"]),
                        "is_tp": int(det.get("is_tp", 0)),
                        "matched_iou": float(det.get("matched_iou", 0.0)),
                        "x1": float(det["box_xyxy"][0]),
                        "y1": float(det["box_xyxy"][1]),
                        "x2": float(det["box_xyxy"][2]),
                        "y2": float(det["box_xyxy"][3]),
                    }
                )

            if saved_visuals < int(args.visualize_images) and len(detections) > 0:
                vis_name = f"{idx:06d}_img{image_id}.jpg"
                save_visualization(
                    dataset=dataset,
                    idx=idx,
                    detections=detections,
                    gt_annots=gt_annots,
                    output_path=vis_dir / vis_name,
                    max_draw_dets=args.max_draw_dets,
                )
                saved_visuals += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == len(dataset):
                print(f"Processed {idx + 1}/{len(dataset)} images")

    summary = summarize_detections(detection_records, num_images=len(dataset))

    detection_csv = out_dir / "detection_uncertainty.csv"
    image_csv = out_dir / "image_uncertainty.csv"
    summary_json = out_dir / "summary.json"

    write_csv(
        detection_csv,
        detection_records,
        fieldnames=[
            "image_id",
            "file_name",
            "label",
            "label_name",
            "score",
            "reg_epistemic",
            "cls_epistemic",
            "total_epistemic",
            "is_tp",
            "matched_iou",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    )
    write_csv(
        image_csv,
        image_records,
        fieldnames=[
            "image_id",
            "file_name",
            "num_detections",
            "tp_count",
            "fp_count",
            "total_epistemic_mean",
            "reg_epistemic_mean",
            "cls_epistemic_mean",
        ],
    )
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nUncertainty evaluation complete.")
    print(f"Detection CSV: {detection_csv}")
    print(f"Image CSV:     {image_csv}")
    print(f"Summary JSON:  {summary_json}")
    print(f"Visuals:       {vis_dir} (saved {saved_visuals} images)")
    if "total_epistemic_mean" in summary:
        print(
            "Total epistemic mean: "
            f"{summary['total_epistemic_mean']:.4f} | "
            f"TP mean: {summary.get('tp_total_epistemic_mean', float('nan')):.4f} | "
            f"FP mean: {summary.get('fp_total_epistemic_mean', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Normalizer, Resizer


COORD_NAMES = ["x1", "y1", "x2", "y2"]
DELTA_COORD_NAMES = ["dx", "dy", "dw", "dh"]
MODEL_BIN_CENTERS = np.array([0.1667, 0.5, 0.8333], dtype=np.float32)
ANNOT_BIN_CENTERS = np.array([0.0, 0.5, 1.0], dtype=np.float32)
TARGET_EPS = 1e-3
REG_STD = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)


def split_minival_ids(image_ids: Sequence[int], minival_size: int, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    minival_ids = ids[:minival_size]
    trainval_ids = ids[minival_size:]
    return trainval_ids, minival_ids


def build_model(args, num_classes: int):
    kwargs = dict(
        num_classes=num_classes,
        pretrained=False,
        use_dirichlet=True,
        use_random_set=args.model_variant == "dirichlet_randomset",
        random_set_betp_loss=args.random_set_betp_loss,
        random_set_path=args.random_set_path,
        random_set_alpha=args.random_set_alpha,
        random_set_beta=args.random_set_beta,
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
    raise ValueError("Unsupported model depth. Choose 18, 34, 50, 101, or 152.")


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


def run_backbone(base_model, img_batch: torch.Tensor):
    x = base_model.conv1(img_batch)
    x = base_model.bn1(x)
    x = base_model.relu(x)
    x = base_model.maxpool(x)

    x1 = base_model.layer1(x)
    x2 = base_model.layer2(x1)
    x3 = base_model.layer3(x2)
    x4 = base_model.layer4(x3)

    features = base_model.fpn([x2, x3, x4])
    reg_raw = torch.cat([base_model.regressionModel(feature) for feature in features], dim=1)
    cls_raw = torch.cat([base_model.classificationModel(feature) for feature in features], dim=1)
    anchors = base_model.anchors(img_batch)

    reg_deltas = base_model.regressionModel.alphas_to_deltas(reg_raw, img_batch.shape[0])
    if base_model.use_random_set:
        cls_scores = base_model.classificationModel.beliefs_to_label_scores(cls_raw)
    else:
        cls_scores = cls_raw

    pred_boxes = base_model.regressBoxes(anchors, reg_deltas)
    pred_boxes = base_model.clipBoxes(pred_boxes, img_batch)
    return reg_raw, cls_scores, pred_boxes, anchors


def box_iou_np(one_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(one_box[0], boxes[:, 0])
    y1 = np.maximum(one_box[1], boxes[:, 1])
    x2 = np.minimum(one_box[2], boxes[:, 2])
    y2 = np.minimum(one_box[3], boxes[:, 3])

    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area_one = max(0.0, one_box[2] - one_box[0]) * max(0.0, one_box[3] - one_box[1])
    area_many = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes[:, 3] - boxes[:, 1], a_min=0.0, a_max=None
    )
    union = np.clip(area_one + area_many - inter, a_min=1e-8, a_max=None)
    return inter / union


def alpha_to_probs(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float64), 1e-8, None)
    s = np.clip(alpha.sum(axis=-1, keepdims=True), 1e-8, None)
    return alpha / s


def probs_to_simplex_xy(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = probs[..., 1] + 0.5 * probs[..., 2]
    y = probs[..., 2] * (math.sqrt(3.0) / 2.0)
    return x, y


def draw_simplex(ax, title: str):
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, math.sqrt(3.0) / 2.0])
    tri = np.vstack([v0, v1, v2, v0])
    ax.plot(tri[:, 0], tri[:, 1], color="black", linewidth=1.2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.95)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    ax.text(v0[0] - 0.03, v0[1] - 0.03, "bin0", fontsize=8)
    ax.text(v1[0] + 0.01, v1[1] - 0.03, "bin1", fontsize=8)
    ax.text(v2[0] - 0.03, v2[1] + 0.02, "bin2", fontsize=8)


def sample_dirichlet(alpha: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    a = np.clip(alpha.astype(np.float64), 1e-6, 1e5)
    if not np.all(np.isfinite(a)) or a.sum() <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    return rng.dirichlet(a, size=int(n_samples)).astype(np.float32)


def norm_to_target_alphas_np(
    norm_anchors_4: np.ndarray,
    target_concentration: float,
    bin_centers: np.ndarray,
) -> np.ndarray:
    t = np.clip(norm_anchors_4, 0.0, 1.0)
    idx = np.searchsorted(bin_centers, t, side="left")
    lower = np.clip(idx - 1, 0, 2)
    upper = np.clip(idx, 0, 2)

    lower_bins = bin_centers[lower]
    upper_bins = bin_centers[upper]
    denom = np.where(upper_bins > lower_bins, upper_bins - lower_bins, 1.0)
    w_upper = np.where(upper_bins > lower_bins, (t - lower_bins) / denom, 0.0)
    w_lower = 1.0 - w_upper

    probs = np.zeros(t.shape + (3,), dtype=np.float32)
    flat_probs = probs.reshape(-1, 3)
    flat_lower = lower.reshape(-1)
    flat_upper = upper.reshape(-1)
    flat_wl = w_lower.reshape(-1)
    flat_wu = w_upper.reshape(-1)
    flat_idx = np.arange(flat_probs.shape[0])

    np.add.at(flat_probs, (flat_idx, flat_lower), flat_wl.astype(np.float32))
    np.add.at(flat_probs, (flat_idx, flat_upper), flat_wu.astype(np.float32))

    alphas = flat_probs.reshape(probs.shape) * float(target_concentration) + TARGET_EPS
    return alphas.astype(np.float32)


def norm_to_interp_probs_np(norm_vals: np.ndarray, bin_centers: np.ndarray) -> np.ndarray:
    t = np.clip(norm_vals.astype(np.float32), 0.0, 1.0)
    idx = np.searchsorted(bin_centers, t, side="left")
    lower = np.clip(idx - 1, 0, 2)
    upper = np.clip(idx, 0, 2)

    lower_bins = bin_centers[lower]
    upper_bins = bin_centers[upper]
    denom = np.where(upper_bins > lower_bins, upper_bins - lower_bins, 1.0)
    w_upper = np.where(upper_bins > lower_bins, (t - lower_bins) / denom, 0.0)
    w_lower = 1.0 - w_upper

    probs = np.zeros(t.shape + (3,), dtype=np.float32)
    flat_probs = probs.reshape(-1, 3)
    flat_lower = lower.reshape(-1)
    flat_upper = upper.reshape(-1)
    flat_wl = w_lower.reshape(-1).astype(np.float32)
    flat_wu = w_upper.reshape(-1).astype(np.float32)
    flat_idx = np.arange(flat_probs.shape[0])

    np.add.at(flat_probs, (flat_idx, flat_lower), flat_wl)
    np.add.at(flat_probs, (flat_idx, flat_upper), flat_wu)
    return flat_probs.reshape(probs.shape)


def decode_boxes_from_deltas_np(anchors_n4: np.ndarray, deltas_n4: np.ndarray) -> np.ndarray:
    anchors = anchors_n4.astype(np.float32)
    deltas = deltas_n4.astype(np.float32)

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = deltas[:, 0] * REG_STD[0]
    dy = deltas[:, 1] * REG_STD[1]
    dw = deltas[:, 2] * REG_STD[2]
    dh = deltas[:, 3] * REG_STD[3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    return np.stack([x1, y1, x2, y2], axis=1)


def build_pred_corner_prob_clouds(
    pred_alphas_nx4x3: np.ndarray,
    anchors_n4: np.ndarray,
    scale: float,
    img_w: float,
    img_h: float,
    delta_clip: float,
    pred_samples_per_anchor: int,
    rng: np.random.Generator,
    corner_bin_centers: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    clouds = [np.zeros((0, 3), dtype=np.float32) for _ in range(4)]
    uvals = [np.zeros((0,), dtype=np.float32) for _ in range(4)]
    if pred_alphas_nx4x3.shape[0] == 0:
        return clouds, uvals

    cloud_parts = [[] for _ in range(4)]
    u_parts = [[] for _ in range(4)]
    s = max(1, int(pred_samples_per_anchor))

    for i in range(pred_alphas_nx4x3.shape[0]):
        a4 = pred_alphas_nx4x3[i]
        anchor = anchors_n4[i : i + 1]
        strength4 = np.clip(a4.sum(axis=-1), 1e-8, None)
        u4 = np.clip(3.0 / strength4, 0.0, 1.0).astype(np.float32)

        deltas_s4 = np.zeros((s, 4), dtype=np.float32)
        for k in range(s):
            for c in range(4):
                p = sample_dirichlet(a4[c], 1, rng)
                if p.shape[0] == 0:
                    p = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32)
                # Decode predicted alphas using model's internal bin centers.
                norm_val = float(np.dot(p[0], MODEL_BIN_CENTERS))
                deltas_s4[k, c] = (norm_val - 0.5) * (2.0 * float(delta_clip))

        anchors_rep = np.repeat(anchor, s, axis=0)
        boxes_scaled = decode_boxes_from_deltas_np(anchors_rep, deltas_s4)
        boxes_orig = boxes_scaled / max(float(scale), 1e-8)

        x1n = np.clip(boxes_orig[:, 0] / max(float(img_w), 1e-8), 0.0, 1.0)
        y1n = np.clip(boxes_orig[:, 1] / max(float(img_h), 1e-8), 0.0, 1.0)
        x2n = np.clip(boxes_orig[:, 2] / max(float(img_w), 1e-8), 0.0, 1.0)
        y2n = np.clip(boxes_orig[:, 3] / max(float(img_h), 1e-8), 0.0, 1.0)
        norms = [x1n, y1n, x2n, y2n]

        for c in range(4):
            probs_c = norm_to_interp_probs_np(norms[c], bin_centers=corner_bin_centers)
            cloud_parts[c].append(probs_c.astype(np.float32))
            u_parts[c].append(np.full((probs_c.shape[0],), u4[c], dtype=np.float32))

    for c in range(4):
        if cloud_parts[c]:
            clouds[c] = np.concatenate(cloud_parts[c], axis=0)
            uvals[c] = np.concatenate(u_parts[c], axis=0)
    return clouds, uvals


def build_anchor_target_alphas(
    anchors_n4: np.ndarray,
    gt_box_4: np.ndarray,
    delta_clip: float,
    target_concentration: float,
) -> np.ndarray:
    if anchors_n4.shape[0] == 0:
        return np.zeros((0, 4, 3), dtype=np.float32)

    anchors = anchors_n4.astype(np.float32)
    gt = gt_box_4.astype(np.float32)

    aw = np.clip(anchors[:, 2] - anchors[:, 0], 1e-6, None)
    ah = np.clip(anchors[:, 3] - anchors[:, 1], 1e-6, None)
    acx = anchors[:, 0] + 0.5 * aw
    acy = anchors[:, 1] + 0.5 * ah

    gw = max(1.0, float(gt[2] - gt[0]))
    gh = max(1.0, float(gt[3] - gt[1]))
    gcx = float(gt[0] + 0.5 * gw)
    gcy = float(gt[1] + 0.5 * gh)

    dx = (gcx - acx) / aw
    dy = (gcy - acy) / ah
    dw = np.log(np.clip(gw / aw, 1e-8, None))
    dh = np.log(np.clip(gh / ah, 1e-8, None))

    targets = np.stack([dx, dy, dw, dh], axis=1)
    targets = targets / REG_STD[None, :]
    targets = np.clip(targets, -float(delta_clip), float(delta_clip))
    norm = targets / (2.0 * float(delta_clip)) + 0.5
    return norm_to_target_alphas_np(
        norm,
        target_concentration=target_concentration,
        bin_centers=MODEL_BIN_CENTERS,
    )


def parse_image_ids_arg(image_ids_text: str) -> List[int]:
    if not image_ids_text:
        return []
    out = []
    for token in image_ids_text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def find_dataset_indices(
    dataset: CocoDataset,
    requested_image_ids: List[int],
    max_images: int,
    random_images: int,
    image_seed: int,
) -> List[int]:
    if requested_image_ids:
        id_to_idx = {int(image_id): idx for idx, image_id in enumerate(dataset.image_ids)}
        return [id_to_idx[image_id] for image_id in requested_image_ids if image_id in id_to_idx]
    if random_images > 0:
        count = min(int(random_images), len(dataset))
        rng = random.Random(int(image_seed))
        return rng.sample(list(range(len(dataset))), count)
    if max_images > 0:
        return list(range(min(max_images, len(dataset))))
    return list(range(len(dataset)))


def select_anchor_candidates(
    gt_box: np.ndarray,
    gt_label: int,
    candidate_boxes: np.ndarray,
    all_scores: np.ndarray,
    score_threshold: float,
    spread_iou_threshold: float,
    spread_topk: int,
) -> np.ndarray:
    class_scores = all_scores[:, gt_label]
    keep = class_scores >= float(score_threshold)
    keep_idx = np.where(keep)[0]
    if keep_idx.size == 0:
        return np.zeros((0,), dtype=np.int64)

    kept_boxes = candidate_boxes[keep_idx]
    kept_scores = class_scores[keep_idx]
    ious = box_iou_np(gt_box, kept_boxes)
    mask = ious >= float(spread_iou_threshold)
    cand_idx = keep_idx[mask]

    if cand_idx.size == 0:
        rank = np.argsort(-(ious * kept_scores))
        rank = rank[: max(1, min(int(spread_topk), rank.shape[0]))]
        return keep_idx[rank]

    cand_scores = class_scores[cand_idx]
    order = np.argsort(-cand_scores)
    order = order[: min(int(spread_topk), order.shape[0])]
    return cand_idx[order]


def _build_cloud_from_alpha_bank(
    alpha_bank_m3: np.ndarray,
    samples_budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if alpha_bank_m3.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    per = max(1, int(samples_budget) // int(alpha_bank_m3.shape[0]))
    parts = [sample_dirichlet(alpha_bank_m3[i], per, rng) for i in range(alpha_bank_m3.shape[0])]
    parts = [p for p in parts if p.shape[0] > 0]
    if not parts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def plot_gt_vs_pred_simplex(
    image_id: int,
    file_name: str,
    label_name: str,
    gt_obj_idx: int,
    gt_alpha_bank_mx4x3: np.ndarray,
    pred_alphas_nx4x3: np.ndarray,
    pred_scores_n: np.ndarray,
    pred_anchor_count: int,
    coord_names: Sequence[str],
    gt_source_name: str,
    gt_samples: int,
    pred_samples_per_anchor: int,
    rng: np.random.Generator,
    out_path: Path,
    pred_cloud_probs_4: List[np.ndarray] = None,
    pred_cloud_u_4: List[np.ndarray] = None,
):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        (
            f"img={image_id} ({file_name}) | gt_idx={gt_obj_idx} | class={label_name} | "
            f"gt_source={gt_source_name} | anchors={pred_anchor_count}"
        ),
        fontsize=11,
    )

    for c in range(4):
        ax = axes[c]
        draw_simplex(ax, coord_names[c])

        gt_coord_bank = gt_alpha_bank_mx4x3[:, c, :]
        gt_cloud = _build_cloud_from_alpha_bank(gt_coord_bank, gt_samples, rng)
        if gt_cloud.shape[0] > 0:
            gx, gy = probs_to_simplex_xy(gt_cloud)
            ax.scatter(
                gx,
                gy,
                color="crimson",
                s=8,
                alpha=0.25,
                edgecolors="none",
                label="GT samples" if c == 0 else None,
            )

        gt_mean_prob = alpha_to_probs(gt_coord_bank).mean(axis=0)
        gt_x, gt_y = probs_to_simplex_xy(gt_mean_prob)
        pred_cloud_given = (
            pred_cloud_probs_4 is not None
            and pred_cloud_u_4 is not None
            and len(pred_cloud_probs_4) == 4
            and len(pred_cloud_u_4) == 4
        )

        if pred_cloud_given and pred_cloud_probs_4[c].shape[0] > 0:
            pred_cloud = pred_cloud_probs_4[c]
            px, py = probs_to_simplex_xy(pred_cloud)

            ax.scatter(
                px,
                py,
                color="royalblue",
                s=8,
                alpha=0.30,
                edgecolors="none",
                label="Pred samples" if c == 0 else None,
            )

            pred_center = pred_cloud.mean(axis=0)
            cx, cy = probs_to_simplex_xy(pred_center)
            ax.scatter(
                [cx],
                [cy],
                marker="X",
                s=70,
                color="white",
                edgecolors="black",
                linewidths=0.8,
                label="Pred mean" if c == 0 else None,
            )

        elif pred_alphas_nx4x3.shape[0] > 0:
            pred_coord_bank = pred_alphas_nx4x3[:, c, :]
            pred_cloud_list = []
            pred_u_list = []

            for i in range(pred_coord_bank.shape[0]):
                a_i = pred_coord_bank[i]
                s_i = float(np.clip(a_i.sum(), 1e-8, None))
                u_i = float(min(1.0, 3.0 / s_i))
                samples_i = sample_dirichlet(a_i, pred_samples_per_anchor, rng)
                if samples_i.shape[0] == 0:
                    continue
                pred_cloud_list.append(samples_i)
                pred_u_list.append(np.full((samples_i.shape[0],), u_i, dtype=np.float32))

            if pred_cloud_list:
                pred_cloud = np.concatenate(pred_cloud_list, axis=0)
                px, py = probs_to_simplex_xy(pred_cloud)

                ax.scatter(
                    px,
                    py,
                    color="royalblue",
                    s=8,
                    alpha=0.30,
                    edgecolors="none",
                    label="Pred samples" if c == 0 else None,
                )

                pred_prob = alpha_to_probs(pred_coord_bank)
                if pred_scores_n.size > 0 and np.any(np.isfinite(pred_scores_n)):
                    w = np.clip(pred_scores_n.astype(np.float64), 1e-8, None)
                    w = w / np.clip(w.sum(), 1e-8, None)
                    pred_center = (pred_prob * w[:, None]).sum(axis=0)
                else:
                    pred_center = pred_prob.mean(axis=0)
                cx, cy = probs_to_simplex_xy(pred_center)
                ax.scatter(
                    [cx],
                    [cy],
                    marker="X",
                    s=70,
                    color="white",
                    edgecolors="black",
                    linewidths=0.8,
                    label="Pred mean" if c == 0 else None,
                )

        ax.scatter(
            [gt_x],
            [gt_y],
            marker="*",
            s=170,
            color="red",
            edgecolors="black",
            linewidths=0.8,
            label="GT mean" if c == 0 else None,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)), fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def clip_box_xyxy(box_xyxy: np.ndarray, img_w: float, img_h: float) -> np.ndarray:
    max_x = max(float(img_w) - 1.0, 0.0)
    max_y = max(float(img_h) - 1.0, 0.0)
    x1 = float(np.clip(box_xyxy[0], 0.0, max_x))
    y1 = float(np.clip(box_xyxy[1], 0.0, max_y))
    x2 = float(np.clip(box_xyxy[2], 0.0, max_x))
    y2 = float(np.clip(box_xyxy[3], 0.0, max_y))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def save_bbox_visualization(
    dataset: CocoDataset,
    ds_idx: int,
    image_id: int,
    file_name: str,
    gt_draw_items: List[Dict[str, object]],
    pred_draw_items: List[Dict[str, object]],
    output_path: Path,
):
    img_rgb = dataset.load_image(ds_idx)
    img_vis = (np.clip(img_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img_vis)
    draw = ImageDraw.Draw(pil_img)

    for pred in pred_draw_items:
        px1, py1, px2, py2 = [int(round(v)) for v in pred["box_xyxy"]]
        draw.rectangle((px1, py1, px2, py2), outline=(0, 255, 0), width=2)
        pred_label = str(pred["label_name"])
        pred_score = float(pred["score"])
        pred_iou = float(pred["gt_iou"])
        pred_caption = f"Pred:{pred_label} s={pred_score:.2f} IoU={pred_iou:.2f}"
        pred_text_y = max(0, py1 - 14)
        draw.text((px1, pred_text_y), pred_caption, fill=(0, 255, 0))

    for gt in gt_draw_items:
        gx1, gy1, gx2, gy2 = [int(round(v)) for v in gt["box_xyxy"]]
        draw.rectangle((gx1, gy1, gx2, gy2), outline=(255, 0, 0), width=2)
        gt_label = str(gt["label_name"])
        gt_caption = f"GT:{gt_label}"
        gt_text_y = min(max(0, gy2 + 2), max(0, pil_img.height - 14))
        draw.text((gx1, gt_text_y), gt_caption, fill=(255, 0, 0))

    title_text = f"img={image_id} {file_name}"
    draw.text((5, 5), title_text, fill=(255, 255, 255))
    pil_img.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot GT and predicted Dirichlet alpha sample clouds on the 2-simplex."
    )
    parser.add_argument("--coco_path", required=True, help="Path to COCO root")
    parser.add_argument("--model_path", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument(
        "--model_variant",
        choices=["dirichlet_std", "dirichlet_randomset"],
        default="dirichlet_std",
    )
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument(
        "--annotation_path",
        default=None,
        help="COCO annotation JSON with dir_ann dirichlet fields (needed for --gt_alpha_source annotation).",
    )
    parser.add_argument(
        "--gt_alpha_source",
        choices=["anchor_target", "annotation"],
        default="anchor_target",
        help="Use anchor-conditioned GT target alphas (recommended) or raw annotation alphas.",
    )
    parser.add_argument(
        "--flip_annotation_bins",
        action="store_true",
        help="Reverse annotation alpha bin order [a0,a1,a2]->[a2,a1,a0] when using annotation source.",
    )
    parser.add_argument(
        "--candidate_box_space",
        choices=["anchor", "pred"],
        default="anchor",
        help="Anchor matching space for candidate selection around each GT object.",
    )
    parser.add_argument(
        "--display_coord_labels",
        choices=["auto", "xyxy", "delta"],
        default="auto",
        help="Axis labels in plots: auto uses mode-based defaults; xyxy forces x1,y1,x2,y2; delta forces dx,dy,dw,dh.",
    )
    parser.add_argument("--image_ids", default="", help="Comma-separated COCO image IDs (e.g. '42,77,123').")
    parser.add_argument("--max_images", type=int, default=3, help="Used when --image_ids is empty.")
    parser.add_argument(
        "--random_images",
        type=int,
        default=0,
        help="Randomly sample this many images when --image_ids is empty (overrides --max_images when > 0).",
    )
    parser.add_argument("--image_seed", type=int, default=42, help="Seed used for --random_images sampling.")
    parser.add_argument("--gt_per_image", type=int, default=2, help="Max GT objects to plot per image.")
    parser.add_argument("--score_threshold", type=float, default=0.01, help="Class score threshold for candidates.")
    parser.add_argument("--spread_iou_threshold", type=float, default=0.3, help="IoU threshold for candidates.")
    parser.add_argument("--spread_topk", type=int, default=20, help="Top-k candidate anchors per GT.")
    parser.add_argument("--gt_samples", type=int, default=300, help="GT simplex samples per coordinate cloud.")
    parser.add_argument(
        "--pred_samples_per_anchor",
        type=int,
        default=40,
        help="Pred simplex samples per candidate anchor per coordinate.",
    )
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for simplex sampling.")
    parser.add_argument("--output_dir", default="alpha_simplex_eval_output")
    parser.add_argument(
        "--save_bbox_visualizations",
        dest="save_bbox_visualizations",
        action="store_true",
        help="Save per-image overlays with predicted boxes in green and GT boxes in red.",
    )
    parser.add_argument(
        "--no_save_bbox_visualizations",
        dest="save_bbox_visualizations",
        action="store_false",
        help="Disable per-image overlay export.",
    )
    parser.set_defaults(save_bbox_visualizations=True)

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
    rng = np.random.default_rng(args.sample_seed)
    print(f"CUDA available: {torch.cuda.is_available()} | device={device}")

    if args.gt_alpha_source == "annotation" and not args.annotation_path:
        raise ValueError("--annotation_path is required when --gt_alpha_source annotation")

    dataset_use_dirichlet = args.gt_alpha_source == "annotation"
    dataset = CocoDataset(
        args.coco_path,
        set_name="val2017",
        transform=transforms.Compose([Normalizer(), Resizer()]),
        use_dirichlet=dataset_use_dirichlet,
        annotation_path=args.annotation_path,
    )

    if args.coco_minival_split:
        _, minival_ids = split_minival_ids(dataset.image_ids, args.minival_size, args.minival_seed)
        dataset.image_ids = minival_ids
        print(f"Using minival subset: {len(dataset.image_ids)} images")

    requested_ids = parse_image_ids_arg(args.image_ids)
    indices = find_dataset_indices(
        dataset,
        requested_ids,
        args.max_images,
        args.random_images,
        args.image_seed,
    )
    if not indices:
        raise ValueError("No valid images selected. Check --image_ids or dataset split.")
    print(f"Selected images: {len(indices)}")
    if not requested_ids and args.random_images > 0:
        chosen_ids = [int(dataset.image_ids[i]) for i in indices]
        print(f"Random image_ids (seed={args.image_seed}): {chosen_ids}")

    net = build_model(args, num_classes=dataset.num_classes())
    missing, unexpected = load_checkpoint(net, args.model_path)
    print(f"Loaded checkpoint: {args.model_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    net = net.to(device)
    net.eval()
    net.freeze_bn()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir = out_dir / "bbox_visualizations"
    if args.save_bbox_visualizations:
        bbox_dir.mkdir(parents=True, exist_ok=True)

    produced = 0
    bbox_saved = 0
    skipped_no_gt = 0

    with torch.no_grad():
        for ds_idx in indices:
            sample = dataset[ds_idx]
            img = sample["img"]
            ann = sample["annot"]
            scale = sample.get("scale", 1.0)
            if isinstance(scale, (list, tuple)):
                scale = scale[0]
            if isinstance(scale, torch.Tensor):
                scale = float(scale.item())
            scale = float(scale)

            if isinstance(ann, torch.Tensor):
                ann = ann.cpu().numpy()
            ann = np.asarray(ann, dtype=np.float32)
            if ann.size == 0 or ann.shape[0] == 0:
                skipped_no_gt += 1
                print(f"Skipping image idx={ds_idx}: no GT boxes.")
                continue

            if isinstance(img, torch.Tensor):
                img_batch = img.permute(2, 0, 1).unsqueeze(0).to(device).float()
            else:
                img_batch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float()

            reg_raw, cls_scores, pred_boxes, anchors = run_backbone(net, img_batch)
            reg_raw = reg_raw[0].detach().cpu().numpy()  # [A,12]
            cls_scores = cls_scores[0].detach().cpu().numpy()  # [A,C]
            pred_boxes = pred_boxes[0].detach().cpu().numpy()  # [A,4]
            anchors = anchors[0].detach().cpu().numpy()  # [A,4]

            candidate_boxes = anchors if args.candidate_box_space == "anchor" else pred_boxes

            image_id = int(dataset.image_ids[ds_idx])
            img_info = dataset.coco.loadImgs(image_id)[0]
            file_name = img_info["file_name"]
            img_w = float(img_info["width"])
            img_h = float(img_info["height"])
            inv_scale = 1.0 / max(scale, 1e-8)
            gt_draw_items: List[Dict[str, object]] = []
            pred_draw_items: List[Dict[str, object]] = []

            gt_count = min(int(args.gt_per_image), ann.shape[0])
            for gt_idx in range(gt_count):
                gt_row = ann[gt_idx]
                gt_box = gt_row[:4]
                gt_label = int(gt_row[4])
                label_name = dataset.labels.get(gt_label, str(gt_label))
                gt_box_orig = clip_box_xyxy(gt_box * inv_scale, img_w=img_w, img_h=img_h)
                gt_draw_items.append(
                    {
                        "box_xyxy": gt_box_orig,
                        "label_name": label_name,
                    }
                )

                cand_anchor_idx = select_anchor_candidates(
                    gt_box=gt_box,
                    gt_label=gt_label,
                    candidate_boxes=candidate_boxes,
                    all_scores=cls_scores,
                    score_threshold=args.score_threshold,
                    spread_iou_threshold=args.spread_iou_threshold,
                    spread_topk=args.spread_topk,
                )

                pred_alphas = np.zeros((0, 4, 3), dtype=np.float32)
                pred_scores = np.zeros((0,), dtype=np.float32)
                if cand_anchor_idx.size > 0:
                    pred_alphas = reg_raw[cand_anchor_idx].reshape(-1, 4, 3)
                    pred_scores = cls_scores[cand_anchor_idx, gt_label]
                    best_idx = int(cand_anchor_idx[0])
                    pred_box_orig = clip_box_xyxy(pred_boxes[best_idx] * inv_scale, img_w=img_w, img_h=img_h)
                    pred_iou = float(box_iou_np(gt_box_orig, pred_box_orig[None, :])[0])
                    pred_draw_items.append(
                        {
                            "box_xyxy": pred_box_orig,
                            "label_name": label_name,
                            "score": float(cls_scores[best_idx, gt_label]),
                            "gt_iou": pred_iou,
                        }
                    )

                pred_cloud_probs_4 = None
                pred_cloud_u_4 = None

                if args.gt_alpha_source == "annotation":
                    if gt_row.shape[0] < 17:
                        print(f"Skipping image {image_id} gt_idx={gt_idx}: annotation has no alpha columns.")
                        continue
                    gt_alpha_bank = gt_row[5:17].reshape(1, 4, 3).astype(np.float32)
                    if args.flip_annotation_bins:
                        gt_alpha_bank = gt_alpha_bank[:, :, ::-1]
                    gt_source_name = "annotation_flipped" if args.flip_annotation_bins else "annotation"
                    coord_names = COORD_NAMES

                    if cand_anchor_idx.size > 0:
                        pred_cloud_probs_4, pred_cloud_u_4 = build_pred_corner_prob_clouds(
                            pred_alphas_nx4x3=pred_alphas,
                            anchors_n4=anchors[cand_anchor_idx],
                            scale=scale,
                            img_w=img_w,
                            img_h=img_h,
                            delta_clip=args.dirichlet_delta_clip,
                            pred_samples_per_anchor=args.pred_samples_per_anchor,
                            rng=rng,
                            corner_bin_centers=ANNOT_BIN_CENTERS,
                        )
                else:
                    anchor_for_gt = anchors[cand_anchor_idx] if cand_anchor_idx.size > 0 else anchors[:1]
                    gt_alpha_bank = build_anchor_target_alphas(
                        anchors_n4=anchor_for_gt,
                        gt_box_4=gt_box,
                        delta_clip=args.dirichlet_delta_clip,
                        target_concentration=args.dirichlet_target_concentration,
                    )
                    gt_source_name = "anchor_target"
                    coord_names = DELTA_COORD_NAMES

                if args.display_coord_labels == "xyxy":
                    coord_names = COORD_NAMES
                elif args.display_coord_labels == "delta":
                    coord_names = DELTA_COORD_NAMES

                out_name = f"img{image_id}_gt{gt_idx}_{label_name}_{gt_source_name}.png"
                out_path = out_dir / out_name
                plot_gt_vs_pred_simplex(
                    image_id=image_id,
                    file_name=file_name,
                    label_name=label_name,
                    gt_obj_idx=gt_idx,
                    gt_alpha_bank_mx4x3=gt_alpha_bank,
                    pred_alphas_nx4x3=pred_alphas,
                    pred_scores_n=pred_scores,
                    pred_anchor_count=int(pred_alphas.shape[0]),
                    coord_names=coord_names,
                    gt_source_name=gt_source_name,
                    gt_samples=args.gt_samples,
                    pred_samples_per_anchor=args.pred_samples_per_anchor,
                    rng=rng,
                    out_path=out_path,
                    pred_cloud_probs_4=pred_cloud_probs_4,
                    pred_cloud_u_4=pred_cloud_u_4,
                )
                produced += 1
                print(
                    f"Saved: {out_path} | gt={label_name} | candidates={pred_alphas.shape[0]} | gt_source={gt_source_name}"
                )

            if args.save_bbox_visualizations and gt_draw_items:
                vis_name = f"img{image_id}_{Path(file_name).stem}_bbox.jpg"
                vis_path = bbox_dir / vis_name
                save_bbox_visualization(
                    dataset=dataset,
                    ds_idx=ds_idx,
                    image_id=image_id,
                    file_name=file_name,
                    gt_draw_items=gt_draw_items,
                    pred_draw_items=pred_draw_items,
                    output_path=vis_path,
                )
                bbox_saved += 1
                print(
                    f"Saved bbox overlay: {vis_path} | gt_boxes={len(gt_draw_items)} | pred_boxes={len(pred_draw_items)}"
                )

    print("\nDone.")
    print(f"Output dir: {out_dir}")
    print(f"Plots produced: {produced}")
    if args.save_bbox_visualizations:
        print(f"BBox overlays saved: {bbox_saved}")
        print(f"BBox overlay dir: {bbox_dir}")
    print(f"Skipped images without GT boxes: {skipped_no_gt}")


if __name__ == "__main__":
    main()

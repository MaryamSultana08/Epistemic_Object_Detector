
from pycocotools.cocoeval import COCOeval
import json
import torch
from torchvision.ops import nms


def evaluate_coco(
    dataset,
    model,
    threshold=0.05,
    write_results=True,
    max_dets=100,
    data_loader=None,
    debug_dump_path=None,
    debug_topk=50,
    debug_max_images=None,
    nms_iou_threshold=0.5,
    pre_nms_topk=1000,
    skip_eval_postprocess=False,
):
    """
    COCO evaluation with per-class NMS (RetinaNet-style).

    Assumes model(img_batch) returns (scores, labels, boxes) for a single image:
      - scores: (N,) float tensor, higher = more confident
      - labels: (N,) int tensor, 0..num_classes-1
      - boxes:  (N,4) float tensor in xyxy coords on the resized image

    This function:
      - rescales boxes back to original image size using data['scale']
      - applies score threshold
      - optional pre-NMS top-k pruning (per image, global)
      - performs PER-CLASS NMS
      - keeps top max_dets overall
      - converts boxes to COCO xywh
      - runs COCOeval
    """

    model.eval()

    results = []
    image_ids = []
    max_scores = []

    # Choose iterator
    if data_loader is None:
        data_iter = enumerate(dataset)
        total = len(dataset)
        get_image_id = lambda idx, data: dataset.image_ids[idx]
    else:
        data_iter = enumerate(data_loader)
        total = len(data_loader)

        # When using DataLoader+collater, dataset indexing != batch index.
        # collater typically keeps image_ids in data if implemented; if not, fall back.
        def get_image_id(idx, data):
            if isinstance(data, dict) and "image_id" in data:
                # allow scalar or list
                iid = data["image_id"]
                if isinstance(iid, (list, tuple)):
                    return int(iid[0])
                if isinstance(iid, torch.Tensor):
                    return int(iid.item())
                return int(iid)
            # fallback (works only if loader iterates in-order without shuffling)
            return int(dataset.image_ids[idx])

    debug_dump = open(debug_dump_path, "w") if debug_dump_path else None

    with torch.no_grad():
        for index, data in data_iter:
            # scale may be float or [float] or tensor
            scale = data.get("scale", 1.0)
            if isinstance(scale, (list, tuple)):
                scale = scale[0]
            if isinstance(scale, torch.Tensor):
                scale = float(scale.item())
            scale = float(scale)

            img = data["img"]

            # Ensure (B,C,H,W)
            if isinstance(img, torch.Tensor) and img.dim() == 4:
                img_batch = img
            else:
                # dataset item typically returns HWC
                img_batch = img.permute(2, 0, 1).unsqueeze(0)

            if torch.cuda.is_available():
                scores, labels, boxes = model(img_batch.cuda().float())
            else:
                scores, labels, boxes = model(img_batch.float())

            # Move to CPU, flatten defensively
            scores = scores.detach().cpu().reshape(-1)
            labels = labels.detach().cpu().reshape(-1)

            if boxes.dim() == 3 and boxes.shape[0] == 1:
                boxes = boxes[0]
            boxes = boxes.detach().cpu()

            img_id = int(get_image_id(index, data))
            image_ids.append(img_id)

            # Handle empty predictions robustly
            if boxes.numel() == 0 or scores.numel() == 0:
                print(f"{index + 1}/{total}", end="\r")
                if debug_max_images is not None and (index + 1) >= int(debug_max_images):
                    break
                continue

            if scores.numel() > 0:
                max_scores.append(float(scores.max().item()))

            # Ensure boxes shape (N,4)
            if boxes.dim() != 2 or boxes.shape[1] != 4:
                raise ValueError(f"Expected boxes as (N,4) or (1,N,4), got shape {tuple(boxes.shape)}")

            # Rescale boxes back to original image size
            boxes = boxes / scale

            # Initial score threshold
            keep_mask = scores >= float(threshold)
            scores = scores[keep_mask]
            labels = labels[keep_mask]
            boxes = boxes[keep_mask]

            if scores.numel() == 0:
                # nothing above threshold for this image
                if debug_dump is not None:
                    rec = {
                        "image_id": img_id,
                        "num_preds": 0,
                        "max_score": None,
                        "topk_scores": [],
                        "topk_labels": [],
                        "topk_boxes_xyxy": [],
                        "threshold": float(threshold),
                        "kept_after_threshold": 0,
                        "kept_after_nms": 0,
                    }
                    debug_dump.write(json.dumps(rec) + "\n")

                print(f"{index + 1}/{total}", end="\r")
                if debug_max_images is not None and (index + 1) >= int(debug_max_images):
                    break
                continue

            if skip_eval_postprocess:
                # Model already returns NMS-filtered detections.
                if max_dets is not None and scores.numel() > int(max_dets):
                    top_vals, top_idx = torch.topk(scores, k=int(max_dets), largest=True, sorted=True)
                    scores_nms = top_vals
                    labels_nms = labels[top_idx]
                    boxes_nms = boxes[top_idx]
                else:
                    top_vals, top_idx = torch.sort(scores, descending=True)
                    scores_nms = top_vals
                    labels_nms = labels[top_idx]
                    boxes_nms = boxes[top_idx]
            else:
                # Optional global pre-NMS top-k to keep NMS fast (common in RetinaNet)
                if pre_nms_topk is not None and scores.numel() > int(pre_nms_topk):
                    top_vals, top_idx = torch.topk(scores, k=int(pre_nms_topk), largest=True, sorted=True)
                    scores = top_vals
                    labels = labels[top_idx]
                    boxes = boxes[top_idx]

                # PER-CLASS NMS
                kept_scores = []
                kept_labels = []
                kept_boxes = []

                unique_labels = labels.unique()
                for cls in unique_labels.tolist():
                    cls_mask = labels == cls
                    cls_scores = scores[cls_mask]
                    cls_boxes = boxes[cls_mask]

                    if cls_scores.numel() == 0:
                        continue

                    # NMS expects xyxy
                    cls_keep = nms(cls_boxes, cls_scores, float(nms_iou_threshold))

                    if cls_keep.numel() == 0:
                        continue

                    kept_scores.append(cls_scores[cls_keep])
                    kept_labels.append(torch.full((cls_keep.numel(),), int(cls), dtype=torch.long))
                    kept_boxes.append(cls_boxes[cls_keep])

                if not kept_scores:
                    if debug_dump is not None:
                        rec = {
                            "image_id": img_id,
                            "num_preds": int(scores.numel()),
                            "max_score": float(scores.max().item()) if scores.numel() else None,
                            "topk_scores": [float(x) for x in scores[: min(int(debug_topk), scores.numel())]],
                            "topk_labels": [int(x) for x in labels[: min(int(debug_topk), labels.numel())]],
                            "topk_boxes_xyxy": [b.tolist() for b in boxes[: min(int(debug_topk), boxes.shape[0])]],
                            "threshold": float(threshold),
                            "kept_after_threshold": int(scores.numel()),
                            "kept_after_nms": 0,
                        }
                        debug_dump.write(json.dumps(rec) + "\n")

                    print(f"{index + 1}/{total}", end="\r")
                    if debug_max_images is not None and (index + 1) >= int(debug_max_images):
                        break
                    continue

                scores_nms = torch.cat(kept_scores, dim=0)
                labels_nms = torch.cat(kept_labels, dim=0)
                boxes_nms = torch.cat(kept_boxes, dim=0)

                # Keep top max_dets overall after per-class NMS
                if max_dets is not None and scores_nms.numel() > int(max_dets):
                    top_vals, top_idx = torch.topk(scores_nms, k=int(max_dets), largest=True, sorted=True)
                    scores_nms = top_vals
                    labels_nms = labels_nms[top_idx]
                    boxes_nms = boxes_nms[top_idx]
                else:
                    # still sort for consistent output
                    top_vals, top_idx = torch.sort(scores_nms, descending=True)
                    scores_nms = top_vals
                    labels_nms = labels_nms[top_idx]
                    boxes_nms = boxes_nms[top_idx]

            # Debug dump (after NMS)
            if debug_dump is not None:
                topk = min(int(debug_topk), scores_nms.numel())
                rec = {
                    "image_id": img_id,
                    "num_preds": int(scores.numel()),
                    "max_score": float(scores_nms[0].item()) if scores_nms.numel() else None,
                    "topk_scores": [float(x) for x in scores_nms[:topk]],
                    "topk_labels": [int(x) for x in labels_nms[:topk]],
                    "topk_boxes_xyxy": [b.tolist() for b in boxes_nms[:topk]],
                    "threshold": float(threshold),
                    "kept_after_threshold": int(scores.numel()),
                    "kept_after_nms": int(scores_nms.numel()),
                    "nms_iou_threshold": None if skip_eval_postprocess else float(nms_iou_threshold),
                    "pre_nms_topk": None if skip_eval_postprocess else (int(pre_nms_topk) if pre_nms_topk is not None else None),
                    "skip_eval_postprocess": bool(skip_eval_postprocess),
                }
                debug_dump.write(json.dumps(rec) + "\n")

            # Convert boxes to COCO xywh
            boxes_xywh = boxes_nms.clone()
            boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]  # w
            boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]  # h

            # Add detections
            for score, label, box in zip(scores_nms, labels_nms, boxes_xywh):
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": dataset.label_to_coco_label(int(label)),
                        "score": float(score.item()),
                        "bbox": box.tolist(),
                    }
                )

            print(f"{index + 1}/{total}", end="\r")

            if debug_max_images is not None and (index + 1) >= int(debug_max_images):
                break

    if debug_dump is not None:
        debug_dump.close()

    if not results:
        print("\nNo detections above threshold; skipping COCO eval.")
        if max_scores:
            print("Max score observed: {:.6f}".format(max(max_scores)))
        return

    coco_true = dataset.coco

    if write_results:
        out_path = "{}_bbox_results.json".format(dataset.set_name)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        coco_pred = coco_true.loadRes(out_path)
    else:
        coco_pred = coco_true.loadRes(results)

    coco_eval = COCOeval(coco_true, coco_pred, "bbox")
    coco_eval.params.imgIds = image_ids

    print("\nRunning COCO evaluation...")
    coco_eval.evaluate()
    print("Accumulating evaluation results...")
    coco_eval.accumulate()
    print("Summarizing evaluation results...")
    coco_eval.summarize()
    print("COCO evaluation complete.")





# original code
# from pycocotools.cocoeval import COCOeval
# import json
# import torch


# def evaluate_coco(
#     dataset,
#     model,
#     threshold=0.05,
#     write_results=True,
#     max_dets=100,
#     data_loader=None,
#     debug_dump_path=None,
#     debug_topk=50,
#     debug_max_images=20,
# ):
    
#     model.eval()
    
#     with torch.no_grad():

#         # start collecting results
#         results = []
#         image_ids = []

#         max_scores = []
#         if data_loader is None:
#             data_iter = enumerate(dataset)
#         else:
#             data_iter = enumerate(data_loader)

#         debug_dump = None
#         if debug_dump_path:
#             debug_dump = open(debug_dump_path, 'w')

#         for index, data in data_iter:
#             scale = data['scale']
#             if isinstance(scale, (list, tuple)):
#                 scale = scale[0]

#             # run network
#             img = data['img']
#             if isinstance(img, torch.Tensor) and img.dim() == 4:
#                 img_batch = img
#             else:
#                 img_batch = img.permute(2, 0, 1).unsqueeze(dim=0)

#             if torch.cuda.is_available():
#                 scores, labels, boxes = model(img_batch.cuda().float())
#             else:
#                 scores, labels, boxes = model(img_batch.float())
#             scores = scores.cpu()
#             labels = labels.cpu()
#             boxes  = boxes.cpu()
#             if scores.numel() > 0:
#                 max_scores.append(float(scores.max()))

#             # correct boxes for image scale
#             boxes /= scale

#             if debug_dump is not None:
#                 # Dump per-image top-k predictions for quick inspection.
#                 if scores.numel() > 0:
#                     topk = min(int(debug_topk), scores.numel())
#                     topk_scores, topk_idx = torch.topk(scores, k=topk)
#                     topk_labels = labels[topk_idx]
#                     topk_boxes = boxes[topk_idx]
#                     img_id = dataset.image_ids[index]
#                     rec = {
#                         "image_id": int(img_id),
#                         "num_preds": int(scores.numel()),
#                         "topk_scores": [float(x) for x in topk_scores],
#                         "topk_labels": [int(x) for x in topk_labels],
#                         "topk_boxes_xywh": [b.tolist() for b in topk_boxes],
#                     }
#                     debug_dump.write(json.dumps(rec) + "\n")
#                 if debug_max_images is not None and (index + 1) >= int(debug_max_images):
#                     break

#             if boxes.shape[0] > 0:
#                 # change to (x, y, w, h) (MS COCO standard)
#                 boxes[:, 2] -= boxes[:, 0]
#                 boxes[:, 3] -= boxes[:, 1]

#                 # compute predicted labels and scores
#                 #for box, score, label in zip(boxes[0], scores[0], labels[0]):
#                 # Limit to top-N detections by score to match COCO maxDets.
#                 if max_dets is not None and boxes.shape[0] > max_dets:
#                     topk = torch.topk(scores, k=max_dets)
#                     keep = topk.indices
#                 else:
#                     keep = torch.arange(boxes.shape[0])

#                 for box_id in keep:
#                     score = float(scores[box_id])
#                     label = int(labels[box_id])
#                     box = boxes[box_id, :]

#                     # scores are sorted, so we can break
#                     if score < threshold:
#                         break

#                     # append detection for each positively labeled class
#                     image_result = {
#                         'image_id'    : dataset.image_ids[index],
#                         'category_id' : dataset.label_to_coco_label(label),
#                         'score'       : float(score),
#                         'bbox'        : box.tolist(),
#                     }

#                     # append detection to results
#                     results.append(image_result)

#             # append image to list of processed images
#             image_ids.append(dataset.image_ids[index])

#             # print progress
#             print('{}/{}'.format(index, len(dataset)), end='\r')

#         if debug_dump is not None:
#             debug_dump.close()

#         if not len(results):
#             if max_scores:
#                 max_score = max(max_scores)
#                 print('No detections above threshold; skipping COCO eval.')
#                 print('Max score observed: {:.6f}'.format(max_score))
#             else:
#                 print('No detections above threshold; skipping COCO eval.')
#             return

#         # load results in COCO evaluation tool
#         coco_true = dataset.coco
#         if write_results:
#             json.dump(
#                 results,
#                 open('{}_bbox_results.json'.format(dataset.set_name), 'w'),
#                 indent=4,
#             )
#             coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))
#         else:
#             coco_pred = coco_true.loadRes(results)

#         # run COCO evaluation
#         coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
#         coco_eval.params.imgIds = image_ids
#         print('Running COCO evaluation...')
#         coco_eval.evaluate()
#         print('Accumulating evaluation results...')
#         coco_eval.accumulate()
#         print('Summarizing evaluation results...')
#         coco_eval.summarize()
#         print('COCO evaluation complete.')

#         model.train()

#         return

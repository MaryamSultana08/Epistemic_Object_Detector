#!/usr/bin/env python3
"""
Convert ROAD dataset annotations to COCO detection format.

ROAD reference:
https://github.com/gurkirt/road-dataset

Expected input is ROAD's `road_trainval_v1.0.json` structure with:
- top-level `db` dict
- each video entry containing `frames`
- each frame containing `annos`
- each annotation containing a `box` in [x1, y1, x2, y2]
  (normalized to [0, 1] in official ROAD annotations)

This converter is frame-level only (not tube-level tracking evaluation).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PIL import Image


Number = Union[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ROAD annotations to COCO detection JSON.")
    parser.add_argument("--road_json", required=True, help="Path to ROAD annotation JSON (e.g., road_trainval_v1.0.json).")
    parser.add_argument("--output_json", required=True, help="Output path for COCO-style JSON.")
    parser.add_argument(
        "--images_root",
        default="",
        help="Optional path to ROAD rgb-images root for file validation and size fallback.",
    )
    parser.add_argument(
        "--label_type",
        choices=["agent", "action", "loc"],
        default="agent",
        help="Which ROAD label type to export as COCO categories.",
    )
    parser.add_argument(
        "--label_map_key",
        default="",
        help=(
            "Optional override for label map key in ROAD JSON "
            "(e.g., all_agent_labels, agent_labels, action_labels, loc_labels)."
        ),
    )
    parser.add_argument(
        "--multi_label_policy",
        choices=["first", "duplicate"],
        default="first",
        help=(
            "How to handle annotations with multiple labels in selected label_type. "
            "'first' keeps one class; 'duplicate' emits one COCO annotation per class."
        ),
    )
    parser.add_argument(
        "--box_mode",
        choices=["auto", "normalized", "pixel"],
        default="auto",
        help="Interpretation of ROAD `box` values.",
    )
    parser.add_argument(
        "--include_split_ids",
        default="",
        help=(
            "Comma-separated ROAD split IDs to keep (based on top-level split_ids). "
            "Example: '1,2'. Empty means keep all videos."
        ),
    )
    parser.add_argument(
        "--keep_class_ids",
        default="",
        help="Optional comma-separated raw label IDs to keep (after choosing label_type).",
    )
    parser.add_argument(
        "--keep_class_names",
        default="",
        help="Optional comma-separated label names to keep (after choosing label_type).",
    )
    parser.add_argument(
        "--require_images",
        action="store_true",
        help="If set, skip frames whose image file cannot be found under images_root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on unknown labels or malformed records instead of skipping them.",
    )
    return parser.parse_args()


def parse_csv_set(raw: str) -> set:
    out = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.add(item)
    return out


def parse_csv_int_set(raw: str) -> set:
    out = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def parse_csv_str_set(raw: str) -> set:
    out = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.add(item)
    return out


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_label_map(raw_map: Any) -> Dict[int, str]:
    """
    Accept common ROAD label-map representations and return {id: name}.
    Supports:
    - list: [name0, name1, ...]
    - dict id->name
    - dict name->id
    - dict id->{"name": ...}
    """
    out: Dict[int, str] = {}

    if isinstance(raw_map, list):
        for idx, name in enumerate(raw_map):
            out[int(idx)] = str(name)
        return out

    if not isinstance(raw_map, dict):
        return out

    parsed_key_as_int = 0
    parsed_value_as_int = 0

    for k, v in raw_map.items():
        if to_int(k) is not None:
            parsed_key_as_int += 1
        if to_int(v) is not None:
            parsed_value_as_int += 1

    if parsed_key_as_int >= parsed_value_as_int:
        for k, v in raw_map.items():
            kid = to_int(k)
            if kid is None:
                continue
            if isinstance(v, dict) and "name" in v:
                out[kid] = str(v["name"])
            else:
                out[kid] = str(v)
        return out

    # Inverted map name->id
    for k, v in raw_map.items():
        vid = to_int(v)
        if vid is None:
            continue
        out[vid] = str(k)
    return out


def choose_label_map_key(road: Dict[str, Any], label_type: str, override: str) -> str:
    if override:
        return override

    if label_type == "agent":
        for key in ("all_agent_labels", "agent_labels"):
            if key in road:
                return key
    elif label_type == "action":
        if "action_labels" in road:
            return "action_labels"
    elif label_type == "loc":
        if "loc_labels" in road:
            return "loc_labels"

    raise KeyError(f"Could not find label map key for label_type='{label_type}'.")


def choose_label_ids_key(label_type: str) -> str:
    if label_type == "agent":
        return "agent_ids"
    if label_type == "action":
        return "action_ids"
    if label_type == "loc":
        return "loc_ids"
    raise ValueError(f"Unsupported label_type: {label_type}")


def candidate_frame_names(frame_key: str) -> List[str]:
    candidates = [str(frame_key)]
    frame_int = to_int(frame_key)
    if frame_int is not None:
        for width in (4, 5, 6, 7):
            candidates.append(str(frame_int).zfill(width))
    return list(dict.fromkeys(candidates))  # preserve order + dedupe


def resolve_file_name(video_name: str, frame_key: str, frame_info: Dict[str, Any], images_root: Path) -> str:
    # Common hints seen in ROAD-style utilities.
    for key in ("file_name", "img_path", "rgb_path", "image_path", "rgb_image"):
        value = frame_info.get(key)
        if isinstance(value, str) and value.strip():
            value = value.strip()
            p = Path(value)
            if p.is_absolute():
                try:
                    return str(p.relative_to(images_root))
                except Exception:
                    return p.name
            if "/" in value:
                return value
            return f"{video_name}/{value}"

    # Fallback patterns.
    frame_stems = candidate_frame_names(str(frame_key))
    extensions = (".jpg", ".jpeg", ".png")

    if images_root and images_root.exists():
        for stem in frame_stems:
            for ext in extensions:
                rel = Path(video_name) / f"{stem}{ext}"
                if (images_root / rel).is_file():
                    return str(rel)

    return f"{video_name}/{frame_stems[0]}.jpg"


def resolve_image_size(
    frame_info: Dict[str, Any],
    image_path: Path,
) -> Tuple[Optional[int], Optional[int]]:
    width_keys = ("width", "img_width", "image_width", "w")
    height_keys = ("height", "img_height", "image_height", "h")

    width = None
    height = None

    for key in width_keys:
        w = to_int(frame_info.get(key))
        if w is not None and w > 0:
            width = w
            break
    for key in height_keys:
        h = to_int(frame_info.get(key))
        if h is not None and h > 0:
            height = h
            break

    if width is not None and height is not None:
        return width, height

    if image_path.is_file():
        with Image.open(image_path) as img:
            return int(img.width), int(img.height)

    return width, height


def looks_normalized_xyxy(box: List[float]) -> bool:
    if len(box) != 4:
        return False
    return min(box) >= -1e-6 and max(box) <= 1.5


def convert_box_xyxy(
    box: List[Number],
    width: int,
    height: int,
    box_mode: str,
) -> Optional[Tuple[float, float, float, float]]:
    if len(box) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in box]

    use_normalized = False
    if box_mode == "normalized":
        use_normalized = True
    elif box_mode == "pixel":
        use_normalized = False
    else:  # auto
        use_normalized = looks_normalized_xyxy([x1, y1, x2, y2])

    if use_normalized:
        x1 *= float(width)
        x2 *= float(width)
        y1 *= float(height)
        y2 *= float(height)

    # clip to image bounds
    x1 = max(0.0, min(x1, float(width)))
    x2 = max(0.0, min(x2, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    y2 = max(0.0, min(y2, float(height)))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def iter_frame_annos(frame_info: Dict[str, Any], video_info: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Return iterable of (anno_id, anno_dict) for a frame.
    Supports:
    - frame['annos'] as dict of anno_id -> anno_obj
    - frame['annos'] as list of anno_obj
    - frame['annos'] as list of anno_ids with video-level dict fallback
    """
    frame_annos = frame_info.get("annos", {})
    video_annos = video_info.get("annos", {})

    if isinstance(frame_annos, dict):
        for aid, aval in frame_annos.items():
            if isinstance(aval, dict) and "box" in aval:
                yield str(aid), aval
        return

    if isinstance(frame_annos, list):
        for idx, item in enumerate(frame_annos):
            if isinstance(item, dict):
                aid = item.get("id", f"idx_{idx}")
                yield str(aid), item
            else:
                item_id = str(item)
                if isinstance(video_annos, dict):
                    aval = video_annos.get(item_id)
                    if isinstance(aval, dict):
                        yield item_id, aval
        return


def handle_error_or_warn(strict: bool, msg: str, counters: Dict[str, int], key: str) -> None:
    counters[key] = counters.get(key, 0) + 1
    if strict:
        raise ValueError(msg)


def main() -> None:
    args = parse_args()

    road_path = Path(args.road_json)
    out_path = Path(args.output_json)
    images_root = Path(args.images_root).expanduser().resolve() if args.images_root else Path("")

    keep_split_ids = parse_csv_str_set(args.include_split_ids)
    keep_class_ids = parse_csv_int_set(args.keep_class_ids) if args.keep_class_ids else set()
    keep_class_names = {x.strip().lower() for x in parse_csv_set(args.keep_class_names)}

    road = json.loads(road_path.read_text())
    if "db" not in road or not isinstance(road["db"], dict):
        raise ValueError("ROAD JSON must contain top-level dict key 'db'.")

    split_map = road.get("split_ids", {})
    if not isinstance(split_map, dict):
        split_map = {}

    label_map_key = choose_label_map_key(road, args.label_type, args.label_map_key)
    raw_label_map = road.get(label_map_key)
    label_map = normalize_label_map(raw_label_map)
    if not label_map:
        raise ValueError(f"Could not parse label map '{label_map_key}' from ROAD JSON.")

    label_ids_key = choose_label_ids_key(args.label_type)

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []

    image_uid_to_id: Dict[str, int] = {}
    categories_by_raw_id: Dict[int, Dict[str, Any]] = {}

    next_image_id = 1
    next_anno_id = 1
    counters: Dict[str, int] = {}

    # Build a deterministic category space up front so train/val/test JSONs share
    # the same semantic ID mapping even when some classes are absent in one split.
    selected_raw_ids: List[int] = sorted(int(k) for k in label_map.keys())
    if keep_class_ids:
        selected_raw_ids = [rid for rid in selected_raw_ids if rid in keep_class_ids]
    if keep_class_names:
        selected_raw_ids = [
            rid for rid in selected_raw_ids if str(label_map[rid]).lower() in keep_class_names
        ]
    if not selected_raw_ids:
        raise ValueError("No classes selected after applying keep_class_ids/keep_class_names filters.")

    for coco_id, rid in enumerate(selected_raw_ids, start=1):
        categories_by_raw_id[rid] = {
            "id": coco_id,
            "name": str(label_map[rid]),
            "supercategory": args.label_type,
            "road_raw_id": rid,
        }

    db = road["db"]
    for video_name, video_info in db.items():
        if not isinstance(video_info, dict):
            handle_error_or_warn(args.strict, f"Video payload for '{video_name}' is not a dict.", counters, "bad_video")
            continue

        # ROAD split can appear as video-level key ('split_ids') and/or in top-level map.
        raw_split = video_info.get("split_ids", split_map.get(video_name))
        split_values = [str(s).strip() for s in as_list(raw_split) if str(s).strip()]
        split_values_set = set(split_values)
        split_value = split_values[0] if split_values else None

        if keep_split_ids and not (split_values_set & keep_split_ids):
            counters["skipped_split_video"] = counters.get("skipped_split_video", 0) + 1
            continue

        frames = video_info.get("frames", {})
        if not isinstance(frames, dict):
            handle_error_or_warn(
                args.strict,
                f"frames for '{video_name}' is not a dict.",
                counters,
                "bad_frames",
            )
            continue

        for frame_key, frame_info in frames.items():
            if not isinstance(frame_info, dict):
                handle_error_or_warn(
                    args.strict,
                    f"frame '{video_name}/{frame_key}' payload is not a dict.",
                    counters,
                    "bad_frame",
                )
                continue

            file_name = resolve_file_name(video_name, str(frame_key), frame_info, images_root)
            image_abs = images_root / file_name if images_root else Path(file_name)
            width, height = resolve_image_size(frame_info, image_abs)

            if width is None or height is None:
                handle_error_or_warn(
                    args.strict,
                    f"Could not resolve image size for frame '{video_name}/{frame_key}' ({file_name}).",
                    counters,
                    "missing_size",
                )
                continue

            if args.require_images and (not image_abs.is_file()):
                counters["missing_image"] = counters.get("missing_image", 0) + 1
                continue

            image_uid = f"{video_name}::{frame_key}"
            image_id = image_uid_to_id.get(image_uid)
            if image_id is None:
                image_id = next_image_id
                next_image_id += 1
                image_uid_to_id[image_uid] = image_id
                images.append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "width": int(width),
                        "height": int(height),
                        "road_meta": {
                            "video_name": str(video_name),
                            "frame_key": str(frame_key),
                            "split_id": split_value,
                            "split_ids": split_values,
                        },
                    }
                )

            frame_ann_count = 0
            for anno_key, anno in iter_frame_annos(frame_info, video_info):
                frame_ann_count += 1
                box = anno.get("box")
                if not isinstance(box, list):
                    handle_error_or_warn(
                        args.strict,
                        f"Missing/invalid box for annotation '{video_name}/{frame_key}/{anno_key}'.",
                        counters,
                        "bad_box",
                    )
                    continue

                xyxy = convert_box_xyxy(box, width=width, height=height, box_mode=args.box_mode)
                if xyxy is None:
                    counters["skipped_invalid_box"] = counters.get("skipped_invalid_box", 0) + 1
                    continue
                x1, y1, x2, y2 = xyxy

                raw_label_ids = as_list(anno.get(label_ids_key))
                if not raw_label_ids:
                    counters["skipped_no_label"] = counters.get("skipped_no_label", 0) + 1
                    continue

                if args.multi_label_policy == "first":
                    raw_label_ids = raw_label_ids[:1]

                for raw_lid in raw_label_ids:
                    lid = to_int(raw_lid)
                    if lid is None:
                        handle_error_or_warn(
                            args.strict,
                            f"Non-integer label id '{raw_lid}' at '{video_name}/{frame_key}/{anno_key}'.",
                            counters,
                            "bad_label_id",
                        )
                        continue

                    label_name = label_map.get(lid)
                    if label_name is None:
                        handle_error_or_warn(
                            args.strict,
                            f"Unknown label id {lid} in '{video_name}/{frame_key}/{anno_key}'.",
                            counters,
                            "unknown_label",
                        )
                        continue

                    if lid not in categories_by_raw_id:
                        counters["skipped_filtered_class"] = counters.get("skipped_filtered_class", 0) + 1
                        continue

                    cat_id = categories_by_raw_id[lid]["id"]
                    w = x2 - x1
                    h = y2 - y1

                    annotations.append(
                        {
                            "id": next_anno_id,
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": [x1, y1, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                            "road_meta": {
                                "video_name": str(video_name),
                                "frame_key": str(frame_key),
                                "anno_key": str(anno_key),
                                "split_id": split_value,
                                "split_ids": split_values,
                                "label_type": args.label_type,
                                "raw_label_id": lid,
                                "raw_label_name": label_name,
                                "agent_ids": as_list(anno.get("agent_ids")),
                                "action_ids": as_list(anno.get("action_ids")),
                                "loc_ids": as_list(anno.get("loc_ids")),
                                "tube_uid": anno.get("tube_uid"),
                                "source_box_xyxy": box,
                            },
                        }
                    )
                    next_anno_id += 1

            if frame_ann_count == 0:
                counters["frames_without_annos"] = counters.get("frames_without_annos", 0) + 1

    categories = sorted(categories_by_raw_id.values(), key=lambda c: c["id"])

    coco = {
        "info": {
            "description": "ROAD converted to COCO detection format",
            "version": "1.0",
            "source": str(road_path),
            "label_type": args.label_type,
            "label_map_key": label_map_key,
            "box_mode": args.box_mode,
            "multi_label_policy": args.multi_label_policy,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2))

    print(f"Wrote COCO JSON: {out_path}")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")
    if keep_split_ids:
        print(f"Included split IDs: {sorted(keep_split_ids)}")
    if keep_class_ids:
        print(f"Class ID filter: {sorted(keep_class_ids)}")
    if keep_class_names:
        print(f"Class name filter: {sorted(keep_class_names)}")

    if counters:
        print("Skip/issue counters:")
        for key in sorted(counters.keys()):
            print(f"  - {key}: {counters[key]}")


if __name__ == "__main__":
    main()

import ast
from pathlib import Path
from typing import List, Dict

import torch
from torch import Tensor


COCO_CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motorcycle"},
    {"id": 5, "name": "airplane"},
    {"id": 6, "name": "bus"},
    {"id": 7, "name": "train"},
    {"id": 8, "name": "truck"},
    {"id": 9, "name": "boat"},
    {"id": 10, "name": "traffic light"},
    {"id": 11, "name": "fire hydrant"},
    {"id": 13, "name": "stop sign"},
    {"id": 14, "name": "parking meter"},
    {"id": 15, "name": "bench"},
    {"id": 16, "name": "bird"},
    {"id": 17, "name": "cat"},
    {"id": 18, "name": "dog"},
    {"id": 19, "name": "horse"},
    {"id": 20, "name": "sheep"},
    {"id": 21, "name": "cow"},
    {"id": 22, "name": "elephant"},
    {"id": 23, "name": "bear"},
    {"id": 24, "name": "zebra"},
    {"id": 25, "name": "giraffe"},
    {"id": 27, "name": "backpack"},
    {"id": 28, "name": "umbrella"},
    {"id": 31, "name": "handbag"},
    {"id": 32, "name": "tie"},
    {"id": 33, "name": "suitcase"},
    {"id": 34, "name": "frisbee"},
    {"id": 35, "name": "skis"},
    {"id": 36, "name": "snowboard"},
    {"id": 37, "name": "sports ball"},
    {"id": 38, "name": "kite"},
    {"id": 39, "name": "baseball bat"},
    {"id": 40, "name": "baseball glove"},
    {"id": 41, "name": "skateboard"},
    {"id": 42, "name": "surfboard"},
    {"id": 43, "name": "tennis racket"},
    {"id": 44, "name": "bottle"},
    {"id": 46, "name": "wine glass"},
    {"id": 47, "name": "cup"},
    {"id": 48, "name": "fork"},
    {"id": 49, "name": "knife"},
    {"id": 50, "name": "spoon"},
    {"id": 51, "name": "bowl"},
    {"id": 52, "name": "banana"},
    {"id": 53, "name": "apple"},
    {"id": 54, "name": "sandwich"},
    {"id": 55, "name": "orange"},
    {"id": 56, "name": "broccoli"},
    {"id": 57, "name": "carrot"},
    {"id": 58, "name": "hot dog"},
    {"id": 59, "name": "pizza"},
    {"id": 60, "name": "donut"},
    {"id": 61, "name": "cake"},
    {"id": 62, "name": "chair"},
    {"id": 63, "name": "couch"},
    {"id": 64, "name": "potted plant"},
    {"id": 65, "name": "bed"},
    {"id": 67, "name": "dining table"},
    {"id": 70, "name": "toilet"},
    {"id": 72, "name": "tv"},
    {"id": 73, "name": "laptop"},
    {"id": 74, "name": "mouse"},
    {"id": 75, "name": "remote"},
    {"id": 76, "name": "keyboard"},
    {"id": 77, "name": "cell phone"},
    {"id": 78, "name": "microwave"},
    {"id": 79, "name": "oven"},
    {"id": 80, "name": "toaster"},
    {"id": 81, "name": "sink"},
    {"id": 82, "name": "refrigerator"},
    {"id": 84, "name": "book"},
    {"id": 85, "name": "clock"},
    {"id": 86, "name": "vase"},
    {"id": 87, "name": "scissors"},
    {"id": 88, "name": "teddy bear"},
    {"id": 89, "name": "hair drier"},
    {"id": 90, "name": "toothbrush"},
]


def resolve_random_set_path(path_str: str) -> Path:
    if not path_str:
        raise ValueError("random_set_path is empty; provide a valid class-cluster file path.")

    path = Path(path_str)
    if path.is_file():
        return path

    cwd_path = Path.cwd() / path_str
    if cwd_path.is_file():
        return cwd_path

    local_path = Path(__file__).resolve().parent / path_str
    if local_path.is_file():
        return local_path

    raise FileNotFoundError(
        f"Could not find random-set classes file: {path_str}. "
        f"Tried absolute path, cwd, and package-local directory."
    )


def load_random_set_classes(path_str: str) -> List[set]:
    path = resolve_random_set_path(path_str)
    content = path.read_text()
    data = ast.literal_eval(content)
    return [set(item) for item in data]


def build_random_set_matrices(
    new_classes: List[set],
    base_class_names: List[str],
) -> Dict[str, Tensor]:
    name_to_base_idx = {name: i for i, name in enumerate(base_class_names)}

    set_indices = []
    for class_set in new_classes:
        indices = []
        for name in class_set:
            if name not in name_to_base_idx:
                raise ValueError(f"Unknown class name in random-set file: {name}")
            indices.append(name_to_base_idx[name])
        set_indices.append(set(indices))

    num_sets = len(set_indices)
    num_base = len(base_class_names)

    membership = torch.zeros((num_base, num_sets), dtype=torch.float32)
    for set_idx, indices in enumerate(set_indices):
        for base_idx in indices:
            membership[base_idx, set_idx] = 1.0

    mass_coeff = torch.zeros((num_sets, num_sets), dtype=torch.float32)
    for a_idx, a_set in enumerate(set_indices):
        for b_idx, b_set in enumerate(set_indices):
            if b_set.issubset(a_set):
                mass_coeff[b_idx, a_idx] = float(((-1) ** (len(a_set) - len(b_set))))

    pignistic = torch.zeros((num_sets + 1, num_base), dtype=torch.float32)
    for set_idx, indices in enumerate(set_indices):
        if not indices:
            continue
        weight = 1.0 / float(len(indices))
        for base_idx in indices:
            pignistic[set_idx, base_idx] = weight
    pignistic[-1, :] = 1.0 / float(num_base)

    return {
        "membership": membership,
        "mass_coeff": mass_coeff,
        "pignistic": pignistic,
    }


def belief_to_mass(belief: Tensor, mass_coeff: Tensor, clamp_negative: bool = False) -> Tensor:
    mass = belief @ mass_coeff.t()
    if clamp_negative:
        mass = torch.clamp(mass, min=0.0)
    return mass


def final_betp(mass: Tensor, pignistic: Tensor) -> Tensor:
    remaining = torch.clamp(1.0 - mass.sum(dim=-1), min=0.0)
    mass_full = torch.cat([mass, remaining.unsqueeze(-1)], dim=-1)
    denom = torch.clamp(mass_full.sum(dim=-1, keepdim=True), min=1e-6)
    mass_full = mass_full / denom
    return mass_full @ pignistic

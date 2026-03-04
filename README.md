# Epistemic Object Detector

PyTorch RetinaNet codebase extended with Dirichlet-based epistemic uncertainty modeling and random-set classification support.

## What this clean repo includes
- Core RetinaNet package in `retinanet/`
- Training/evaluation entrypoints:
  - `train.py`
  - `coco_validation.py`
  - `csv_validation.py`
  - `epistemic_uncertainty_eval.py`
  - `alpha_simplex_eval.py`
  - `visualize.py`, `visualize_single_image.py`
- Data conversion and reporting helpers:
  - `road_to_coco.py`
  - `make_uncertainty_tables.py`
- Example random-set clustering files:
  - `road_agent_clustering_66.txt`
  - `road_agent_clustering_pruned_31.txt`
  - `road_agent_clustering_pruned_9cls.txt`

The repository intentionally excludes generated experiment outputs, caches, and checkpoints.

## Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Expected COCO layout
Point `--coco_path` to a directory with this structure:

```text
<COCO_ROOT>/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

## Training examples
Baseline:
```bash
python train.py \
  --dataset coco \
  --coco_path /path/to/coco \
  --model_variant baseline \
  --depth 50
```

Dirichlet standard classifier:
```bash
python train.py \
  --dataset coco \
  --coco_path /path/to/coco \
  --model_variant dirichlet_std \
  --depth 50
```

Dirichlet random-set classifier (requires cluster file):
```bash
python train.py \
  --dataset coco \
  --coco_path /path/to/coco \
  --model_variant dirichlet_randomset \
  --random_set_path /path/to/random_set_clusters.txt \
  --depth 50
```

## Evaluation examples
COCO mAP:
```bash
python coco_validation.py \
  --coco_path /path/to/coco \
  --model_path /path/to/checkpoint.pt \
  --model_variant baseline
```

Epistemic uncertainty report:
```bash
python epistemic_uncertainty_eval.py \
  --coco_path /path/to/coco \
  --model_path /path/to/checkpoint.pt \
  --model_variant dirichlet_std \
  --output_dir uncertainty_eval_output
```

Alpha simplex visualization:
```bash
python alpha_simplex_eval.py \
  --coco_path /path/to/coco \
  --model_path /path/to/checkpoint.pt \
  --model_variant dirichlet_std \
  --output_dir alpha_simplex_eval_output
```

## Notes
- `--random_set_path` is required when `--model_variant dirichlet_randomset` is used.
- This repository does not include datasets or pretrained checkpoints.
- Generated artifacts are ignored by `.gitignore` for cleaner commits.

## License
See `LICENSE`.

# SAR-based Marine Oil Spill Detection (Hybrid DeepLabV3+ + SegNet)

This repository contains code for SAR oil spill segmentation (SegNet, DeepLabV3+, and a hybrid fusion model) plus Grad-CAM visualizations.

## What was removed (GitHub-friendly)

To keep the repo small for GitHub, these are **not included**:
- Datasets (`dataset/`)
- Trained weights / checkpoints (`*.pth`)
- Generated results (`outputs/`, `results/`, `results_*`, `results_gradcam/`, `gradcam_*`)

They are already in `.gitignore`, and will be created locally when you run training/inference/visualization scripts.

## Included Grad-CAM outputs (3 images)

Sample Grad-CAM comparison grids:
- `assets/gradcam_examples/01_10001_all_models_gradcam_grid.png`
- `assets/gradcam_examples/06_10006_all_models_gradcam_grid.png`
- `assets/gradcam_examples/09_10009_all_models_gradcam_grid.png`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Expected dataset path

Most scripts default to:
- `dataset/SOS_dataset/train`
- `dataset/SOS_dataset/test`

## Train

```bash
python3 train_segnet.py
python3 train_deeplab.py
python3 train_hybrid.py
```

Outputs go to `outputs/<model_type>/checkpoints/` (ignored by git).

## Inference (predictions + visualizations)

```bash
python3 infer.py --checkpoint outputs/hybrid/checkpoints/best_model.pth --model_type hybrid
```

## Grad-CAM grids (all models)

After you have the three model checkpoints saved under `outputs/`, run:

```bash
python3 generate_all_models_gradcam_grid.py --limit 3
```
# oilspill_gradcam

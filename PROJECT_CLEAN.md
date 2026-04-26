# Project Cleanup Notes (GitHub Upload)

This folder was cleaned to make it small and safe to push to GitHub.

## Removed (not uploaded)

- Datasets: `dataset/`
- Virtual environments: `venv/`, etc.
- Model weights / checkpoints: `*.pth`
- Generated outputs/results: `outputs/`, `output/`, `results/`, `results_*`, `results_gradcam/`, `gradcam_*`
- Cache/system files: `__pycache__/`, `.DS_Store`, `__MACOSX/`

These items are now ignored via `.gitignore`.

## Kept

- Core training + inference code (`train_*.py`, `infer.py`, `models/`, `utils.py`, `data_loader.py`, etc.)
- Documentation (`START_HERE.txt`, `EVALUATION_GUIDE.txt`, `PROJECT_REPORT.txt`, etc.)
- Grad-CAM visual examples (only 3 images): `assets/gradcam_examples/`

## How to regenerate results locally

1. Put the dataset back under `dataset/SOS_dataset/` (train/test).
2. Train models (creates `outputs/...` locally):
   - `python3 train_segnet.py`
   - `python3 train_deeplab.py`
   - `python3 train_hybrid.py`
3. Generate Grad-CAM grids:
   - `python3 generate_all_models_gradcam_grid.py --limit 3`

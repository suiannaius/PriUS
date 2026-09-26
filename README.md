# PriUS

Principle-guided interpretable uncertainty for medical image segmentation, with training, inference, and UCC/UR evaluation.

- **3D:** WHS cardiac CT (repository root).
- **2D:** ACDC cardiac MRI and ISIC 2018 ([`2D/`](2D/)).

## WHS (3D)

Run from the repository root. Prepare Center A CT data under `train/image`, `train/label`, `test/image`, and `test/label`, preserving the original filenames.

```bash
pip install -r requirements.txt
export WHS_DATA_ROOT=/path/to/WHS_DATA_ROOT
python get_data_stats.py
python count_class_ratio.py
```

Train the initialization model, then PriUS:

```bash
python run_training.py --task_id 1001 --date 2026-09-27 \
  --method PureEvidential --batch_size 4 --num_epochs 100 \
  --use_noise_aug --use_prior --skip_test

python run_training.py --task_id 1018 --date 2026-09-27 \
  --method Ours --batch_size 12 --num_epochs 100 \
  --threshold_sigma 2 --threshold_d 2 --threshold_far 2 \
  --sigma_blur 0.5 --beta 1.0 --gamma 0.1 \
  --coef_sigma 0.1 --coef_far 100.0 --coef_d 100.0 \
  --use_prior --use_pretrain \
  --pretrain_path saved_models/Task_1001_2026-09-27.pth --skip_test
```

Evaluate:

```bash
python run_inference.py --project Task_1018_2026-09-27
```

Checkpoints are saved in `saved_models/`; evaluation results are saved in `results/inference/WHS/`.

## ACDC and ISIC (2D)

```bash
cd 2D
pip install -r requirements.txt
```

Use `configs/acdc.json` for ACDC and `configs/isic.json` for ISIC. Training and evaluation commands are provided in the [2D README](2D/README.md).

## Third-party code

Surface-distance functions by Oskar Maier are licensed under GPL-3.0-or-later; see [LICENSE.binary](LICENSE.binary).

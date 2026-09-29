# PriUS 2D

Run these commands from `2D/`.

## Installation

```bash
pip install -r requirements.txt
```

## ACDC

The data directory contains `patient001` through `patient100`, with each annotated image named `patientXXX_frameYY.nii.gz` and its mask named `patientXXX_frameYY_gt.nii.gz`.

Train all five folds:

```bash
python run_training.py --config configs/acdc.json \
  --data-dir /path/to/ACDC/training --output-dir outputs/acdc
```

Use `--fold 0` to train a single fold. Checkpoints are written to `saved_models/fold_0.pth` through `saved_models/fold_4.pth` under the output directory.

Evaluate each held-out fold:

```bash
for fold in 0 1 2 3 4; do
  python run_inference.py --config configs/acdc.json \
    --data-dir /path/to/ACDC/training --output-dir outputs/acdc_eval \
    --checkpoint "outputs/acdc/saved_models/fold_${fold}.pth" \
    --fold "$fold" --uncertainty-metrics
done
```

Results are saved to `results/fold_N.json`. Each file contains sample, frame and patient records, metric-specific valid counts, and the fold summary. For an overall ACDC result, average the valid patient records across the five folds with equal patient weights.

## ISIC 2018

Supply separate image and mask directories for the desired training or evaluation split. Images are `.jpg` files; masks are `.png` files with matching identifiers and an optional `_segmentation` suffix.

```bash
python run_training.py --config configs/isic.json \
  --image-dir /path/to/train/images --mask-dir /path/to/train/masks \
  --output-dir outputs/isic

python run_inference.py --config configs/isic.json \
  --image-dir /path/to/test/images --mask-dir /path/to/test/masks \
  --checkpoint outputs/isic/saved_models/fold_0.pth \
  --output-dir outputs/isic_eval --uncertainty-metrics
```

## Evaluation

`--uncertainty-metrics` sets the ACDC evaluation batch size to 1. ACDC uncertainty forwards use train mode without parameter updates. BatchNorm buffers are restored to checkpoint values at each new patient. ISIC retains eval-mode uncertainty evaluation and its configured batch size. Segmentation is evaluated separately in eval mode with checkpoint BatchNorm buffers. All samples are retained. Training batch sizes in the configuration files are unchanged.

UCC uses average-rank Spearman correlation. Constant inputs or fewer than two observations produce an undefined correlation, written as JSON `null`. Means use valid observations with equal weights and report coverage. ACDC uses slice-to-frame-to-patient aggregation; ISIC uses image means. Segmentation HD95 retains its existing units: millimetres for ACDC and pixels at 256×256 for ISIC.

For URg and URd, a pair is accepted when the attribute and uncertainty differences have opposite signs, uncertainty is exactly equal, or the attribute is exactly equal and the absolute uncertainty difference is strictly below `1e-4`. URσ uses the corresponding increasing direction; with two increasing noise levels it accepts `u_high >= u_low`. The tolerance applies to the equal-attribute branch, not to Spearman ranks or every small uncertainty decrease.

σ uses corresponding pixels across two noise levels and averages pixel correlations over valid pixels. g retains gradient sampling at interval 0.04 and its existing region threshold. Geometry scores average clean and two noisy states over valid scores. Region thresholds continue to come from the configuration.

URd visits all unordered pairs within each represented class, including cross-block pairs. The existing additional 2D filter `d[j] - d[i] > 3` for original pixel indices `i < j` is retained. Only pairs passing this filter contribute to URd; UCCd uses the selected class pixels. Thus this retained filter is directional, depends on pixel ordering, and differs from an unfiltered all-pair URd evaluation. The WHS implementation has no such additional filter.

`--save-predictions` writes eval-mode segmentation predictions and their eval-mode uncertainty maps to `predictions/fold_N/`. The train-mode uncertainty scores are recorded in the result JSON.

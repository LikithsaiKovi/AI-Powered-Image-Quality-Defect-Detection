# Evaluation protocol

The model is a multi-output Random Forest: it predicts independent binary probabilities for `blur`, `underexposure`, `overexposure`, `noise`, `defect`, and `severe_degradation`. Inputs are eleven interpretable image features extracted with OpenCV.

## Data generation

Start with clean source images in `data/clean/`. The training script creates controlled transformations: Gaussian blur, reduced/increased exposure, additive Gaussian noise, localized lines/blocks for visual defects, and severe resize/JPEG degradation. It also makes multi-issue examples. Every generated variant inherits labels from the transformation that created it.

## Leakage prevention

The split is grouped by the original clean source image using `GroupShuffleSplit`. A source image and any of its transformed variants are therefore never present in both train and validation data.

## Run the evaluation

```powershell
cd backend
python -m app.ml.train --clean-dir ../data/clean --output models/quality_model.joblib --report-dir ../docs/results
```

The training command prints validation F1 for each label and writes `metrics.json` and `per_class_metrics.csv` to the report directory. Record this output in the final submission after training with real, unseen source images. Do not report the Docker demo bootstrap model as an evaluation result.

## Limitations and expected failure cases

- A deliberately dark photograph may resemble underexposure even when that aesthetic is intentional.
- Fine texture can be confused with sensor noise; highly textured images are a useful negative test.
- Synthetic defect shapes only approximate real manufacturing or camera defects. Add representative real defects if available.
- An unreadable file is rejected with HTTP 422 rather than passed to the ML model; this is the correct behaviour for corrupted encodings.

# AI-Powered Image Quality & Defect Detection

A deployable full-stack image-quality assessment application for the internship technical assessment. It detects blur, underexposure, overexposure, noise, and potential visual defects, returns a structured assessment, persists results, and displays analysis history.

## Why this approach

The solution is a **hybrid computer-vision and machine-learning system**. OpenCV derives interpretable visual features (sharpness, exposure clipping, contrast, noise residual, entropy, saturation, edge density, blockiness, and colourfulness). A multi-output Random Forest learns issue probabilities from those features. This is an AI-based decision component, rather than a fixed threshold-only rules engine, while remaining fast and explainable.

The overall score starts at 100 and deducts severity-weighted learned issue probabilities; visual defects and severe degradation carry the largest penalty. Labels are `ACCEPTABLE` (80–100), `DEGRADED` (50–79), and `POTENTIALLY_DEFECTIVE` (0–49). The API includes both the probabilities and source statistics so the decision can be reviewed.

## Technology

- Backend: Python 3.11, FastAPI, OpenCV, scikit-learn, SQLAlchemy
- Persistence: SQLite
- Frontend: React, TypeScript, Vite
- Deployment: Docker and Docker Compose
- Tests: pytest

No external AI services, image-analysis APIs, or API keys are used.

## Run with Docker (recommended)

```powershell
docker compose up --build
```

Open `http://localhost:8080`. The API is available at `http://localhost:8000`; its interactive OpenAPI documentation is at `http://localhost:8000/docs`.

On first startup, the container creates a small procedural demo training set and trains a model automatically so the application is immediately runnable. For the assessment, train with real clean source images before presenting final evaluation results.

## Train with your own clean images

Put at least 30 copyright-permitted clean JPEG/PNG/WEBP/BMP images under `data/clean/`, then run:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.ml.train --clean-dir ../data/clean --output models/quality_model.joblib --report-dir ../docs/results
uvicorn app.main:app --reload
```

The training script makes deterministic controlled degradations (including severe JPEG/pixelation degradation), uses a grouped train/validation split, and writes per-class precision, recall, F1, and confusion-matrix counts. See [the evaluation protocol](docs/evaluation.md) for limitations.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service and model health |
| `POST` | `/api/analyses` | Upload and analyse an image |
| `GET` | `/api/analyses` | List latest 100 saved analyses |
| `GET` | `/api/analyses/{id}` | Retrieve one analysis |

Example upload:

```powershell
curl.exe -X POST http://localhost:8000/api/analyses -F "file=@sample.jpg;type=image/jpeg"
```

Example response:

```json
{
  "id": 1,
  "filename": "sample.jpg",
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "issues": [{"type": "noise", "severity": "low", "confidence": 0.41}],
  "statistics": {"laplacian_variance": 223.8, "brightness_mean": 136.2},
  "model_version": "rf-synthetic-v1"
}
```

## Data, modelling, and evaluation

The controlled transformations are in `backend/app/ml/degradations.py`; feature extraction is in `backend/app/ml/features.py`; model training is in `backend/app/ml/train.py`. Use unseen source images for validation, retain the generated results, and create representative samples with `python -m app.ml.generate_samples --clean-dir ../data/clean --output-dir ../data/samples`.

See [architecture](docs/architecture.md) and [evaluation](docs/evaluation.md) for the technical explanation, data-split methodology, limitations, and failure cases.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/quality.db` | Database connection URL |
| `MODEL_PATH` | `./models/quality_model.joblib` | Saved model location |
| `UPLOAD_DIR` | `./uploads` | Uploaded image storage |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum upload size |
| `CORS_ORIGINS` | localhost ports | Permitted frontend origins |

## Verification checklist

```powershell
cd backend
pytest
docker compose up --build
curl.exe http://localhost:8000/health
```

Before submission, add real clean source images, retrain the model, save the reported validation results, test invalid image handling, test the UI history screen, and run the complete Docker deployment from a fresh checkout.

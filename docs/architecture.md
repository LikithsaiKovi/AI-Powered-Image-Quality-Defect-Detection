# Architecture

```text
React UI → FastAPI REST API → OpenCV feature extraction → Multi-output Random Forest
                 ↓                         ↓
              SQLite history          JSON assessment
```

The browser uploads a single image with multipart form data. FastAPI validates size and MIME type, decodes it with OpenCV, obtains model probabilities, calculates a weighted 0–100 score, persists the structured result in SQLite, and returns it to the UI. The frontend presents the upload preview, score, issues, per-feature statistics, and saved history.

The deployment is intentionally local-first: Docker Compose runs frontend and backend with named volumes for the database, model, and uploads. No API keys or external AI services are used.

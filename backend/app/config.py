from pathlib import Path
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/quality.db")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "./models/quality_model.joblib"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
CORS_ORIGINS = [item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",")]
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

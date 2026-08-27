import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .config import CORS_ORIGINS, MAX_UPLOAD_BYTES, UPLOAD_DIR
from .database import Base, engine, get_db
from .models import Analysis
from .schemas import AnalysisResult
from .services import QualityAnalyzer

analyzer: QualityAnalyzer | None = None

@asynccontextmanager
async def lifespan(_: FastAPI):
    global analyzer
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True); Base.metadata.create_all(bind=engine)
    analyzer = QualityAnalyzer()
    yield

app = FastAPI(title="Image Quality Assessment API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health(): return {"status": "ok", "model_loaded": analyzer is not None}

@app.post("/api/analyses", response_model=AnalysisResult, status_code=201)
async def create_analysis(file: UploadFile = File(...), db: Session = Depends(get_db)):
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed: raise HTTPException(415, "Only JPEG, PNG, WEBP, and BMP images are accepted.")
    content = await file.read()
    if not content: raise HTTPException(400, "The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES: raise HTTPException(413, f"Image exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.")
    try: result = analyzer.analyse(content) if analyzer else (_ for _ in ()).throw(RuntimeError("Model unavailable"))
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    suffix = Path(file.filename or "image.jpg").suffix.lower() or ".img"; stored = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"; stored.write_bytes(content)
    row = Analysis(filename=file.filename or "unnamed-image", media_type=file.content_type, image_path=str(stored), result_json=json.dumps(result), **{key: result[key] for key in ("quality_score", "quality_label")})
    db.add(row); db.commit(); db.refresh(row)
    return {"id": row.id, "filename": row.filename, "created_at": row.created_at, **result}

@app.get("/api/analyses", response_model=list[AnalysisResult])
def list_analyses(db: Session = Depends(get_db)):
    rows = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(100).all()
    return [{"id": row.id, "filename": row.filename, "created_at": row.created_at, **json.loads(row.result_json)} for row in rows]

@app.get("/api/analyses/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    row = db.get(Analysis, analysis_id)
    if not row: raise HTTPException(404, "Analysis not found.")
    return {"id": row.id, "filename": row.filename, "created_at": row.created_at, **json.loads(row.result_json)}

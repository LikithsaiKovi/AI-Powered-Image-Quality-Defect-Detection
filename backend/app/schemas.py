from datetime import datetime
from pydantic import BaseModel, Field

class Issue(BaseModel):
    type: str
    severity: str
    confidence: float = Field(ge=0, le=1)

class AnalysisResult(BaseModel):
    id: int | None = None
    filename: str
    quality_score: int = Field(ge=0, le=100)
    quality_label: str
    issues: list[Issue]
    statistics: dict[str, float]
    explanation: str
    model_version: str
    created_at: datetime | None = None

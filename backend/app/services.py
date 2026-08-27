from pathlib import Path
import joblib
import cv2
import numpy as np
from .config import MODEL_PATH
from .ml.features import extract_features, vectorize

class QualityAnalyzer:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise RuntimeError(f"Model not found at {model_path}. Run python -m app.ml.train first.")
        self.artifact = joblib.load(model_path)
        self.model = self.artifact["model"]
        self.issues = self.artifact["issues"]

    def analyse(self, raw_bytes: bytes) -> dict:
        decoded = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("The file is not a readable image or is severely corrupted.")
        features = extract_features(decoded)
        vector = np.asarray([vectorize(features)])
        probabilities = self.model.predict_proba(vector)
        issue_probs = {issue: float(probability[0][list(self.model.classes_[idx]).index(1)]) if 1 in self.model.classes_[idx] else 0.0 for idx, (issue, probability) in enumerate(zip(self.issues, probabilities))}
        weights = {"blur": 24, "underexposure": 19, "overexposure": 19, "noise": 18, "defect": 25}
        score = int(round(np.clip(100 - sum(weights[name] * value for name, value in issue_probs.items()), 0, 100)))
        label = "ACCEPTABLE" if score >= 80 else "DEGRADED" if score >= 50 else "POTENTIALLY_DEFECTIVE"
        detected = []
        for issue, confidence in issue_probs.items():
            if confidence >= .35:
                severity = "high" if confidence >= .75 else "medium" if confidence >= .55 else "low"
                detected.append({"type": issue, "severity": severity, "confidence": round(confidence, 3)})
        detected.sort(key=lambda item: item["confidence"], reverse=True)
        top = detected[0] if detected else None
        explanation = (f"Quality score combines learned probabilities from {self.artifact['version']}. " +
                       (f"The strongest signal is {top['type']} ({top['confidence']:.0%}); feature values are included for review." if top else "No issue probability exceeded the reporting threshold."))
        return {"quality_score": score, "quality_label": label, "issues": detected, "statistics": {key: round(value, 3) for key, value in features.items()}, "explanation": explanation, "model_version": self.artifact["version"]}

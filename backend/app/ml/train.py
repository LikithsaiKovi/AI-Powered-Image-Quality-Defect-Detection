"""Train the multi-label Random Forest from clean images and controlled degradations."""
import argparse
import csv
import json
from pathlib import Path
import joblib
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from .degradations import ISSUES, apply_degradation
from .features import FEATURE_NAMES, extract_features, vectorize

def procedural_clean_images(count: int = 80) -> list[np.ndarray]:
    """Fallback set for a runnable demo; replace with real clean images for final evaluation."""
    rng = np.random.default_rng(42); images = []
    for _ in range(count):
        h, w = 256, 256
        yy, xx = np.mgrid[0:h, 0:w]
        base = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            base[:, :, c] = np.clip(rng.integers(20, 120) + xx * rng.uniform(-.2, .6) + yy * rng.uniform(-.2, .6), 0, 255)
        for _ in range(rng.integers(5, 16)):
            center = tuple(int(v) for v in rng.integers(0, 256, size=2)); radius = int(rng.integers(8, 70))
            cv2.circle(base, center, radius, tuple(int(v) for v in rng.integers(0, 256, size=3)), -1)
        images.append(base)
    return images

def load_clean_images(directory: Path) -> list[np.ndarray]:
    items = []
    for path in directory.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            image = cv2.imread(str(path));
            if image is not None: items.append(cv2.resize(image, (256, 256)))
    return items

def make_dataset(clean_images: list[np.ndarray], seed: int = 7):
    rng = np.random.default_rng(seed); X, y, groups = [], [], []
    for group, image in enumerate(clean_images):
        variants = [(image, np.zeros(len(ISSUES), dtype=int))]
        for issue_index, issue in enumerate(ISSUES):
            for _ in range(3):
                labels = np.zeros(len(ISSUES), dtype=int); labels[issue_index] = 1
                variants.append((apply_degradation(image, issue, rng), labels))
        # realistic multi-issue combinations
        for _ in range(3):
            selected = rng.choice(len(ISSUES), size=int(rng.integers(2, 4)), replace=False)
            altered, labels = image, np.zeros(len(ISSUES), dtype=int)
            for index in selected:
                altered = apply_degradation(altered, ISSUES[index], rng); labels[index] = 1
            variants.append((altered, labels))
        for altered, labels in variants:
            X.append(vectorize(extract_features(altered))); y.append(labels); groups.append(group)
    return np.asarray(X), np.asarray(y), np.asarray(groups)

def write_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray, report_dir: Path, source_count: int):
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, issue in enumerate(ISSUES):
        tn, fp, fn, tp = confusion_matrix(y_true[:, index], y_pred[:, index], labels=[0, 1]).ravel()
        rows.append({"issue": issue, "precision": round(float(precision_score(y_true[:, index], y_pred[:, index], zero_division=0)), 4), "recall": round(float(recall_score(y_true[:, index], y_pred[:, index], zero_division=0)), 4), "f1": round(float(f1_score(y_true[:, index], y_pred[:, index], zero_division=0)), 4), "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)})
    with (report_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    overall_true, overall_pred = y_true.any(axis=1).astype(int), y_pred.any(axis=1).astype(int)
    overall_cm = confusion_matrix(overall_true, overall_pred, labels=[0, 1]).tolist()
    summary = {"source_images": source_count, "split": "grouped 80/20 by clean source image", "classes": list(ISSUES), "per_class": rows, "overall_any_issue_confusion_matrix": {"labels": ["acceptable", "has_issue"], "matrix": overall_cm}}
    (report_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows

def train(clean_dir: Path | None, output: Path, bootstrap_if_missing: bool = False, report_dir: Path | None = None):
    clean_images = load_clean_images(clean_dir) if clean_dir and clean_dir.exists() else []
    if len(clean_images) < 10:
        if not bootstrap_if_missing:
            raise RuntimeError("Provide at least 10 clean images via --clean-dir; demo bootstrap is intentionally opt-in.")
        clean_images = procedural_clean_images()
    X, y, groups = make_dataset(clean_images)
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=11)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    model = RandomForestClassifier(n_estimators=250, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=11, class_weight="balanced")
    model.fit(X[train_idx], y[train_idx])
    predicted = model.predict(X[test_idx])
    report_rows = write_evaluation_report(y[test_idx], predicted, report_dir, len(clean_images)) if report_dir else []
    metrics = {row["issue"]: row["f1"] for row in report_rows} if report_rows else {issue: float(f1_score(y[test_idx, i], predicted[:, i], zero_division=0)) for i, issue in enumerate(ISSUES)}
    artifact = {"model": model, "feature_names": FEATURE_NAMES, "issues": ISSUES, "version": "rf-controlled-degradation-v2", "validation_f1": metrics, "training_sources": len(clean_images)}
    output.parent.mkdir(parents=True, exist_ok=True); joblib.dump(artifact, output)
    print({"saved": str(output), "validation_f1": metrics, "training_sources": len(clean_images), "rows": len(X), "report_dir": str(report_dir) if report_dir else None})
    return artifact

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=Path, default=Path("../data/clean"))
    parser.add_argument("--output", type=Path, default=Path("./models/quality_model.joblib"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--bootstrap-if-missing", action="store_true")
    args = parser.parse_args(); train(args.clean_dir, args.output, args.bootstrap_if_missing, args.report_dir)

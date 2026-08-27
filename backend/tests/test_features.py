from pathlib import Path
import cv2
import numpy as np
import pytest
from app.ml.features import FEATURE_NAMES, extract_features, vectorize
from app.ml.degradations import ISSUES, apply_degradation
from app.services import QualityAnalyzer

def test_feature_extractor_returns_finite_features():
    image = np.full((128, 128, 3), 127, dtype=np.uint8)
    features = extract_features(image)
    assert list(features) == FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values())

def test_vectorize_matches_feature_names():
    image = np.full((128, 128, 3), 127, dtype=np.uint8)
    features = extract_features(image)
    vec = vectorize(features)
    assert len(vec) == len(FEATURE_NAMES)
    assert all(np.isfinite(val) for val in vec)

def test_blur_lowers_sharpness_for_a_textured_image():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    blurred = apply_degradation(image, "blur", rng)
    assert extract_features(blurred)["laplacian_variance"] < extract_features(image)["laplacian_variance"]

def test_all_degradations_produce_valid_image():
    rng = np.random.default_rng(42)
    base = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
    for issue in ISSUES:
        degraded = apply_degradation(base, issue, rng)
        assert degraded is not None
        assert degraded.shape == base.shape
        assert degraded.dtype == np.uint8

def test_quality_analyzer_evaluates_sample_image():
    model_file = Path("models/quality_model.joblib")
    if not model_file.exists():
        pytest.skip("Model artifact not yet built")
    analyzer = QualityAnalyzer(model_file)
    clean_sample = Path("../data/samples/clean.jpg")
    if clean_sample.exists():
        res = analyzer.analyse(clean_sample.read_bytes())
        assert "quality_score" in res
        assert "quality_label" in res
        assert "issues" in res
        assert "statistics" in res
        assert 0 <= res["quality_score"] <= 100
        assert res["quality_label"] in ["ACCEPTABLE", "DEGRADED", "POTENTIALLY_DEFECTIVE"]

def test_quality_analyzer_rejects_corrupt_data():
    model_file = Path("models/quality_model.joblib")
    if not model_file.exists():
        pytest.skip("Model artifact not yet built")
    analyzer = QualityAnalyzer(model_file)
    with pytest.raises(ValueError, match="not a readable image or is severely corrupted"):
        analyzer.analyse(b"corrupted_non_image_bytes")


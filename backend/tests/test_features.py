import cv2
import numpy as np
from app.ml.features import FEATURE_NAMES, extract_features
from app.ml.degradations import apply_degradation

def test_feature_extractor_returns_finite_features():
    image = np.full((128, 128, 3), 127, dtype=np.uint8)
    features = extract_features(image)
    assert list(features) == FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values())

def test_blur_lowers_sharpness_for_a_textured_image():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    blurred = apply_degradation(image, "blur", rng)
    assert extract_features(blurred)["laplacian_variance"] < extract_features(image)["laplacian_variance"]

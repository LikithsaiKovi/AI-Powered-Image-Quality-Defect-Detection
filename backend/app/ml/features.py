"""Interpretable OpenCV features used by the learned multi-label model."""
import cv2
import numpy as np

FEATURE_NAMES = [
    "brightness_mean", "brightness_std", "dark_clip_ratio", "bright_clip_ratio",
    "laplacian_variance", "edge_density", "noise_mad", "entropy",
    "saturation_mean", "blockiness", "colorfulness",
]

def extract_features(image: np.ndarray) -> dict[str, float]:
    if image is None or image.size == 0:
        raise ValueError("Image could not be decoded")
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image[:, :, :3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    grayf = gray.astype(np.float32)
    lap = cv2.Laplacian(grayf, cv2.CV_32F)
    median = cv2.medianBlur(gray, 5).astype(np.float32)
    residual = np.abs(grayf - median)
    edges = cv2.Canny(gray, 80, 160)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probs = hist / max(hist.sum(), 1)
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    # JPEG-like blocking discontinuities measured across 8px boundaries.
    vertical = np.abs(np.diff(grayf, axis=1))
    horizontal = np.abs(np.diff(grayf, axis=0))
    boundary_v = vertical[:, 7::8].mean() if vertical.shape[1] >= 8 else 0.0
    boundary_h = horizontal[7::8, :].mean() if horizontal.shape[0] >= 8 else 0.0
    nonboundary_v = np.delete(vertical, np.arange(7, vertical.shape[1], 8), axis=1).mean() if vertical.shape[1] > 8 else 1.0
    nonboundary_h = np.delete(horizontal, np.arange(7, horizontal.shape[0], 8), axis=0).mean() if horizontal.shape[0] > 8 else 1.0
    rg = bgr[:, :, 2].astype(float) - bgr[:, :, 1].astype(float)
    yb = 0.5 * (bgr[:, :, 2].astype(float) + bgr[:, :, 1].astype(float)) - bgr[:, :, 0].astype(float)
    colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return {
        "brightness_mean": float(grayf.mean()), "brightness_std": float(grayf.std()),
        "dark_clip_ratio": float((gray < 15).mean()), "bright_clip_ratio": float((gray > 240).mean()),
        "laplacian_variance": float(lap.var()), "edge_density": float((edges > 0).mean()),
        "noise_mad": float(np.median(residual)), "entropy": float(entropy),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "blockiness": float((boundary_v + boundary_h) / max(nonboundary_v + nonboundary_h, 0.01)),
        "colorfulness": float(colorfulness),
    }

def vectorize(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]

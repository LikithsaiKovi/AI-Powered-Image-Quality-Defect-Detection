"""Controlled degradations make label generation reproducible without external AI services."""
import cv2
import numpy as np

ISSUES = ("blur", "underexposure", "overexposure", "noise", "defect")

def apply_degradation(image: np.ndarray, issue: str, rng: np.random.Generator) -> np.ndarray:
    output = image.copy()
    if issue == "blur":
        k = int(rng.choice([5, 7, 9, 11])); return cv2.GaussianBlur(output, (k, k), 0)
    if issue == "underexposure":
        return cv2.convertScaleAbs(output, alpha=float(rng.uniform(.3, .65)), beta=-int(rng.integers(5, 25)))
    if issue == "overexposure":
        return cv2.convertScaleAbs(output, alpha=float(rng.uniform(1.35, 2.0)), beta=int(rng.integers(20, 70)))
    if issue == "noise":
        noise = rng.normal(0, float(rng.uniform(15, 35)), output.shape).astype(np.int16)
        return np.clip(output.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if issue == "defect":
        h, w = output.shape[:2]
        for _ in range(int(rng.integers(1, 4))):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            size = max(3, int(min(h, w) * rng.uniform(.02, .12)))
            color = tuple(int(v) for v in rng.integers(0, 256, size=3))
            if rng.random() > .5:
                cv2.rectangle(output, (x, y), (min(w - 1, x + size), min(h - 1, y + size)), color, -1)
            else:
                cv2.line(output, (x, y), (min(w - 1, x + size * 3), min(h - 1, y + size)), color, max(1, size // 5))
        return output
    raise ValueError(f"Unknown issue: {issue}")

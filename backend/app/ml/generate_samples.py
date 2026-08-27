"""Create reproducible demo images from one clean source image."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from .degradations import ISSUES, apply_degradation

def generate(clean_dir: Path, output_dir: Path, seed: int = 20260828):
    candidates = sorted(path for path in clean_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    if not candidates:
        raise RuntimeError(f"No supported images found in {clean_dir}")
    source = cv2.imread(str(candidates[0]))
    if source is None:
        raise RuntimeError(f"Unable to read {candidates[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "clean.jpg"), source)
    rng = np.random.default_rng(seed)
    for issue in ISSUES:
        cv2.imwrite(str(output_dir / f"{issue}.jpg"), apply_degradation(source, issue, rng))
    print({"source": str(candidates[0]), "output_dir": str(output_dir), "samples": ["clean", *ISSUES]})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); generate(args.clean_dir, args.output_dir)

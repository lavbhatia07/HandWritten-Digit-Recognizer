"""
Train a lightweight deployment-safe digit recognizer for digits 1 through 9.

This creates digit_model.npz from synthetic digit samples, avoiding TensorFlow
startup memory issues on Render free instances.
"""

import os
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "digit_model.npz")).resolve()
DIGITS = range(1, 10)
FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
]


def render_digit(digit: int, font: int, scale: float, thickness: int, angle: float, dx: int, dy: int) -> np.ndarray:
    canvas = np.zeros((28, 28), dtype=np.uint8)
    text = str(digit)
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = (28 - tw) // 2 + dx
    y = (28 + th) // 2 + dy
    cv2.putText(canvas, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)

    matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    canvas = cv2.warpAffine(canvas, matrix, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return canvas.astype("float32") / 255.0


def normalize_sample(sample: np.ndarray) -> np.ndarray:
    vector = sample.reshape(-1).astype("float32")
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def main() -> None:
    samples = []
    labels = []

    for digit in DIGITS:
        for font in FONTS:
            for scale in (0.75, 0.85, 0.95, 1.05, 1.15):
                for thickness in (1, 2, 3):
                    for angle in (-14, -8, -4, 0, 4, 8, 14):
                        for dx, dy in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)):
                            sample = render_digit(digit, font, scale, thickness, angle, dx, dy)
                            samples.append(normalize_sample(sample))
                            labels.append(digit)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_PATH,
        samples=np.asarray(samples, dtype="float32"),
        labels=np.asarray(labels, dtype="int64"),
    )
    print(f"Saved lightweight digit model to {MODEL_PATH} with {len(labels)} samples")


if __name__ == "__main__":
    main()

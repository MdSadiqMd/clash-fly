"""Find buttons by template matching and read numbers by OCR.

Templates live in assets/templates/<name>.png and must be cropped from
screenshots at the phone's native resolution (never scaled).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from flyclash.config import ASSETS

Point = tuple[int, int]


@lru_cache(maxsize=64)
def load_template(name: str, directory: Path = ASSETS / "templates") -> np.ndarray:
    path = directory / f"{name}.png"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"template missing: {path}")
    return img


def crop(frame: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = region
    return frame[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


def find(
    frame: np.ndarray, template: np.ndarray, threshold: float = 0.85
) -> Point | None:
    """Centre of the best match if its score clears the threshold."""
    if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
        return None
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < threshold:
        return None
    th, tw = template.shape[:2]
    return loc[0] + tw // 2, loc[1] + th // 2


def find_all(
    frame: np.ndarray, template: np.ndarray, threshold: float = 0.85
) -> list[Point]:
    """All non-overlapping match centres (used for counting stars)."""
    if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
        return []
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    th, tw = template.shape[:2]
    hits: list[Point] = []
    ys, xs = np.where(res >= threshold)
    for y, x in sorted(zip(ys, xs), key=lambda p: -res[p[0], p[1]]):
        if all(abs(x - hx) >= tw // 2 or abs(y - hy) >= th // 2 for hx, hy in hits):
            hits.append((int(x), int(y)))
    return [(x + tw // 2, y + th // 2) for x, y in hits]


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr_onnxruntime import (
        RapidOCR,
    )  # optional dependency, pip install flyclash[ocr]

    return RapidOCR()


def ocr_text(img: np.ndarray) -> str:
    result, _ = _ocr_engine()(img)
    return " ".join(r[1] for r in result or [])


_NUM = re.compile(r"[-+]?\d[\d,]*")


def ocr_number(img: np.ndarray) -> int | None:
    """First integer in the OCR output, commas stripped. None if nothing readable."""
    m = _NUM.search(ocr_text(img))
    return int(m.group().replace(",", "")) if m else None

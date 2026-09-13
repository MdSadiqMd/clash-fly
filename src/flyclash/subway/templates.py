"""Sprite-template vision for Subway Surfers: the game has a fixed camera and a small set
of fixed obstacle sprites, so multi-scale template matching beats any colour heuristic.

Templates are auto-cropped once from saved gameplay frames (`build_templates`) using the
track geometry to locate the sprite regions, then matched per frame at several scales
inside the track area. Coins are saturated yellow blobs, no template needed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from flyclash.config import ASSETS
from flyclash.game.base_reader import Detection
from flyclash.subway.reader import Geometry

TEMPLATE_DIR = ASSETS / "templates" / "subway" / "obstacles"
SCALES = (0.5, 0.65, 0.8, 1.0, 1.3, 1.7)
MATCH_THRESHOLD = 0.50
# frame -> (kind, fractional crop box) pairs used to build the library once
CROP_SOURCES: list[tuple[str, str, tuple[float, float, float, float]]] = [
    # train fronts filling the centre lane at two distances (dark windshield + headlights)
    ("/tmp/deathdbg/ep0_03.jpg", "train", (0.38, 0.28, 0.62, 0.72)),
    ("/tmp/dbg/0000.jpg", "train", (0.42, 0.26, 0.58, 0.56)),
    ("/tmp/deathdbg/ep1_30.jpg", "train", (0.41, 0.42, 0.59, 0.78)),
    ("/tmp/deathdbg/ep2_05.jpg", "train", (0.39, 0.30, 0.60, 0.72)),  # tunnel lighting
    # red/white striped jump barrier, near and mid
    ("/tmp/diag2/060_290.jpg", "low_barrier", (0.40, 0.28, 0.60, 0.62)),
    ("/tmp/diag/039_376.jpg", "low_barrier", (0.41, 0.36, 0.56, 0.60)),
]


def build_templates(out_dir: Path = TEMPLATE_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (src, kind, (x0, y0, x1, y1)) in enumerate(CROP_SOURCES):
        f = cv2.imread(src)
        if f is None:
            continue
        h, w = f.shape[:2]
        crop = f[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
        p = out_dir / f"{kind}_{i}.png"
        cv2.imwrite(str(p), crop)
        written.append(p)
    return written


def load_templates(directory: Path = TEMPLATE_DIR) -> list[tuple[str, np.ndarray]]:
    out = []
    for p in sorted(directory.glob("*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            out.append((p.stem.rsplit("_", 1)[0], img))
    return out


def match_obstacles(
    frame: np.ndarray,
    templates: list[tuple[str, np.ndarray]],
    geo: Geometry,
    threshold: float = MATCH_THRESHOLD,
) -> list[Detection]:
    """Multi-scale TM_CCOEFF_NORMED over the track area; best hit per (lane, depth, kind)."""
    h, w = frame.shape[:2]
    y0, y1 = int(geo.far_y * h * 0.8), int(min(geo.near_y + 0.05, 1.0) * h)
    roi = frame[y0:y1]
    best: dict[tuple[int, int, str], Detection] = {}
    for kind, tpl in templates:
        for s in SCALES:
            tw, th = int(tpl.shape[1] * s), int(tpl.shape[0] * s)
            if tw < 8 or th < 8 or tw >= roi.shape[1] or th >= roi.shape[0]:
                continue
            tsc = cv2.resize(tpl, (tw, th))
            res = cv2.matchTemplate(roi, tsc, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= threshold)
            for yy, xx in zip(ys, xs):
                cx, cy = (
                    xx + tw / 2,
                    y0 + yy + th * 0.75,
                )  # anchor low: the sprite's base sits on the track
                rel = geo.rel_of(cx / w, cy / h)
                if not 0.0 <= rel < 1.0:
                    continue
                d = Detection(kind, cx, cy, float(res[yy, xx]))
                key = (geo.lane_of(cx / w, cy / h), geo.depth_of(cy / h), kind)
                if key not in best or best[key].conf < d.conf:
                    best[key] = d
    return list(best.values())


def coin_detections(frame: np.ndarray, geo: Geometry) -> list[Detection]:
    """Coins are small saturated-yellow blobs on the track."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (18, 150, 150), (34, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    dets = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (6 <= area <= 400 and bw <= 0.08 * w):
            continue
        cx, cy = cents[i]
        if (
            0.0 <= geo.rel_of(cx / w, cy / h) < 1.0
            and geo.far_y * h < cy < geo.near_y * h
        ):
            dets.append(Detection("coin", cx, cy, 0.9))
    return dets

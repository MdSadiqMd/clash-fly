"""Turn a scout screenshot into the 53-dim vector the fly brain smells.

53 = number of ORN glomerulus types in FlyWire. Layout of the vector:
  [0:32]  8 sectors around the base centre x 4 defense groups, distance-weighted counts
  [32:37] gold, elixir, dark elixir, trophies, town hall level (all scaled to ~[0,1])
  [37:53] zero padding, reserved for later defense classes
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

N_GLOM = 53
N_SECTORS = 8
GROUPS = ("point", "splash", "air", "other")
CLASS_GROUP = {
    "Cannon": "point",
    "ArcherTower": "point",
    "Xbow": "point",
    "Mortar": "splash",
    "WizardTower": "splash",
    "BombTower": "splash",
    "Scattershot": "splash",
    "AirDefense": "air",
    "AirSweeper": "air",
    "Inferno": "other",
    "Eagle": "other",
    "ClanCastle": "other",
    "Tesla": "other",
    "HeroPad": "other",
}
CANONICAL_CLASSES = sorted(CLASS_GROUP) + ["TownHall"]
# Names used by public datasets (find-this-base, its Hugging Face mirror, Roboflow forks) -> ours.
# Keys are compared after lowercasing and stripping non-letters, so "Archer Tower", "archer-tower", "ArcherTower" all match.
_ALIASES = {
    "canon": "Cannon",
    "wizztower": "WizardTower",
    "ad": "AirDefense",
    "th": "TownHall",
    "townhall": "TownHall",
    "kingpad": "HeroPad",
    "queenpad": "HeroPad",
    "rcpad": "HeroPad",
    "wardenpad": "HeroPad",
    "hiddentesla": "Tesla",
    "airdefence": "AirDefense",
    "infernotower": "Inferno",
    "eagleartillery": "Eagle",
}
_ALIASES.update({re.sub(r"[^a-z]", "", c.lower()): c for c in CANONICAL_CLASSES})
_ALIASES.update({f"th{i}": "TownHall" for i in range(1, 20)})


def canonical_class(name: str) -> str | None:
    """Our class name for a dataset/model label, or None if we do not model it."""
    return _ALIASES.get(re.sub(r"[^a-z0-9]", "", name.lower()))


EDGES = ("TL", "BL", "BR", "TR")
# Screen angle (atan2 with y down) at the middle of each isometric edge
EDGE_ANGLE = {
    "TR": -math.pi / 4,
    "BR": math.pi / 4,
    "BL": 3 * math.pi / 4,
    "TL": -3 * math.pi / 4,
}


@dataclass
class Detection:
    cls: str
    x: float
    y: float
    conf: float = 1.0


@dataclass
class Loot:
    gold: int = 0
    elixir: int = 0
    dark: int = 0
    trophies: int = 0


@dataclass
class BaseObs:
    vector: np.ndarray
    detections: list[Detection]
    center: tuple[float, float]
    edge_points: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    loot: Loot = field(default_factory=Loot)
    th_level: int = 0


def red_line_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask of the red deploy boundary (two hue bands because red wraps in HSV)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, (0, 120, 120), (8, 255, 255))
    hi = cv2.inRange(hsv, (172, 120, 120), (180, 255, 255))
    mask = cv2.bitwise_or(lo, hi)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def red_line_edges(
    mask: np.ndarray, n_points: int = 12
) -> tuple[tuple[float, float], dict[str, list[tuple[int, int]]]]:
    """Centre of the red diamond and n_points sample points along each of its 4 edges.

    Pixels are assigned to an edge by their angle from the diamond centre; MyBot
    does the same split into TL/BL/BR/TR before choosing a drop side.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        h, w = mask.shape
        return (w / 2, h / 2), {}
    cx, cy = float(xs.mean()), float(ys.mean())
    ang = np.arctan2(ys - cy, xs - cx)
    edges: dict[str, list[tuple[int, int]]] = {}
    for name, a0 in EDGE_ANGLE.items():
        d = np.abs((ang - a0 + math.pi) % (2 * math.pi) - math.pi)
        sel = d < math.pi / 4
        if sel.sum() < 5:
            continue
        pts = np.stack([xs[sel], ys[sel]], 1)
        # order along the edge, then take evenly spaced samples
        axis = pts[:, 0] - pts[:, 1] if name in ("TL", "BR") else pts[:, 0] + pts[:, 1]
        pts = pts[np.argsort(axis)]
        idx = np.linspace(0, len(pts) - 1, n_points).astype(int)
        edges[name] = [(int(x), int(y)) for x, y in pts[idx]]
    return (cx, cy), edges


def detect(frame: np.ndarray, weights: Path, conf: float = 0.4) -> list[Detection]:
    """YOLO detections; empty list when no weights are trained yet."""
    if not Path(weights).exists():
        return []
    from ultralytics import YOLO  # optional dependency, pip install flyclash[yolo]

    model = YOLO(str(weights))
    out = model.predict(frame, conf=conf, verbose=False)[0]
    names = out.names
    dets = []
    for box in out.boxes:
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        name = canonical_class(names[int(box.cls)]) or names[int(box.cls)]
        dets.append(Detection(name, (x0 + x1) / 2, (y0 + y1) / 2, float(box.conf)))
    return dets


def vectorize(
    dets: list[Detection],
    center: tuple[float, float],
    loot: Loot,
    th_level: int,
    scale: float,
) -> np.ndarray:
    """53-dim base vector. `scale` is a pixel distance (roughly the diamond half-width)."""
    v = np.zeros(N_GLOM, dtype=np.float32)
    cx, cy = center
    for d in dets:
        g = CLASS_GROUP.get(d.cls)
        if g is None:
            continue
        ang = math.atan2(d.y - cy, d.x - cx)
        sector = int(((ang + math.pi) / (2 * math.pi)) * N_SECTORS) % N_SECTORS
        dist = math.hypot(d.x - cx, d.y - cy) / max(scale, 1.0)
        v[sector * len(GROUPS) + GROUPS.index(g)] += d.conf * math.exp(-dist)
    v[:32] = np.tanh(v[:32])
    v[32] = math.log1p(loot.gold) / math.log1p(1_000_000)
    v[33] = math.log1p(loot.elixir) / math.log1p(1_000_000)
    v[34] = math.log1p(loot.dark) / math.log1p(10_000)
    v[35] = np.clip(loot.trophies / 50.0, -1, 1)
    v[36] = th_level / 17.0
    return v


def defense_pressure_per_edge(vector: np.ndarray) -> dict[str, float]:
    """Sum of defense features in the sectors facing each edge; used as the innate prior."""
    per_sector = vector[:32].reshape(N_SECTORS, len(GROUPS)).sum(1)
    out = {}
    for name, a0 in EDGE_ANGLE.items():
        s = int(((a0 + math.pi) / (2 * math.pi)) * N_SECTORS) % N_SECTORS
        out[name] = float(
            per_sector[s]
            + 0.5 * per_sector[(s - 1) % N_SECTORS]
            + 0.5 * per_sector[(s + 1) % N_SECTORS]
        )
    return out


def weakest_edge(vector: np.ndarray) -> str:
    p = defense_pressure_per_edge(vector)
    return min(p, key=p.get)


def read_base(frame: np.ndarray, loot: Loot, th_level: int, weights: Path) -> BaseObs:
    mask = red_line_mask(frame)
    center, edges = red_line_edges(mask)
    dets = detect(frame, weights)
    for d in dets:
        if d.cls == "TownHall":
            center = (d.x, d.y)
    scale = frame.shape[1] / 4
    return BaseObs(
        vectorize(dets, center, loot, th_level, scale),
        dets,
        center,
        edges,
        loot,
        th_level,
    )

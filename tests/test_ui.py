"""Template matching and base vectorisation on synthetic images (no phone needed)."""

import math

import numpy as np

from flyclash.game import ui
from flyclash.game.base_reader import (
    Detection,
    Loot,
    red_line_edges,
    red_line_mask,
    vectorize,
    weakest_edge,
)


def _button(w=60, h=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_find_template_locates_button():
    frame = np.full((400, 600, 3), 40, np.uint8)
    btn = _button()
    frame[200:230, 300:360] = btn
    assert ui.find(frame, btn) == (330, 215)
    assert ui.find(frame, _button(seed=1)) is None


def test_find_all_counts_stars():
    frame = np.full((200, 400, 3), 40, np.uint8)
    star = _button(20, 20)
    for x in (50, 150, 250):
        frame[90:110, x : x + 20] = star
    assert len(ui.find_all(frame, star)) == 3


def test_red_line_edges_split_into_four_sides():
    frame = np.zeros((600, 800, 3), np.uint8)
    cx, cy, r = 400, 300, 200
    for t in np.linspace(0, 2 * math.pi, 2000):
        x = int(cx + r * math.cos(t))
        y = int(cy + r * 0.6 * math.sin(t))
        frame[y - 1 : y + 2, x - 1 : x + 2] = (0, 0, 255)  # BGR red
    center, edges = red_line_edges(red_line_mask(frame))
    assert abs(center[0] - cx) < 5 and abs(center[1] - cy) < 5
    assert set(edges) == {"TL", "BL", "BR", "TR"}
    assert all(len(v) == 12 for v in edges.values())


def test_vectorize_puts_defenses_in_sector_and_weakest_edge():
    center = (500.0, 500.0)
    # isometric edges are diagonal: BR sits at +45deg (screen y down), TL at -135deg
    dets = [
        Detection("Cannon", 650, 650),
        Detection("Cannon", 700, 700),
        Detection("Mortar", 650, 350),
        Detection("ArcherTower", 350, 650),
    ]
    v = vectorize(dets, center, Loot(100000, 50000, 0, 20), 5, scale=250)
    assert v.shape == (53,)
    assert v[:32].sum() > 0 and v[37:].sum() == 0
    assert 0 < v[32] < 1 and abs(v[35] - 0.4) < 1e-6
    assert weakest_edge(v) == "TL"

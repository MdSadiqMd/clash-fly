"""Detector classes for Subway Surfers and how they map onto the fly's input."""

from __future__ import annotations

import re

SUBWAY_CLASSES = [
    "train",
    "wall",
    "boxtrain",
    "low_barrier",
    "high_barrier",
    "coin",
    "player",
    "powerup",
]

# obstacle kind index used in the 53-dim vector: 0 = must change lane, 1 = jump over, 2 = roll under
KIND = {"train": 0, "wall": 0, "boxtrain": 0, "low_barrier": 1, "high_barrier": 2}
N_KINDS = 3

_ALIASES = {
    "train": "train",
    "trains": "train",
    "wall": "wall",
    "boxtrain": "boxtrain",
    "box": "boxtrain",
    "jumpbarrier": "low_barrier",
    "lowbarrier": "low_barrier",
    "barrierlow": "low_barrier",
    "jump": "low_barrier",
    "highbarrier": "high_barrier",
    "barrierhigh": "high_barrier",
    "rollbarrier": "high_barrier",
    "duck": "high_barrier",
    "barrier": "high_barrier",
    "coin": "coin",
    "coins": "coin",
    "player": "player",
    "jake": "player",
    "character": "player",
    "bootspowerup": "powerup",
    "powerup": "powerup",
    "jetpack": "powerup",
    "magnet": "powerup",
    "multiplier": "powerup",
    "hoverboard": "powerup",
}


def canonical_class(name: str) -> str | None:
    key = re.sub(r"[^a-z]", "", name.lower())
    return _ALIASES.get(key)

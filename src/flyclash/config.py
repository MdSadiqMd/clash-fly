"""Screen layout and timing constants.

Regions are fractions of the screen (x0, y0, x1, y1) so one layout file works
across phone resolutions approximately; calibrate once with `flyclash calibrate`
and edit assets/layout.json for the exact phone
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
LAYOUT_PATH = ASSETS / "layout.json"

Region = tuple[float, float, float, float]


@dataclass
class Layout:
    # Scout screen: available loot block, top-left (MyBot reads gold/elixir/DE/trophies there)
    loot_gold: Region = (0.02, 0.06, 0.20, 0.10)
    loot_elixir: Region = (0.02, 0.10, 0.20, 0.14)
    loot_dark: Region = (0.02, 0.14, 0.20, 0.18)
    loot_trophy: Region = (0.02, 0.18, 0.20, 0.22)
    # Battle result screen
    result_percent: Region = (0.40, 0.30, 0.60, 0.38)
    result_gold: Region = (0.35, 0.45, 0.65, 0.50)
    result_elixir: Region = (0.35, 0.50, 0.65, 0.55)
    result_stars: Region = (0.30, 0.15, 0.70, 0.30)
    # Troop bar: y band at the bottom, evenly spaced slot centres between x0 and x1
    troop_bar_y: float = 0.92
    troop_bar_x0: float = 0.08
    troop_bar_x1: float = 0.92
    troop_bar_slots: int = 6
    troop_count_dy: float = -0.05  # count label sits above the icon
    # Template match threshold
    match_threshold: float = 0.85


@dataclass
class Timing:
    home: float = 10.0
    attack_window: float = 10.0
    clouds: float = 120.0
    decide: float = 20.0
    decide_hard: float = 25.0
    analyse: float = 28.0
    deploy: float = 150.0
    end: float = 30.0
    unknown: float = 60.0
    max_skips: int = (
        15  # Next presses per search before attacking anyway (each costs gold)
    )
    min_attack_interval: float = 240.0
    tap_delay: tuple[float, float] = (0.3, 1.2)
    session_max: float = 40 * 60


@dataclass
class Config:
    layout: Layout = field(default_factory=Layout)
    timing: Timing = field(default_factory=Timing)
    yolo_weights: Path = ASSETS / "yolo" / "best.pt"
    templates: Path = ASSETS / "templates"
    circuit_cache: Path = DATA / "circuit.npz"
    plastic_path: Path = DATA / "plastic.npz"
    log_path: Path = DATA / "attacks.jsonl"


def load_config(path: Path = LAYOUT_PATH) -> Config:
    cfg = Config()
    if path.exists():
        raw = json.loads(path.read_text())
        for k, v in raw.get("layout", {}).items():
            setattr(cfg.layout, k, tuple(v) if isinstance(v, list) else v)
        for k, v in raw.get("timing", {}).items():
            setattr(cfg.timing, k, tuple(v) if isinstance(v, list) else v)
    return cfg

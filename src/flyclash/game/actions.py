"""High-level game actions on top of ADB + templates.

Every method re-screenshots before acting so the agent never taps blind.
Required templates in assets/templates (crop from your phone once):
  attack_button, find_match, next_button, end_battle, okay, return_home,
  multiplayer_tab, star, home_marker (any fixed home-screen element)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from flyclash.config import Config
from flyclash.game import ui
from flyclash.game.adb import Adb
from flyclash.game.base_reader import Loot


@dataclass
class Result:
    stars: int = 0
    percent: int = 0
    gold: int = 0
    elixir: int = 0


class GameIO(Protocol):
    """What the agent loop needs; RealGame and the test FakeGame both satisfy it."""

    def at_home(self) -> bool: ...
    def open_attack_window(self) -> bool: ...
    def find_match(self) -> bool: ...
    def next_base(self) -> bool: ...
    def scout_frame(self) -> np.ndarray | None: ...
    def scout_loot(self, frame: np.ndarray) -> Loot | None: ...
    def troop_counts(self) -> list[int]: ...
    def drop(self, slot: int, points: list[tuple[int, int]]) -> None: ...
    def end_battle(self) -> bool: ...
    def result(self) -> Result | None: ...
    def return_home(self) -> bool: ...


class RealGame:
    def __init__(self, adb: Adb, cfg: Config, rng: random.Random | None = None):
        self.adb, self.cfg, self.rng = adb, cfg, rng or random.Random()
        self.layout = cfg.layout

    def _pause(self) -> None:
        time.sleep(self.rng.uniform(*self.cfg.timing.tap_delay))

    def _tap_template(self, name: str, frame: np.ndarray | None = None) -> bool:
        frame = self.adb.screencap() if frame is None else frame
        pt = ui.find(
            frame,
            ui.load_template(name, self.cfg.templates),
            self.layout.match_threshold,
        )
        if pt is None:
            return False
        self._pause()
        self.adb.tap(*pt)
        return True

    def _see(self, name: str, frame: np.ndarray | None = None) -> bool:
        frame = self.adb.screencap() if frame is None else frame
        return (
            ui.find(
                frame,
                ui.load_template(name, self.cfg.templates),
                self.layout.match_threshold,
            )
            is not None
        )

    def at_home(self) -> bool:
        return self._see("attack_button")

    def open_attack_window(self) -> bool:
        return self._tap_template("attack_button")

    def find_match(self) -> bool:
        frame = self.adb.screencap()
        if not self._see("multiplayer_tab", frame):
            self._tap_template("multiplayer_tab", frame)
            frame = self.adb.screencap()
        return self._tap_template("find_match", frame)

    def next_base(self) -> bool:
        return self._tap_template("next_button")

    def scout_frame(self) -> np.ndarray | None:
        return self.adb.screencap()

    def scout_loot(self, frame: np.ndarray) -> Loot | None:
        gold = ui.ocr_number(ui.crop(frame, self.layout.loot_gold))
        elixir = ui.ocr_number(ui.crop(frame, self.layout.loot_elixir))
        if gold is None or elixir is None:
            return None
        dark = ui.ocr_number(ui.crop(frame, self.layout.loot_dark)) or 0
        trophies = ui.ocr_number(ui.crop(frame, self.layout.loot_trophy)) or 0
        return Loot(gold, elixir, dark, trophies)

    def _slot_xy(self, slot: int, frame: np.ndarray) -> tuple[int, int]:
        h, w = frame.shape[:2]
        L = self.layout
        xs = np.linspace(L.troop_bar_x0, L.troop_bar_x1, L.troop_bar_slots)
        return int(xs[slot] * w), int(L.troop_bar_y * h)

    def troop_counts(self) -> list[int]:
        frame = self.adb.screencap()
        h, w = frame.shape[:2]
        counts = []
        for s in range(self.layout.troop_bar_slots):
            x, y = self._slot_xy(s, frame)
            y += int(self.layout.troop_count_dy * h)
            box = frame[
                max(0, y - int(0.02 * h)) : y + int(0.02 * h),
                max(0, x - int(0.04 * w)) : x + int(0.04 * w),
            ]
            counts.append(ui.ocr_number(box) or 0)
        return counts

    def drop(self, slot: int, points: list[tuple[int, int]]) -> None:
        frame = self.adb.screencap()
        self.adb.tap(*self._slot_xy(slot, frame))
        for x, y in points:
            time.sleep(self.rng.uniform(0.08, 0.2))
            self.adb.tap(x, y)

    def end_battle(self) -> bool:
        if self._tap_template("end_battle"):
            time.sleep(1.0)
            self._tap_template("okay")
            return True
        return False

    def result(self) -> Result | None:
        frame = self.adb.screencap()
        if not self._see("return_home", frame):
            return None
        stars = len(
            ui.find_all(
                ui.crop(frame, self.layout.result_stars),
                ui.load_template("star", self.cfg.templates),
                self.layout.match_threshold,
            )
        )
        return Result(
            stars=min(stars, 3),
            percent=ui.ocr_number(ui.crop(frame, self.layout.result_percent)) or 0,
            gold=ui.ocr_number(ui.crop(frame, self.layout.result_gold)) or 0,
            elixir=ui.ocr_number(ui.crop(frame, self.layout.result_elixir)) or 0,
        )

    def return_home(self) -> bool:
        return self._tap_template("return_home")

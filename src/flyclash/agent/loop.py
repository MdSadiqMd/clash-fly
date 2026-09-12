"""The attack loop as a state machine with hard deadlines.

HOME -> ATTACK_WINDOW -> CLOUDS -> SCOUT_DECIDE -> (NEXT -> CLOUDS | ANALYSE -> DEPLOY -> END) -> LEARN -> HOME
Any state that misses its deadline falls back to return_home(); an unknown screen
for `timing.unknown` seconds stops the agent instead of tapping blindly.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np

from flyclash.agent.reward import compute_reward
from flyclash.brain.learn import Policy
from flyclash.config import Config
from flyclash.game.actions import GameIO
from flyclash.game.base_reader import BaseObs, Loot, weakest_edge


class State(Enum):
    HOME = auto()
    ATTACK_WINDOW = auto()
    CLOUDS = auto()
    SCOUT_DECIDE = auto()
    ANALYSE = auto()
    DEPLOY = auto()
    END = auto()
    LEARN = auto()
    ABORT = auto()


class Stuck(RuntimeError):
    pass


@dataclass
class AttackRecord:
    attacked: bool
    skips: int
    edge: str
    pattern: str
    troop_order: str
    score: float
    stars: int
    percent: int
    reward: float
    delta: float
    elapsed_decide: float
    elapsed_total: float
    loot: dict
    n_kc_active: int


def drop_points(
    obs: BaseObs, edge: str, pattern: str, frame_shape: tuple[int, int]
) -> list[tuple[int, int]]:
    """Screen points for a drop along `edge`. Falls back to a diamond inscribed in the
    screen when the red line was not found."""
    pts = obs.edge_points.get(edge)
    if not pts:
        h, w = frame_shape
        cx, cy, r = w / 2, h / 2, min(w, h) * 0.42
        corner = {
            "TL": (cx - r, cy - r * 0.55),
            "TR": (cx + r, cy - r * 0.55),
            "BL": (cx - r, cy + r * 0.55),
            "BR": (cx + r, cy + r * 0.55),
        }
        a, b = {
            "TL": ("TL", "BL"),
            "BL": ("BL", "BR"),
            "BR": ("BR", "TR"),
            "TR": ("TR", "TL"),
        }[edge]
        pts = [
            (
                int(corner[a][0] + (corner[b][0] - corner[a][0]) * t),
                int(corner[a][1] + (corner[b][1] - corner[a][1]) * t),
            )
            for t in np.linspace(0.15, 0.85, 12)
        ]
    if pattern == "line":
        return pts
    if pattern == "pincer":
        return [pts[2], pts[-3]]
    return [pts[len(pts) // 2]]


class Agent:
    def __init__(
        self,
        game: GameIO,
        policy: Policy,
        cfg: Config,
        read_base: Callable[[np.ndarray, Loot], BaseObs],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        publish: Callable[[dict], None] | None = None,
    ):
        self.game, self.policy, self.cfg, self.read_base = game, policy, cfg, read_base
        self.clock, self.sleep, self.publish = clock, sleep, publish or (lambda d: None)
        self.t = cfg.timing
        self.state = State.HOME

    def _wait(self, pred: Callable[[], object], timeout: float, poll: float = 0.5):
        t0 = self.clock()
        while self.clock() - t0 < timeout:
            v = pred()
            if v:
                return v
            self.sleep(poll)
        return None

    def _set(self, state: State, **info) -> None:
        self.state = state
        self.publish({"state": state.name, **info})

    def _bail(self, reason: str) -> None:
        self._set(State.ABORT, reason=reason)
        self.game.end_battle()
        if (
            not self._wait(self.game.at_home, self.t.end)
            and not self.game.return_home()
        ):
            raise Stuck(reason)

    def run_once(self, troop_slots: tuple[int, int] = (0, 1)) -> AttackRecord:
        t_start = self.clock()
        self._set(State.HOME)
        if not self._wait(self.game.at_home, self.t.home):
            raise Stuck("not at home screen")
        self._set(State.ATTACK_WINDOW)
        if not self.game.open_attack_window() or not self._wait(
            self.game.find_match, self.t.attack_window
        ):
            self._bail("attack window / find match not found")
            raise Stuck("could not start a search")

        skips = 0
        while True:
            self._set(State.CLOUDS, skips=skips)
            found = self._wait(self._scout, self.t.clouds)
            if not found:
                self._bail("clouds never cleared")
                raise Stuck("clouds never cleared")
            frame, loot, t0 = found
            self._set(State.SCOUT_DECIDE)
            obs = self.read_base(frame, loot)
            act, d = self.policy.decide(
                obs.vector, innate_edge=weakest_edge(obs.vector)
            )
            elapsed_decide = self.clock() - t0
            self.publish(
                {
                    "state": "SCOUT_DECIDE",
                    "kc_idx": act.kc_idx.tolist(),
                    "mbon": act.mbon.tolist(),
                    "score": d.score,
                    "attack": d.attack,
                    "elapsed": elapsed_decide,
                }
            )
            if elapsed_decide > self.t.decide_hard:
                d.attack = True  # too late to press Next: the battle will auto-start, so attack with what we have
            if skips >= self.t.max_skips:
                d.attack = True
            if d.attack:
                break
            delta = self.policy.learn_skip(act)
            skips += 1
            self.publish({"state": "NEXT", "delta": delta})
            if not self.game.next_base():
                d.attack = True
                break

        self._set(State.ANALYSE, edge=d.edge, pattern=d.pattern)
        points = drop_points(obs, d.edge, d.pattern, frame.shape[:2])
        self._set(State.DEPLOY)
        t_deploy = self.clock()
        self._deploy(points, d.troop_order, troop_slots)
        self._wait(
            lambda: (
                self.clock() - t_deploy > self.t.deploy
                or sum(self.game.troop_counts()) == 0
            ),
            self.t.deploy,
            poll=5.0,
        )

        self._set(State.END)
        self.game.end_battle()
        result = self._wait(self.game.result, self.t.end)
        self.game.return_home()
        if result is None:
            self._bail("no result screen")
            raise Stuck("no result screen")

        self._set(State.LEARN)
        reward = compute_reward(result, loot)
        delta = self.policy.learn_attack(act, d, reward)
        rec = AttackRecord(
            True,
            skips,
            d.edge,
            d.pattern,
            d.troop_order,
            d.score,
            result.stars,
            result.percent,
            reward,
            delta,
            elapsed_decide,
            self.clock() - t_start,
            asdict(loot),
            len(act.kc_idx),
        )
        self._log(rec)
        self.publish({"state": "LEARN", **asdict(rec)})
        return rec

    def _scout(self):
        frame = self.game.scout_frame()
        if frame is None:
            return None
        loot = self.game.scout_loot(frame)
        return (frame, loot, self.clock()) if loot is not None else None

    def _deploy(
        self, points: list[tuple[int, int]], order: str, slots: tuple[int, int]
    ) -> None:
        counts = self.game.troop_counts()
        first, second = slots
        seq = {
            "barch": [first, second],
            "archers_first": [second, first],
            "all": [first, second],
        }[order]
        for slot in seq:
            n = counts[slot] if slot < len(counts) else 0
            if n <= 0:
                continue
            taps = [points[i % len(points)] for i in range(n)]
            self.game.drop(slot, taps)

    def _log(self, rec: AttackRecord) -> None:
        self.cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.log_path, "a") as f:
            f.write(json.dumps({"t": time.time(), **asdict(rec)}) + "\n")

    def run(
        self, n_attacks: int, plastic_path: Path | None = None
    ) -> list[AttackRecord]:
        records = []
        session_start = self.clock()
        for _ in range(n_attacks):
            if self.clock() - session_start > self.t.session_max:
                break
            t0 = self.clock()
            records.append(self.run_once())
            if plastic_path is not None:
                self.policy.p.save(plastic_path)
            self.sleep(max(0.0, self.t.min_attack_interval - (self.clock() - t0)))
        return records

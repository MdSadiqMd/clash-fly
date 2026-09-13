"""Frame detections -> the 53-dim vector the fly smells, plus the innate reflex prior.

Vector layout (53 = number of ORN glomerulus types in FlyWire):
  [0:45]  3 lanes x 5 depth bins (near -> far) x 3 obstacle kinds, max confidence per cell
  [45:48] coins per lane (tanh of count)
  [48:51] player lane one-hot
  [51]    speed proxy (score gained per second, scaled)
  [52]    powerup active
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from flyclash.game.base_reader import Detection
from flyclash.subway.classes import KIND, N_KINDS

N_GLOM = 53
N_LANES, N_DEPTH = 3, 5
ACTIONS = ("left", "right", "jump", "roll", "none")


@dataclass
class Geometry:
    """Where the track sits in the canvas, as fractions of its size. The track is a fixed
    perspective: lanes are a third of `width_at(y)` each, centred on `center_x`."""

    # measured on the poki.com canvas at 1280x800 viewport (see docs/subway-plan.md)
    near_y: float = 0.80  # y where the player stands
    far_y: float = 0.35  # y where obstacles first appear
    near_width: float = 0.55  # track width at near_y
    far_width: float = 0.12  # track width at far_y
    center_x: float = 0.5

    def width_at(self, yf: float) -> float:
        t = (self.near_y - yf) / max(self.near_y - self.far_y, 1e-6)
        return self.near_width + (self.far_width - self.near_width) * min(
            max(t, 0.0), 1.0
        )

    def rel_of(self, xf: float, yf: float) -> float:
        """Cross-track position: 0..1 on the track, outside that range = roadside scenery."""
        return (xf - self.center_x) / max(self.width_at(yf), 1e-6) + 0.5

    def lane_of(self, xf: float, yf: float) -> int:
        return int(min(max(self.rel_of(xf, yf) * N_LANES, 0), N_LANES - 1))

    def depth_of(self, yf: float) -> int:
        t = (self.near_y - yf) / max(self.near_y - self.far_y, 1e-6)
        return int(min(max(t * N_DEPTH, 0), N_DEPTH - 1))


def cell(lane: int, depth: int, kind: int) -> int:
    return (lane * N_DEPTH + depth) * N_KINDS + kind


def vectorize(
    dets: list[Detection],
    frame_wh: tuple[int, int],
    geo: Geometry,
    player_lane: int,
    speed: float,
    powerup: bool,
) -> tuple[np.ndarray, int]:
    """Returns the vector and the player lane actually used (detected if a 'player' box exists)."""
    w, h = frame_wh
    v = np.zeros(N_GLOM, np.float32)
    coins = np.zeros(N_LANES, np.float32)
    for d in dets:
        xf, yf = d.x / w, d.y / h
        if d.cls == "player":
            player_lane = geo.lane_of(xf, geo.near_y)
            continue
        rel = geo.rel_of(xf, yf)
        if not 0.0 <= rel < 1.0:
            continue  # roadside scenery (market umbrellas read as striped barriers otherwise)
        lane = geo.lane_of(xf, yf)
        if d.cls == "coin":
            coins[lane] += 1
        elif d.cls in KIND:
            i = cell(lane, geo.depth_of(yf), KIND[d.cls])
            v[i] = max(v[i], d.conf)
    v[45:48] = np.tanh(coins / 3)
    v[48 + int(min(max(player_lane, 0), N_LANES - 1))] = 1.0
    v[51] = min(speed / 200.0, 1.0)
    v[52] = float(powerup)
    return v, player_lane


def cells(v: np.ndarray) -> np.ndarray:
    return v[:45].reshape(N_LANES, N_DEPTH, N_KINDS)


def innate_prior(v: np.ndarray, player_lane: int) -> np.ndarray:
    """Reflex bias over ACTIONS. React EARLY: at ~10 decisions/s a hazard crosses the near
    depth bins in one or two frames, so waiting for depth 0-1 dodges too late (the fly was
    seen dying to trains it had already detected). Jump/roll fire from depth 0-2, a blocker
    triggers a lane change from depth 0-3, and once committed the action is decisive."""
    c = cells(v)
    prior = np.zeros(len(ACTIONS), np.float32)
    # barrier ahead in this lane within the reaction window -> jump / roll pre-emptively
    jump_evidence = float(c[player_lane, :3, 1].max())
    roll_evidence = float(c[player_lane, :3, 2].max())
    if jump_evidence > 0.3:
        prior[ACTIONS.index("jump")] += 1.0 + jump_evidence
    if roll_evidence > 0.3:
        prior[ACTIONS.index("roll")] += 1.0 + roll_evidence
    # solid blocker (train/wall/crate) anywhere in the near four bins -> get out of the lane
    blocker = float(c[player_lane, :4, 0].max())
    if blocker > 0.3:
        # pick the safer neighbour by its own near-bin hazard load; ties go outward from centre
        load = c[:, :4].sum((1, 2))
        options = [(load[player_lane - 1], "left")] if player_lane > 0 else []
        options += (
            [(load[player_lane + 1], "right")] if player_lane < N_LANES - 1 else []
        )
        best = min(options, key=lambda o: o[0])
        prior[ACTIONS.index(best[1])] += 1.5 * blocker / (1.0 + float(best[0]))
    if prior.sum() == 0:
        # clear track: chase the richest coin lane (visible purpose between hazards),
        # otherwise hold lane strongly enough to beat circuit-head drift (damped to ~0.1)
        coins = v[45:48]
        neighbours = [
            ln for ln in (player_lane - 1, player_lane + 1) if 0 <= ln < N_LANES
        ]
        best = max(neighbours, key=lambda ln: coins[ln], default=None)
        if best is not None and coins[best] > coins[player_lane] + 0.25:
            prior[ACTIONS.index("left" if best < player_lane else "right")] = 0.45
        else:
            prior[ACTIONS.index("none")] = 0.3
    return prior

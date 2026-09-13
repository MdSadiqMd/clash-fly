"""Offline Subway Surfers: three lanes, obstacles scroll toward the player, same 53-dim vector.

Rules: a blocker (train) in the player's lane at depth 0 kills unless the lane changed;
a low barrier kills unless jumping; a high barrier kills unless rolling; coins are picked
up when reached. Speed rises with distance. This is the training ground for the fly and
the place where the real-vs-random wiring ablation is measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyclash.subway.reader import ACTIONS, N_DEPTH, N_GLOM, N_KINDS, N_LANES, cell

STEP_REWARD = 0.01
COIN_REWARD = 0.1
DEATH_REWARD = -1.0


@dataclass
class Obstacle:
    lane: int
    depth: float  # in depth-bin units, 0 = at the player
    kind: int  # 0 blocker, 1 low (jump), 2 high (roll)


@dataclass
class Coin:
    lane: int
    depth: float


@dataclass
class SimEnv:
    seed: int = 0
    max_steps: int = 2000
    base_speed: float = 0.25  # depth bins per step
    hazard_rate: float = 0.12  # spawn probability per step
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.lane, self.steps, self.coins, self.jump_t, self.roll_t = 1, 0, 0, 0, 0
        self.obstacles: list[Obstacle] = []
        self.coin_items: list[Coin] = []
        self.score = 0.0
        return self.observe()

    @property
    def speed(self) -> float:
        return self.base_speed * (1 + self.steps / 1500)

    def observe(self) -> np.ndarray:
        v = np.zeros(N_GLOM, np.float32)
        for o in self.obstacles:
            d = int(min(max(o.depth, 0), N_DEPTH - 1))
            v[cell(o.lane, d, o.kind)] = 1.0
        coins = np.zeros(N_LANES)
        for c in self.coin_items:
            coins[c.lane] += 1
        v[45:48] = np.tanh(coins / 3)
        v[48 + self.lane] = 1.0
        v[51] = min(self.speed / 1.0, 1.0)
        return v

    def _spawn(self) -> None:
        if self.rng.random() < self.hazard_rate:
            lane = int(self.rng.integers(N_LANES))
            kind = int(self.rng.choice(N_KINDS, p=[0.5, 0.25, 0.25]))
            # never spawn a blocker in every lane at the same depth: keep one lane open
            far = [o for o in self.obstacles if o.depth > N_DEPTH - 1.5 and o.kind == 0]
            if (
                kind == 0
                and len({o.lane for o in far}) >= N_LANES - 1
                and lane not in {o.lane for o in far}
            ):
                kind = 1
            self.obstacles.append(Obstacle(lane, float(N_DEPTH), kind))
        if self.rng.random() < 0.15:
            self.coin_items.append(
                Coin(int(self.rng.integers(N_LANES)), float(N_DEPTH))
            )

    def step(self, action: str) -> tuple[np.ndarray, float, bool, dict]:
        reward = STEP_REWARD
        if action == "left" and self.lane > 0:
            self.lane -= 1
        elif action == "right" and self.lane < N_LANES - 1:
            self.lane += 1
        elif action == "jump":
            self.jump_t = 3
        elif action == "roll":
            self.roll_t = 3
        self.steps += 1
        self.score += 1
        dead = False
        for o in self.obstacles:
            o.depth -= self.speed
            if o.lane == self.lane and -0.5 <= o.depth < 0.5:
                if (
                    (o.kind == 0)
                    or (o.kind == 1 and self.jump_t <= 0)
                    or (o.kind == 2 and self.roll_t <= 0)
                ):
                    dead = True
        for c in self.coin_items:
            c.depth -= self.speed
            if c.lane == self.lane and -0.5 <= c.depth < 0.5:
                self.coins += 1
                reward += COIN_REWARD
                c.depth = -9
        self.obstacles = [o for o in self.obstacles if o.depth > -1]
        self.coin_items = [c for c in self.coin_items if c.depth > -1]
        self.jump_t, self.roll_t = max(self.jump_t - 1, 0), max(self.roll_t - 1, 0)
        self._spawn()
        if dead:
            reward += DEATH_REWARD
        done = dead or self.steps >= self.max_steps
        return (
            self.observe(),
            reward,
            done,
            {
                "dead": dead,
                "steps": self.steps,
                "coins": self.coins,
                "score": self.score,
            },
        )

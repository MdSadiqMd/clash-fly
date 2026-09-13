"""The RL loop. One policy, two environments (offline SimEnv and the live browser).

Environment interface: reset() -> vector, step(action) -> (vector, reward, done, info).
Policy: circuit.forward -> 5 action MBONs + innate reflex prior -> epsilon-greedy.
Learning: death punishes the eligibility trace (PPL1), coins/score credit it (PAM),
per-step reward prediction error is logged as the "loss".
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from flyclash.brain.circuit import Circuit
from flyclash.brain.learn import Plastic, Trace
from flyclash.config import DATA
from flyclash.subway.reader import ACTIONS, innate_prior

SUBWAY_DIR = DATA / "subway"


class Env(Protocol):
    def reset(self) -> np.ndarray: ...
    def step(self, action: str) -> tuple[np.ndarray, float, bool, dict]: ...


@dataclass
class SurfPolicy:
    circuit: Circuit
    plastic: Plastic
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    innate_weight: float = 1.0
    learning: bool = True
    trace: Trace = field(default_factory=Trace)
    baseline: float = 0.0  # running mean of per-step reward (the prediction)
    mbon_mean: np.ndarray | None = None  # running mean output per action MBON

    @property
    def action_mbons(self) -> np.ndarray:
        return self.circuit.action_mbons[: len(ACTIONS)]

    def act(self, v: np.ndarray, t: float) -> tuple[str, dict]:
        act = self.circuit.forward(v, self.plastic.w)
        raw = act.mbon[self.action_mbons].astype(np.float64)
        # Output adaptation: each action MBON is read relative to its own running mean. The five
        # MBON types receive very different structural KC input (up to 23x), and without this the
        # wiring, not the reflex or learning, picks the action. Measured: +41% survival on FlyWire wiring.
        if self.mbon_mean is None:
            self.mbon_mean = raw.copy() + 1e-9
        self.mbon_mean += 0.01 * (raw - self.mbon_mean)
        # damp the head's deviation from 1: with fresh weights the slow running mean drifts
        # between episodes and a ~1.2x ratio spike out-shouts the reflex for whole runs
        # (observed as one action spammed until death). Learned weight changes are much
        # larger than drift, so they still get through the 0.5 factor.
        scores = 1.0 + 0.5 * (raw / self.mbon_mean - 1.0)
        lane = int(np.argmax(v[48:51]))
        prior = innate_prior(v, lane)
        scores += self.innate_weight * prior
        if self.rng.random() < self.plastic.epsilon:
            a = int(self.rng.integers(len(ACTIONS)))
        else:
            a = int(np.argmax(scores))
        # record every action, "none" included: a crash while idle must punish idleness,
        # or repeated blind deaths depress only the moving actions and the fly freezes
        self.trace.add(t, act.kc_idx, int(self.action_mbons[a]))
        return ACTIONS[a], {"kc_idx": act.kc_idx, "scores": scores, "prior": prior}

    def learn(self, reward: float, dead: bool, t: float) -> float:
        """Returns the reward prediction error (logged as loss)."""
        delta = reward - self.baseline
        self.baseline += 0.05 * delta
        if not self.learning:
            return delta
        if dead:
            self.trace.punish(self.plastic, t, amount=1.0)
        elif delta > 0.05:
            self.trace.reward(self.plastic, t, amount=min(delta, 0.5))
        return delta


@dataclass
class EpisodeLog:
    episode: int
    steps: int
    reward: float
    coins: int
    score: float
    dead: bool
    loss: float  # mean |RPE| over the episode
    wall_s: float
    decisions_per_s: float


def run_episodes(
    env: Env,
    policy: SurfPolicy,
    n: int,
    log_dir: Path = SUBWAY_DIR,
    tag: str = "run",
    clock=time.monotonic,
    on_step=None,
    max_seconds: float | None = None,
    should_stop=None,
) -> list[EpisodeLog]:
    """Runs n episodes. `max_seconds` caps an episode by wall clock (the demo plays 30 s runs);
    `should_stop()` lets a UI abort between steps."""
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"{tag}.csv"
    new = not csv_path.exists()
    logs = []
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(EpisodeLog.__dataclass_fields__))
        if new:
            w.writeheader()
        for ep in range(n):
            v = env.reset()
            policy.trace.entries.clear()
            total, losses, done, info, t0 = 0.0, [], False, {}, clock()
            steps = 0
            while not done:
                t = clock()
                action, extra = policy.act(v, t)
                v, r, done, info = env.step(action)
                losses.append(abs(policy.learn(r, info.get("dead", False), clock())))
                total += r
                steps += 1
                elapsed = t - t0  # no extra clock() call: test clocks advance per call
                if on_step is not None:
                    on_step(
                        {
                            "state": "SURF",
                            "action": action,
                            "reward": r,
                            "kc_idx": extra["kc_idx"].tolist(),
                            "scores": extra["scores"].tolist(),
                            "steps": steps,
                            "score": info.get("score", 0),
                            "elapsed": elapsed,
                            "loss": losses[-1],
                        }
                    )
                if max_seconds is not None and elapsed >= max_seconds:
                    done = True
                    info = {**info, "timeout": True}
                if should_stop is not None and should_stop():
                    done = True
                    info = {**info, "stopped": True}
            wall = max(clock() - t0, 1e-6)
            log = EpisodeLog(
                ep,
                steps,
                total,
                int(info.get("coins", 0)),
                float(info.get("score", steps)),
                bool(info.get("dead", False)),
                float(np.mean(losses)) if losses else 0.0,
                wall,
                steps / wall,
            )
            w.writerow(asdict(log))
            f.flush()
            logs.append(log)
            policy.plastic.recover()
    (log_dir / f"{tag}.last.json").write_text(
        json.dumps([asdict(l) for l in logs[-5:]])
    )
    return logs


def plot_curves(csv_path: Path, png_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path)))
    ep = np.array([int(r["episode"]) for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for a, key, title in zip(
        ax,
        ("steps", "reward", "loss"),
        ("survival (steps)", "episode reward", "mean |RPE| (loss)"),
    ):
        y = np.array([float(r[key]) for r in rows])
        a.plot(ep, y, ".", alpha=0.4)
        if len(y) >= 10:
            k = max(len(y) // 10, 5)
            a.plot(ep[k - 1 :], np.convolve(y, np.ones(k) / k, "valid"), "-")
        a.set_title(title)
        a.set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    return png_path

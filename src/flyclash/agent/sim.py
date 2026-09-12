"""Offline simulator: synthetic bases with a hidden difficulty and a weakest edge.

Used for the conditioning test and the real-vs-random wiring ablation. Outcome model:
stars depend on (1 - difficulty), a bonus for dropping on the weakest edge, and noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyclash.agent.reward import ARMY_COST
from flyclash.brain.circuit import Circuit
from flyclash.brain.learn import SKIP_REWARD, Plastic, Policy
from flyclash.game.base_reader import GROUPS, N_GLOM, N_SECTORS, weakest_edge


@dataclass
class SynthBase:
    vector: np.ndarray
    difficulty: float
    weakest: str


def make_base(rng: np.random.Generator, difficulty: float | None = None) -> SynthBase:
    d = rng.uniform(0, 1) if difficulty is None else difficulty
    v = np.zeros(N_GLOM, np.float32)
    counts = rng.poisson(1 + 6 * d, size=(N_SECTORS, len(GROUPS))).astype(np.float32)
    weak_sector = int(rng.integers(N_SECTORS))
    counts[weak_sector] *= 0.2
    v[:32] = np.tanh(counts.ravel() / 3)
    v[32:37] = rng.uniform(0.2, 0.8, 5)
    return SynthBase(v, d, weakest_edge(v))


def outcome(rng: np.random.Generator, base: SynthBase, edge: str) -> float:
    """Stars minus the army cost. Easy ~2 stars, medium ~1, hard 0 (net -0.5, worse than a skip)."""
    p = (
        (1 - base.difficulty) * 0.9
        + (0.2 if edge == base.weakest else 0.0)
        - 0.15
        + rng.normal(0, 0.05)
    )
    return float(np.clip(np.floor(3 * p), 0, 3)) - ARMY_COST


def run(circuit: Circuit, n_attacks: int = 200, seed: int = 0, lr: float = 0.3) -> dict:
    rng = np.random.default_rng(seed)
    policy = Policy(circuit, Plastic.fresh(circuit, lr=lr), rng=rng)
    rows = []  # (trial, difficulty, attacked, reward, weakest_hit)
    for i in range(n_attacks):
        base = make_base(rng)
        act, d = policy.decide(base.vector, innate_edge=base.weakest)
        if d.attack:
            r = outcome(rng, base, d.edge)
            policy.learn_attack(act, d, r)
            hit = d.edge == base.weakest
        else:
            r = SKIP_REWARD
            policy.learn_skip(act)
            hit = np.nan
        rows.append((i, base.difficulty, d.attack, r, hit))
    a = np.array(rows, dtype=float)
    q = max(n_attacks // 4, 1)
    first, last = a[:q], a[-q:]

    def rate(block, lo, hi, attacked):
        m = (block[:, 1] >= lo) & (block[:, 1] < hi)
        return float(np.mean(block[m, 2] == attacked)) if m.any() else float("nan")

    return {
        "reward_first_quarter": float(first[:, 3].mean()),
        "reward_last_quarter": float(last[:, 3].mean()),
        "attack_rate_easy_first": rate(first, 0.0, 0.35, 1),
        "attack_rate_easy_last": rate(last, 0.0, 0.35, 1),
        "skip_rate_hard_first": rate(first, 0.65, 1.01, 0),
        "skip_rate_hard_last": rate(last, 0.65, 1.01, 0),
        "weakest_edge_hit_rate": float(np.nanmean(a[:, 4]))
        if np.isfinite(a[:, 4]).any()
        else float("nan"),
        "rewards": a[:, 3].tolist(),
    }


def ablation(circuit: Circuit, n_attacks: int = 200, seeds: int = 5) -> dict:
    real = [run(circuit, n_attacks, s)["reward_last_quarter"] for s in range(seeds)]
    rand = [
        run(circuit.with_random_pn_kc(s), n_attacks, s)["reward_last_quarter"]
        for s in range(seeds)
    ]
    return {
        "real_mean": float(np.mean(real)),
        "real_sd": float(np.std(real)),
        "random_mean": float(np.mean(rand)),
        "random_sd": float(np.std(rand)),
        "real": real,
        "random": rand,
    }

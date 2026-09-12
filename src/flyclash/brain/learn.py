"""Three-factor plasticity at KC->MBON synapses

Rule (Hige 2015, Springer & Nawrot 2021): dopamine paired with KC activity depresses the KC->MBON synapses in the dopamine-receiving compartment. Punishment (PPL1) depresses inputs to approach-driving MBONs, reward (PAM) depresses inputs to avoidance-driving MBONs. Only active KCs change (factor 1), only the reinforced compartment (factor 2), scaled by the reward prediction error (factor 3). Weights recover slowly toward 1. No backprop anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flyclash.brain.circuit import Circuit

EDGES = ("TL", "BL", "BR", "TR")
PATTERNS = ("line", "pincer", "point")
TROOP_ORDERS = ("barch", "archers_first", "all")
EDGE_PATTERN_ACTIONS = [(e, p) for e in EDGES for p in PATTERNS]  # 12
SKIP_REWARD = -0.1  # the gold a Next press costs, in star units


@dataclass
class Plastic:
    w: np.ndarray  # (nKC, nMBON) multiplicative weights, start at 1
    baseline: float = 0.0  # running mean attack reward, the "prediction" in RPE
    lr: float = 0.3
    recovery: float = 0.01  # per update, fraction of the way back toward 1
    w_min: float = 0.05
    w_max: float = 3.0
    epsilon: float = 0.1  # exploration rate for the action heads
    replay: list[dict] = field(default_factory=list)

    @classmethod
    def fresh(cls, circuit: Circuit, **kw) -> "Plastic":
        return cls(np.ones((circuit.n_kc, circuit.n_mbon), dtype=np.float32), **kw)

    def depress(self, kc_idx: np.ndarray, mbon_idx: np.ndarray, amount: float) -> None:
        """Dopamine in one compartment: shrink active-KC synapses onto those MBONs."""
        block = self.w[np.ix_(kc_idx, mbon_idx)]
        self.w[np.ix_(kc_idx, mbon_idx)] = np.clip(
            block * (1.0 - self.lr * amount), self.w_min, self.w_max
        )

    def credit(self, kc_idx: np.ndarray, mbon_idx: np.ndarray, delta: float) -> None:
        """Action heads: signed multiplicative update on the chosen action's MBONs."""
        block = self.w[np.ix_(kc_idx, mbon_idx)]
        self.w[np.ix_(kc_idx, mbon_idx)] = np.clip(
            block * (1.0 + self.lr * delta), self.w_min, self.w_max
        )

    def recover(self) -> None:
        self.w += self.recovery * (1.0 - self.w)

    def record(self, kc_idx: np.ndarray, kind: str, reward: float) -> None:
        self.replay.append(
            {"kc": kc_idx.tolist(), "kind": kind, "reward": float(reward)}
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, w=self.w, baseline=self.baseline)
        path.with_suffix(".replay.json").write_text(json.dumps(self.replay[-500:]))

    @classmethod
    def load(cls, path: Path, circuit: Circuit) -> "Plastic":
        if not path.exists():
            return cls.fresh(circuit)
        z = np.load(path)
        p = cls(z["w"], float(z["baseline"]))
        rp = path.with_suffix(".replay.json")
        if rp.exists():
            p.replay = json.loads(rp.read_text())
        return p


@dataclass
class Trace:
    """Eligibility trace for delayed reinforcement: remembers which KCs drove which action
    MBON and when, so dopamine arriving up to `window` seconds later still finds the right
    synapses (KC-DAN coincidence window, Aso & Rubin 2016)."""

    # tight window: punishment should land on the fatal decision (the last step or two),
    # not on the whole approach. A wide window over mostly-idle traces depresses "none"
    # brain-wide and makes the policy jittery (measured: sim survival 113 -> 76).
    window: float = 0.6
    tau: float = 0.25
    entries: list[tuple[float, np.ndarray, int]] = field(
        default_factory=list
    )  # (t, kc_idx, mbon)

    def add(self, t: float, kc_idx: np.ndarray, mbon: int) -> None:
        self.entries.append((t, kc_idx, mbon))
        self.entries = [e for e in self.entries if t - e[0] <= self.window]

    def punish(self, plastic: Plastic, t: float, amount: float = 1.0) -> float:
        """PPL1-style: depress the synapses that produced the last actions, most recent hardest."""
        total = 0.0
        for t0, kc, mbon in self.entries:
            a = amount * float(np.exp(-(t - t0) / self.tau))
            plastic.depress(kc, np.array([mbon]), a)
            total += a
        self.entries.clear()
        return total

    def reward(
        self, plastic: Plastic, t: float, amount: float, window: float = 0.5
    ) -> float:
        """PAM-style: strengthen the action synapses used in the last `window` seconds."""
        total = 0.0
        for t0, kc, mbon in self.entries:
            if t - t0 <= window:
                plastic.credit(kc, np.array([mbon]), amount)
                total += amount
        return total


@dataclass
class Decision:
    attack: bool
    edge: str
    pattern: str
    troop_order: str
    score: float
    edge_action: int
    order_action: int


class Policy:
    """Reads the circuit's MBONs as attack/skip plus two action heads."""

    def __init__(
        self,
        circuit: Circuit,
        plastic: Plastic,
        rng: np.random.Generator | None = None,
        attack_bias: float = 0.1,
        innate_weight: float = 0.3,
    ):
        self.c, self.p, self.rng = circuit, plastic, rng or np.random.default_rng()
        self.attack_bias, self.innate_weight = attack_bias, innate_weight
        self.approach = np.flatnonzero(circuit.mbon_valence > 0)
        self.avoid = np.flatnonzero(circuit.mbon_valence < 0)

    def decide(self, x: np.ndarray, innate_edge: str | None = None):
        act = self.c.forward(x, self.p.w)
        attack = act.decision_score + self.attack_bias > 0
        heads = act.mbon[self.c.action_mbons]
        edge_scores, order_scores = heads[:12].copy(), heads[12:15]
        if innate_edge is not None:
            # innate prior (lateral-horn-like): a fixed bonus on the weakest edge that learning can override
            e = EDGES.index(innate_edge)
            edge_scores[e * len(PATTERNS) : (e + 1) * len(PATTERNS)] += (
                self.innate_weight * edge_scores.mean()
            )
        if self.rng.random() < self.p.epsilon:
            ea = int(self.rng.integers(12))
        else:
            ea = int(np.argmax(edge_scores))
        oa = (
            int(self.rng.integers(3))
            if self.rng.random() < self.p.epsilon
            else int(np.argmax(order_scores))
        )
        edge, pattern = EDGE_PATTERN_ACTIONS[ea]
        return act, Decision(
            bool(attack), edge, pattern, TROOP_ORDERS[oa], act.decision_score, ea, oa
        )

    # note: SurfPolicy (subway) records every action in its Trace, including "none";
    # otherwise crashes while idle never punish idleness and the policy collapses to it.

    def learn_attack(self, act, d: Decision, reward: float) -> float:
        """Battle outcome as dopamine: RPE < 0 is PPL1 (depress approach), RPE > 0 is PAM (depress avoid)."""
        delta = reward - self.p.baseline
        if delta < 0:
            self.p.depress(act.kc_idx, self.approach, -delta)
        else:
            self.p.depress(act.kc_idx, self.avoid, delta)
        chosen = self.c.action_mbons[[d.edge_action, 12 + d.order_action]]
        self.p.credit(act.kc_idx, chosen, delta)
        self.p.baseline += 0.1 * (reward - self.p.baseline)
        self.p.recover()
        self.p.record(act.kc_idx, "attack", reward)
        return float(delta)

    def learn_skip(self, act) -> float:
        """No battle, no prediction error. A small fixed depression of avoid synapses keeps the
        fly from skipping the same base forever (curiosity), paid for by the Next gold."""
        self.p.depress(act.kc_idx, self.avoid, -SKIP_REWARD)
        self.p.record(act.kc_idx, "skip", SKIP_REWARD)
        return SKIP_REWARD

"""The offline simulator learns to skip hard bases while still attacking easy ones."""

from flyclash.agent import sim
from flyclash.brain.circuit import Circuit


def test_learns_to_skip_hard_and_keeps_attacking_easy():
    r = sim.run(Circuit.random(), n_attacks=400, seed=0)
    assert r["skip_rate_hard_last"] > r["skip_rate_hard_first"] + 0.15
    assert r["attack_rate_easy_last"] > 0.8


def test_ablation_runs():
    out = sim.ablation(Circuit.random(), n_attacks=60, seeds=2)
    assert len(out["real"]) == 2 and len(out["random"]) == 2

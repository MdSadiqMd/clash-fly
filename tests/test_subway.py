"""Subway Surfers: lane geometry, eligibility trace, and that the fly learns to dodge in the sim."""

import numpy as np

from flyclash.brain.circuit import Circuit
from flyclash.brain.learn import Plastic, Trace
from flyclash.game.base_reader import Detection
from flyclash.subway import sim
from flyclash.subway.agent import SurfPolicy, run_episodes
from flyclash.subway.reader import ACTIONS, Geometry, cell, innate_prior, vectorize


def test_geometry_maps_near_left_to_lane0_depth0():
    g = Geometry()
    # x=300 sits in the left third of the track at near depth (track spans ~x 225-775 there)
    v, lane = vectorize(
        [Detection("train", 300, 780, 0.9), Detection("player", 500, 950)],
        (1000, 1000),
        g,
        1,
        50.0,
        False,
    )
    assert v[cell(0, 0, 0)] == 0.9
    assert lane == 1 and v[48 + 1] == 1.0
    v2, _ = vectorize(
        [Detection("low_barrier", 500, 400, 0.8)], (1000, 1000), g, 1, 0.0, False
    )
    assert v2[cell(1, 4, 1)] == 0.8  # centre lane, far bin, jump kind


def test_vectorize_discards_roadside_scenery():
    g = Geometry()
    # striped market umbrella far right of the track must not become a lane-2 barrier
    v, _ = vectorize(
        [Detection("low_barrier", 900, 700, 0.9)], (1000, 1000), g, 1, 0.0, False
    )
    assert v[:45].sum() == 0


def test_innate_prior_reflexes():
    v = np.zeros(53, np.float32)
    v[48 + 1] = 1
    v[cell(1, 0, 1)] = 1.0
    assert ACTIONS[int(np.argmax(innate_prior(v, 1)))] == "jump"
    v[:] = 0
    v[48 + 1] = 1
    v[cell(1, 0, 0)] = 1.0
    v[cell(0, 1, 0)] = 1.0  # train ahead, left lane also blocked
    assert ACTIONS[int(np.argmax(innate_prior(v, 1)))] == "right"


def test_trace_punishes_only_recent_synapses():
    c = Circuit.random()
    p = Plastic.fresh(c)
    tr = Trace(window=1.5)
    old_kc, new_kc = np.array([1, 2, 3]), np.array([10, 11])
    tr.add(0.0, old_kc, 5)
    tr.add(
        2.0, new_kc, 7
    )  # the old entry (2.0 s ago) is outside the window and dropped
    tr.punish(p, 2.1)
    assert np.all(p.w[old_kc, 5] == 1.0)
    assert np.all(p.w[new_kc, 7] < 1.0)
    assert np.all(p.w[new_kc, 5] == 1.0)


def test_sim_learning_beats_no_learning(tmp_path):
    """The three-factor rule converges within the first episodes (0.6 s trace = fast credit
    assignment), so first-vs-last curves wash out. The meaningful invariant is the ablation:
    learning ON beats learning OFF, pooled over seeds, with the reflex prior turned down so
    the rule has headroom (at full prior the reflex alone sits near the sim's ceiling).
    Measured pooled survival: ON ~92 steps vs OFF ~58."""

    class Clock:
        t = 0.0

        def __call__(self):
            self.t += 0.1
            return self.t

    means = {}
    for learning in (True, False):
        per_seed = []
        for seed in range(4):
            c = Circuit.random(seed=1)
            # no epsilon exploration: a random lane change into a train is death
            policy = SurfPolicy(
                c,
                Plastic.fresh(c, epsilon=0.0, lr=0.5),
                rng=np.random.default_rng(seed),
                innate_weight=0.2,
                learning=learning,
            )
            logs = run_episodes(
                sim.SimEnv(seed=seed),
                policy,
                50,
                log_dir=tmp_path,
                tag=f"t{learning}{seed}",
                clock=Clock(),
            )
            per_seed.append(np.mean([l.steps for l in logs]))
            assert all(np.isfinite(l.loss) for l in logs)
        means[learning] = float(np.mean(per_seed))
    assert means[True] > 1.15 * means[False], means

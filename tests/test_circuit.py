"""Circuit sanity: sparseness, decorrelation, and that the three-factor rule learns."""

import numpy as np
import pytest

from flyclash.agent import sim
from flyclash.brain.circuit import ANNOTATIONS, CONNECTIVITY, Circuit
from flyclash.brain.learn import Plastic, Policy
from flyclash.config import DATA

HAVE_DATA = ANNOTATIONS.exists() and CONNECTIVITY.exists()


@pytest.fixture(scope="module")
def circuit():
    return (
        Circuit.from_flywire(cache=DATA / "circuit.npz")
        if HAVE_DATA
        else Circuit.random()
    )


def test_kc_sparseness(circuit):
    rng = np.random.default_rng(0)
    for _ in range(5):
        act = circuit.forward(rng.uniform(0, 1, 53))
        frac = len(act.kc_idx) / circuit.n_kc
        assert 0.03 <= frac <= 0.10, frac


def test_kc_codes_less_correlated_than_pn(circuit):
    rng = np.random.default_rng(1)
    a, b = rng.uniform(0, 1, 53), rng.uniform(0, 1, 53)
    b = 0.7 * a + 0.3 * b
    xa, xb = circuit.forward(a), circuit.forward(b)
    pn_corr = np.corrcoef(xa.pn, xb.pn)[0, 1]
    ka, kb = np.zeros(circuit.n_kc), np.zeros(circuit.n_kc)
    ka[xa.kc_idx] = 1
    kb[xb.kc_idx] = 1
    kc_corr = np.corrcoef(ka, kb)[0, 1]
    assert kc_corr < pn_corr


def test_conditioning_learns_attack_vs_skip(circuit):
    rng = np.random.default_rng(2)
    policy = Policy(circuit, Plastic.fresh(circuit, lr=0.5, epsilon=0.0), rng=rng)
    easy, hard = sim.make_base(rng, 0.1), sim.make_base(rng, 0.9)
    for _ in range(30):
        act, d = policy.decide(easy.vector)
        policy.learn_attack(act, d, 3.0)
        act, d = policy.decide(hard.vector)
        policy.learn_attack(act, d, 0.0)
    assert policy.decide(easy.vector)[1].score > policy.decide(hard.vector)[1].score


@pytest.mark.skipif(not HAVE_DATA, reason="FlyWire data not downloaded")
def test_real_circuit_shape(circuit):
    assert len(circuit.glomeruli) == 53
    assert circuit.n_kc > 4000
    assert (circuit.mbon_valence != 0).sum() >= 20
    claws = np.diff(circuit.W_pn_kc.tocsc().indptr)
    assert 4 <= np.median(claws[claws > 0]) <= 7

"""The fly olfactory circuit as a rate model, wired from the FlyWire v783 connectome.

ORN glomeruli (53) -> uniglomerular PNs -> Kenyon cells -> APL winner-take-all -> MBON types.
Only the KC->MBON synapses are plastic (see learn.py); everything else is the real wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyclash.config import DATA

ANNOTATIONS = DATA / "raw" / "flywire" / "neuron_annotations_783.tsv"
CONNECTIVITY = DATA / "raw" / "shiu" / "Connectivity_783.parquet"

# Aso et al. 2014 (eLife 3:e04580): activating glutamatergic MBONs drives avoidance,
# activating GABAergic / cholinergic MBONs drives approach. Used as the valence sign.
NT_VALENCE = {"glutamate": -1.0, "gaba": 1.0, "acetylcholine": 1.0}

N_ACTION_MBONS = 15  # 12 edge x pattern actions + 3 troop-order actions


@dataclass
class Activity:
    orn: np.ndarray
    pn: np.ndarray
    kc_idx: np.ndarray  # indices of active KCs
    mbon: np.ndarray  # per MBON type
    decision_score: float


@dataclass
class Circuit:
    glomeruli: list[str]
    W_glom_pn: np.ndarray  # (53, nPN)
    W_pn_kc: sp.csr_matrix  # (nPN, nKC)
    W_kc_mbon: sp.csr_matrix  # (nKC, nMBON types)
    mbon_types: list[str]
    mbon_valence: np.ndarray  # +1 approach, -1 avoid, 0 unknown
    action_mbons: np.ndarray  # (15,) MBON type indices used as actions
    k_frac: float = 0.07  # APL keeps this fraction of KCs active (Turner 2008: 5-10%)
    sigma: float = 0.1  # divisive normalisation floor (Olsen 2010)

    @property
    def n_kc(self) -> int:
        return self.W_pn_kc.shape[1]

    @property
    def n_mbon(self) -> int:
        return self.W_kc_mbon.shape[1]

    def forward(self, x: np.ndarray, w_plastic: np.ndarray | None = None) -> Activity:
        orn = np.clip(np.asarray(x, dtype=np.float32), 0, None)
        pn_in = self.W_glom_pn.T @ orn
        pn = pn_in / (self.sigma + pn_in.mean())
        kc_in = np.asarray(self.W_pn_kc.T @ pn).ravel()
        k = max(1, int(self.k_frac * self.n_kc))
        cand = np.argpartition(-kc_in, k)[:k]
        kc_idx = cand[kc_in[cand] > 0]
        W = self.W_kc_mbon[kc_idx]
        if w_plastic is not None:
            W = W.multiply(w_plastic[kc_idx])
        mbon = np.asarray(W.sum(0)).ravel() / max(len(kc_idx), 1)
        # group means (28 approach vs 7 avoid types in FlyWire) so the untrained score sits near 0;
        # normalised to [-1, 1] so attack_bias is in comparable units
        approach = mbon[self.mbon_valence > 0].mean()
        avoid = mbon[self.mbon_valence < 0].mean()
        score = (approach - avoid) / (approach + avoid + 1e-9)
        return Activity(orn, pn, kc_idx, mbon, float(score))

    def with_random_pn_kc(self, seed: int = 0) -> "Circuit":
        """Degree-matched control: each KC keeps its number of PN inputs, partners drawn
        with probability proportional to PN out-degree. Weight of each claw kept."""
        rng = np.random.default_rng(seed)
        W = self.W_pn_kc.tocsc()
        out_deg = np.asarray(W.sum(1)).ravel()
        p = out_deg / out_deg.sum()
        rows, cols, vals = [], [], []
        for kc in range(W.shape[1]):
            col = W.getcol(kc)
            n = col.nnz
            if n == 0:
                continue
            pns = rng.choice(W.shape[0], size=n, replace=False, p=p)
            rows += list(pns)
            cols += [kc] * n
            vals += list(col.data)
        Wr = sp.csr_matrix((vals, (rows, cols)), shape=W.shape)
        return Circuit(
            self.glomeruli,
            self.W_glom_pn,
            Wr,
            self.W_kc_mbon,
            self.mbon_types,
            self.mbon_valence,
            self.action_mbons,
            self.k_frac,
            self.sigma,
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            glomeruli=np.array(self.glomeruli),
            W_glom_pn=self.W_glom_pn,
            pn_kc_data=self.W_pn_kc.data,
            pn_kc_indices=self.W_pn_kc.indices,
            pn_kc_indptr=self.W_pn_kc.indptr,
            pn_kc_shape=self.W_pn_kc.shape,
            kc_mbon_data=self.W_kc_mbon.data,
            kc_mbon_indices=self.W_kc_mbon.indices,
            kc_mbon_indptr=self.W_kc_mbon.indptr,
            kc_mbon_shape=self.W_kc_mbon.shape,
            mbon_types=np.array(self.mbon_types),
            mbon_valence=self.mbon_valence,
            action_mbons=self.action_mbons,
        )

    @classmethod
    def load(cls, path: Path) -> "Circuit":
        z = np.load(path, allow_pickle=False)
        return cls(
            list(z["glomeruli"]),
            z["W_glom_pn"],
            sp.csr_matrix(
                (z["pn_kc_data"], z["pn_kc_indices"], z["pn_kc_indptr"]),
                shape=tuple(z["pn_kc_shape"]),
            ),
            sp.csr_matrix(
                (z["kc_mbon_data"], z["kc_mbon_indices"], z["kc_mbon_indptr"]),
                shape=tuple(z["kc_mbon_shape"]),
            ),
            list(z["mbon_types"]),
            z["mbon_valence"],
            z["action_mbons"],
        )

    @classmethod
    def from_flywire(
        cls,
        annotations: Path = ANNOTATIONS,
        connectivity: Path = CONNECTIVITY,
        cache: Path | None = DATA / "circuit.npz",
    ) -> "Circuit":
        if cache is not None and cache.exists():
            return cls.load(cache)
        a = pd.read_csv(annotations, sep="\t", low_memory=False)
        e = pd.read_parquet(
            connectivity, columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"]
        )
        orn = a[a.cell_type.astype(str).str.startswith("ORN_")]
        glomeruli = sorted(orn.cell_type.unique())
        upn = a[(a.cell_class == "ALPN") & (a.cell_sub_class == "uniglomerular")]
        kc = a[a.cell_class == "Kenyon_Cell"]
        mbon = a[a.cell_class == "MBON"]
        mbon_types = sorted(mbon.cell_type.astype(str).unique())

        glom_of = dict(zip(orn.root_id, orn.cell_type))
        pn_col = {rid: i for i, rid in enumerate(upn.root_id)}
        kc_col = {rid: i for i, rid in enumerate(kc.root_id)}
        mbon_col = {
            rid: mbon_types.index(t)
            for rid, t in zip(mbon.root_id, mbon.cell_type.astype(str))
        }

        def block(pre_map, post_map, shape):
            m = e[
                e.Presynaptic_ID.isin(pre_map.keys())
                & e.Postsynaptic_ID.isin(post_map.keys())
            ]
            r = m.Presynaptic_ID.map(pre_map).to_numpy()
            c = m.Postsynaptic_ID.map(post_map).to_numpy()
            return sp.csr_matrix(
                (m.Connectivity.to_numpy(dtype=np.float32), (r, c)), shape=shape
            )

        glom_idx = {g: i for i, g in enumerate(glomeruli)}
        W_glom_pn = block(
            {rid: glom_idx[g] for rid, g in glom_of.items()},
            pn_col,
            (len(glomeruli), len(upn)),
        ).toarray()
        W_glom_pn /= np.maximum(
            W_glom_pn.sum(0, keepdims=True), 1e-6
        )  # each PN's input sums to 1
        W_pn_kc = block(pn_col, kc_col, (len(upn), len(kc)))
        W_pn_kc = _col_normalise(W_pn_kc)
        W_kc_mbon = _col_normalise(block(kc_col, mbon_col, (len(kc), len(mbon_types))))

        nt = (
            mbon.assign(t=mbon.cell_type.astype(str))
            .groupby("t")
            .top_nt.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
        )
        valence = np.array(
            [NT_VALENCE.get(str(nt.get(t, "")), 0.0) for t in mbon_types],
            dtype=np.float32,
        )
        kc_input = np.asarray(W_kc_mbon.sum(0)).ravel()
        action_mbons = np.argsort(-kc_input)[:N_ACTION_MBONS]
        c = cls(
            glomeruli,
            W_glom_pn.astype(np.float32),
            W_pn_kc,
            W_kc_mbon,
            mbon_types,
            valence,
            action_mbons,
        )
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            c.save(cache)
        return c

    @classmethod
    def random(
        cls,
        seed: int = 0,
        n_glom: int = 53,
        n_pn: int = 150,
        n_kc: int = 2000,
        n_mbon: int = 35,
        claws: int = 6,
    ) -> "Circuit":
        """Synthetic stand-in with fly-like statistics, for tests without the dataset."""
        rng = np.random.default_rng(seed)
        W_glom_pn = np.zeros((n_glom, n_pn), np.float32)
        W_glom_pn[rng.integers(0, n_glom, n_pn), np.arange(n_pn)] = 1.0
        rows = rng.integers(0, n_pn, (n_kc, claws))
        W_pn_kc = _col_normalise(
            sp.csr_matrix(
                (
                    rng.uniform(0.5, 1.5, rows.size),
                    (rows.ravel(), np.repeat(np.arange(n_kc), claws)),
                ),
                shape=(n_pn, n_kc),
            )
        )
        W_kc_mbon = _col_normalise(
            sp.csr_matrix(
                rng.uniform(0, 1, (n_kc, n_mbon))
                * (rng.uniform(0, 1, (n_kc, n_mbon)) < 0.3)
            )
        )
        valence = np.where(np.arange(n_mbon) % 3 == 0, -1.0, 1.0).astype(np.float32)
        return cls(
            [f"G{i}" for i in range(n_glom)],
            W_glom_pn,
            W_pn_kc,
            W_kc_mbon,
            [f"MBON{i:02d}" for i in range(n_mbon)],
            valence,
            np.arange(N_ACTION_MBONS),
        )


def _col_normalise(W: sp.spmatrix) -> sp.csr_matrix:
    W = sp.csr_matrix(W, dtype=np.float32)
    s = np.asarray(W.sum(0)).ravel()
    s[s == 0] = 1.0
    return sp.csr_matrix(W @ sp.diags(1.0 / s))

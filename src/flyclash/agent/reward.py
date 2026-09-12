"""Reward from a battle result: stars plus fraction of the offered loot taken, minus the army."""

from __future__ import annotations

from flyclash.game.actions import Result
from flyclash.game.base_reader import Loot

ARMY_COST = 0.5  # every attack spends the trained army; a Next press costs only gold (SKIP_REWARD)


def compute_reward(result: Result, loot: Loot) -> float:
    offered = loot.gold + loot.elixir
    taken = result.gold + result.elixir
    frac = min(taken / offered, 1.0) if offered > 0 else 0.0
    return float(result.stars + frac - ARMY_COST)

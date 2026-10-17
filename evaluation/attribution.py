"""Per-agent attribution (spec §6.7).

- Leave-one-out: re-run protocol with each specialist removed in turn;
  attribute Sharpe gap to that agent.
- Monte Carlo permutation Shapley: average marginal Sharpe contribution of
  each agent over random orderings (v(empty set) taken as 0, i.e. flat).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd

from evaluation.metrics import sharpe

AGENT_NAMES = (
    "macro_regime",
    "narrative_event",
    "cross_asset_transmission",
    "technical_trend",
    "fundamentals_carry",
    "risk_correlation",
)


def leave_one_out(
    run_with_agents: Callable[[tuple[str, ...]], pd.Series],
    full_set: tuple[str, ...] = AGENT_NAMES,
) -> dict[str, float]:
    """For each agent, sharpe(full) - sharpe(full \\ {agent})."""
    base = sharpe(run_with_agents(full_set))
    out: dict[str, float] = {}
    for a in full_set:
        subset = tuple(x for x in full_set if x != a)
        out[a] = float(base - sharpe(run_with_agents(subset)))
    return out


def dag_shapley_approx(
    run_with_agents: Callable[[tuple[str, ...]], pd.Series],
    full_set: tuple[str, ...] = AGENT_NAMES,
    n_permutations: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    """Truncated Monte Carlo Shapley.

    For each random permutation of agents, walk the prefix and assign each
    agent its marginal Sharpe contribution. Average across permutations.
    """
    rng = random.Random(seed)
    contrib = {a: 0.0 for a in full_set}
    counts = {a: 0 for a in full_set}
    for _ in range(n_permutations):
        order = list(full_set)
        rng.shuffle(order)
        prev_sr = 0.0
        for k in range(1, len(order) + 1):
            subset = tuple(order[:k])
            sr = sharpe(run_with_agents(subset))
            marginal = sr - prev_sr
            agent_added = order[k - 1]
            contrib[agent_added] += marginal
            counts[agent_added] += 1
            prev_sr = sr
    return {a: contrib[a] / max(counts[a], 1) for a in full_set}

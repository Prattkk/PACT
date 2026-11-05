"""Controlled communication x adjudication 2x2 (pre-registration v3, section 1): contrasts and mechanisms.

Contrasts (Sharpe-ratio differences; one deterministic run per regime, so the bootstrap resamples
21-day circular blocks of days, common to all four cells; sensitivity 5 and 63):
  principal D-B (communication, judge fixed) and B-A (judge vs vote, isolated revision);
  supporting C-A (communication under the vote) and the interaction (D-B)-(C-A).
Mechanisms (offline replay from the response cache, checked against the saved cells): how often
revision changes a specialist's direction in each condition, how often the two conditions'
revised directions differ, and how often final directions differ between cells; plus a check
that A/B and C/D used identical revised views. Writes results/tables/controlled_<regime>.json.

  python -m scripts.analyze_controlled --regime clean
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from data.universe import TRADEABLE_SYMBOLS
from evaluation.statistical_tests import _circular_block_indices
from portfolio.construction import PortfolioConfig
from scripts._common import RAW_DIR, TABLES_DIR, build_agents, build_llm_client, build_protocol, load_config
from scripts.analyze_decision_flow import window

CELLS = {"A": "ctrl_isolated_vote", "B": "ctrl_isolated_judge", "C": "ctrl_shared_vote", "D": "ctrl_shared_judge"}
CONTRASTS = {"D-B": (("D", 1), ("B", -1)), "B-A": (("B", 1), ("A", -1)), "C-A": (("C", 1), ("A", -1)),
             "interaction": (("D", 1), ("B", -1), ("C", -1), ("A", 1))}
PRINCIPAL = ("D-B", "B-A")
MARGIN, ANN = 0.20, np.sqrt(252)


def _sr(x: np.ndarray) -> np.ndarray:
    return x.mean(-1) / x.std(-1, ddof=1) * ANN


def _cls(lo95, hi95, lo90, hi90):
    if lo95 > 0:
        return "I"
    if lo90 > -MARGIN and hi90 < MARGIN:
        return "E"
    if hi95 < MARGIN:
        return "X"
    return "?"


def contrasts(regime: str, w: str, n_boot: int = 2000) -> dict:
    ret = {k: pd.read_parquet(RAW_DIR / f"{w}__{regime}__{c}" / "excess_returns.parquet").iloc[:, 0]
           for k, c in CELLS.items()}
    df = pd.DataFrame(ret).dropna()
    X = df[list(CELLS)].to_numpy().T            # (4, T)
    n = X.shape[1]
    sr = dict(zip(CELLS, _sr(X)))
    out = {"sharpe": {k: float(v) for k, v in sr.items()}, "T": n, "n_boot": n_boot}
    for block in (5, 21, 63):
        rng = np.random.default_rng(20261105)
        draws = np.array([_sr(X[:, _circular_block_indices(n, block, rng)]) for _ in range(n_boot)])  # (K, 4)
        res = {}
        for name, terms in CONTRASTS.items():
            obs = sum(s * sr[c] for c, s in terms)
            d = sum(s * draws[:, list(CELLS).index(c)] for c, s in terms)
            lo95, hi95, lo90, hi90, lo975, hi975 = np.quantile(d, [0.025, 0.975, 0.05, 0.95, 0.0125, 0.9875])
            p = float(np.mean(np.abs(d - d.mean()) >= abs(obs)))
            res[name] = {"diff": float(obs), "lo": float(lo95), "hi": float(hi95), "lo90": float(lo90),
                         "hi90": float(hi90), "p": p, "cls": _cls(lo95, hi95, lo90, hi90)}
            if name in PRINCIPAL:
                res[name].update({"simlo": float(lo975), "simhi": float(hi975)})
        out[f"block{block}"] = res
    return out


def replay(regime: str, key: str, w: str) -> dict:
    cfg = load_config("ablation_matrix.yaml")
    start, end = window(cfg, w)
    llm = build_llm_client(regime, offline=True)
    proto = build_protocol(key, build_agents(llm, TRADEABLE_SYMBOLS, cell_window=(start, end), enable_altdata=True),
                           llm, TRADEABLE_SYMBOLS)
    rec = {"first": {}, "revised": {}, "final": {}}
    rev0, run0 = proto._revise, proto.run

    def rev(as_of, first):
        out = rev0(as_of, first)
        rec["first"][as_of] = {n: {v.instrument: v.direction for v in d.views} for n, d in first.items()}
        rec["revised"][as_of] = {n: {v.instrument: (v.direction, round(v.conviction, 6)) for v in d.views}
                                 for n, d in out.items()}
        return out

    def run(as_of):
        r = run0(as_of)
        rec["final"][as_of] = {v.instrument: v.direction for v in r.final_views}
        return r
    proto._revise, proto.run = rev, run
    res = run_backtest(proto, TRADEABLE_SYMBOLS, BacktestConfig(
        start=start, end=end, rebalance_freq=cfg["backtest"]["rebalance_freq"], portfolio=PortfolioConfig()))
    fails = sum(v for k, v in res.metadata.get("llm_stats", {}).items() if k.endswith(":call_failed"))
    if fails:
        raise RuntimeError(f"{key}: {fails} calls missing from the cache")
    saved = pd.read_parquet(RAW_DIR / f"{w}__{regime}__{key}" / "excess_returns.parquet").iloc[:, 0]
    r = res.excess_returns.iloc[:, 0] if isinstance(res.excess_returns, pd.DataFrame) else res.excess_returns
    rec["replay_max_abs_diff"] = float((r.reindex(saved.index) - saved).abs().max())
    rec["llm_stats"] = res.metadata.get("llm_stats", {})
    return rec


def mechanisms(regime: str, w: str) -> dict:
    R = {k: replay(regime, c, w) for k, c in CELLS.items()}
    dates = sorted(R["A"]["final"])
    revised_specs = ("macro_regime", "narrative_event", "fundamentals_carry", "cross_asset_transmission")

    def share(f):
        vals = [f(d, s) for d in dates for s in TRADEABLE_SYMBOLS]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def rev_changed(cell):
        return share(lambda d, s: np.mean([R[cell]["revised"][d][n][s][0] != R[cell]["first"][d][n][s]
                                           for n in revised_specs]))
    out = {
        "identical_revisions_A_B": all(R["A"]["revised"][d] == R["B"]["revised"][d] for d in dates),
        "identical_revisions_C_D": all(R["C"]["revised"][d] == R["D"]["revised"][d] for d in dates),
        "identical_first_pass": all(R["A"]["first"][d] == R[k]["first"][d] for d in dates for k in "BCD"),
        "revision_changes_direction_isolated": rev_changed("A"),
        "revision_changes_direction_shared": rev_changed("C"),
        "revised_direction_differs_shared_vs_isolated": share(
            lambda d, s: np.mean([R["C"]["revised"][d][n][s][0] != R["A"]["revised"][d][n][s][0] for n in revised_specs])),
        "final_direction_differs": {f"{x}-{y}": share(lambda d, s, x=x, y=y: R[x]["final"][d][s] != R[y]["final"][d][s])
                                    for x, y in (("C", "A"), ("B", "A"), ("D", "B"), ("D", "C"))},
        "replay_max_abs_diff": {k: R[k]["replay_max_abs_diff"] for k in R},
        "unparseable_or_failed": {k: {s: v for s, v in R[k]["llm_stats"].items()
                                      if s.endswith(("unparseable", "call_failed"))} for k in R},
        "n_decisions": len(dates),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="clean")
    ap.add_argument("--window", default="b")
    ap.add_argument("--no-replay", action="store_true")
    a = ap.parse_args()
    out = {"regime": a.regime, "window": a.window, "contrasts": contrasts(a.regime, a.window)}
    if not a.no_replay:
        out["mechanisms"] = mechanisms(a.regime, a.window)
    (TABLES_DIR / f"controlled_{a.regime}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out["contrasts"]["block21"].items()}, indent=1))
    if "mechanisms" in out:
        print(json.dumps(out["mechanisms"], indent=1, default=float))


if __name__ == "__main__":
    main()

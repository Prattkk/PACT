"""Pre-registered disagreement diagnostic (pre-registration v2, section 3), return side.

For each matched protocol and run: daily difference-return contributions
(w_protocol - w_ensemble)_{t-1} x r_t per instrument, split by the decision-time disagreement
class of the most recent decision (deterministic anchors of the macro, narrative and
fundamentals specialists; high = above the pooled median). Reads saved cells; one offline
replay of the no-LLM ensemble supplies the anchors. Writes results/tables/disagreement_<window>.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from backtest.engine import WARMUP_CALENDAR_DAYS
from data.fetchers.prices import fetch_panel
from data.universe import TRADEABLE_SYMBOLS
from scripts._common import RAW_DIR, TABLES_DIR, load_config
from scripts.analyze_decision_flow import GROUPS, MATCHED, SPECIALISTS, capture, window


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="b")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    a = ap.parse_args()
    cfg = load_config("ablation_matrix.yaml")
    start, end = window(cfg, a.window)
    base = capture("none", "independent_ensemble", a.window)
    dis = pd.DataFrame({t: 1 - pd.DataFrame({s: ag[s] for s in SPECIALISTS if s in ag}).mean(axis=1).abs()
                        for t, ag in base["agents"].items()}).T.sort_index()
    dis.index = pd.to_datetime(dis.index)
    med = float(np.nanmedian(dis.to_numpy()))
    high = dis > med
    px = fetch_panel(list(TRADEABLE_SYMBOLS), start - timedelta(days=WARMUP_CALENDAR_DAYS), end + timedelta(days=1),
                     field="Adj Close")
    rets = px.pct_change()
    out = {"median": med, "share_high_instrument_weeks": float(high.to_numpy().mean()), "groups": {}}
    for g in a.groups:
        regs = GROUPS[g] if a.window == "b" else (["clean"] if g == "clean" else [])
        res = {p: [] for p in MATCHED}
        for r in regs:
            we = pd.read_parquet(RAW_DIR / f"{a.window}__{r}__independent_ensemble" / "weights.parquet")
            for p in MATCHED:
                wp = pd.read_parquet(RAW_DIR / f"{a.window}__{r}__{p}" / "weights.parquet")
                dW = (wp - we).reindex(columns=we.columns).fillna(0.0)
                contrib = dW.shift(1) * rets.reindex(dW.index)[dW.columns]
                lab = high.reindex(dW.index, method="ffill").shift(1).reindex(columns=dW.columns)
                row = {}
                for name, m in (("high", lab == True), ("low", lab == False)):  # noqa: E712
                    mm = m.to_numpy()
                    c = contrib.to_numpy()
                    row[name] = {"instrument_days": int(mm.sum()),
                                 "mean_abs_weight_diff": float(np.abs(dW.shift(1).to_numpy()[mm]).mean()),
                                 "ann_contribution": float(np.nansum(c[mm]) / len(dW) * 252),
                                 "mean_abs_daily_contribution_bp": float(np.nanmean(np.abs(c[mm])) * 1e4)}
                res[p].append(row)
        out["groups"][g] = {p: {k: {m: float(np.mean([x[k][m] for x in v])) for m in v[0][k]} for k in ("high", "low")}
                            for p, v in res.items() if v}
    (TABLES_DIR / f"disagreement_{a.window}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()

"""Where the debate judge's difference return comes from (2022-2024, reported Claude runs).

Daily difference-return contributions (w_debate - w_ensemble)_{t-1} x r_t per instrument are split
into instrument-days on which the two held positions have the same sign (including both flat:
the judge resized a position) and days on which the signs differ (the judge flipped or opened /
closed it). Run with PACT_PX_DIR set to the 2022-2024 price snapshot. Writes
results/tables/judge_decomposition_b.json.
"""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from backtest.engine import WARMUP_CALENDAR_DAYS
from data.fetchers.prices import fetch_panel
from data.universe import TRADEABLE_SYMBOLS
from scripts._common import RAW_DIR, TABLES_DIR, load_config
from scripts.analyze_decision_flow import GROUPS, window


def main() -> None:
    cfg = load_config("ablation_matrix.yaml")
    start, end = window(cfg, "b")
    px = fetch_panel(list(TRADEABLE_SYMBOLS), start - timedelta(days=WARMUP_CALENDAR_DAYS), end + timedelta(days=1),
                     field="Adj Close")
    rets = px.pct_change()
    out = {}
    for g in ("sonnet", "haiku"):
        rows = []
        for r in GROUPS[g]:
            we = pd.read_parquet(RAW_DIR / f"b__{r}__independent_ensemble" / "weights.parquet")
            wd = pd.read_parquet(RAW_DIR / f"b__{r}__debate" / "weights.parquet").reindex(columns=we.columns).fillna(0.0)
            we, wd = we.align(wd, join="inner")
            c = ((wd - we).shift(1) * rets.reindex(we.index)[we.columns]).to_numpy()
            flip = (np.sign(wd.shift(1).round(6)) != np.sign(we.shift(1).round(6))).to_numpy()
            ann = lambda m: float(np.nansum(np.where(m, c, 0.0)) / len(we) * 252)  # noqa: E731
            tot, fl = ann(np.ones_like(flip, dtype=bool)), ann(flip)
            rows.append({"run": r, "ann_total": tot, "ann_sign_differs": fl, "ann_same_sign": tot - fl,
                         "share_instrument_days_sign_differs": float(flip[1:].mean())})
        m = {k: float(np.mean([x[k] for x in rows])) for k in rows[0] if k != "run"}
        m["share_of_total_from_same_sign"] = m["ann_same_sign"] / m["ann_total"] if m["ann_total"] else float("nan")
        m["runs_same_sign_positive"] = int(sum(x["ann_same_sign"] > 0 for x in rows))
        m["runs_sign_differs_positive"] = int(sum(x["ann_sign_differs"] > 0 for x in rows))
        m["n_runs"] = len(rows)
        out[g] = {"mean": m, "runs": rows}
        print(g, {k: round(v, 4) for k, v in m.items()})
    (TABLES_DIR / "judge_decomposition_b.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

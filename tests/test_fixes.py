"""Regression tests for the accounting / data / inference fixes."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

import data.fetchers.prices as P
from backtest.simulator import PortfolioSimulator, simulate_target_schedule
from evaluation.metrics import coordination_breakeven_spread, net_excess_at_cost, sharpe
from evaluation.statistical_tests import (
    bh_qvalues,
    ledoit_wolf_sharpe_test,
    paired_bootstrap_sharpe_diff,
)


def _frame(index, close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({c: close for c in P.PRICE_COLUMNS} | {"Volume": 1.0}, index=pd.DatetimeIndex(index))


def test_btc_splice_is_return_continuous(monkeypatch):
    btc_idx = pd.date_range("2024-01-08", "2024-01-15", freq="D")
    btc = _frame(btc_idx, [44000, 45000, 46000, 46500, 46000, 45500, 45000, 45200])
    ibit_idx = pd.bdate_range("2024-01-11", "2024-01-15")
    ibit = _frame(ibit_idx, [27.0, 26.7, 26.3])

    def fake(symbol, start, end):
        df = btc if symbol == "BTC-USD" else ibit
        return df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]

    monkeypatch.setattr(P, "_cached_yf_download", fake)
    out = P._fetch_btc_spliced(date(2024, 1, 8), date(2024, 1, 16))
    r = out["Adj Close"].pct_change()
    # splice day: level continuous (no -99.9% jump)
    assert abs(r.loc["2024-01-11"]) < 0.05
    # after the splice the series moves with IBIT
    assert np.isclose(r.loc["2024-01-12"], 26.7 / 27.0 - 1)


def test_calendar_alignment_drops_crypto_weekends():
    idx = pd.date_range("2024-03-01", "2024-03-11", freq="D")  # Fri..Mon
    etf = pd.Series([100.0 + i for i in range(len(idx))], index=idx)
    etf[idx.dayofweek >= 5] = np.nan
    btc = pd.Series(np.linspace(60000, 61000, len(idx)), index=idx)
    out = P.align_to_exchange_calendar(pd.DataFrame({"SPY": etf, "BTC": btc}))
    assert (out.index.dayofweek < 5).all()
    assert out.loc["2024-03-04", "BTC"] == btc.loc["2024-03-04"]  # Monday crypto close kept


def test_simulator_charges_half_round_trip_per_one_way_unit():
    sim = PortfolioSimulator(["A"], 1_000_000.0, cost_bps_roundtrip=30.0)
    sim.step(pd.Timestamp("2024-01-02"), None, 0.0, {"A": 1.0})
    res = sim.result()
    assert np.isclose(res.turnover.iloc[0], 1.0)
    assert np.isclose(res.equity.iloc[0], 1_000_000.0 * (1 - 0.0015))


def test_cash_only_portfolio_has_zero_excess_return():
    idx = pd.bdate_range("2024-01-01", periods=10)
    rets = pd.DataFrame({"A": 0.01}, index=idx)
    rf = pd.Series(0.0002, index=idx)
    res = simulate_target_schedule(rets, rf, {}, cost_bps_roundtrip=30.0)
    assert np.allclose(res.excess_returns.iloc[1:], 0.0)
    assert np.allclose(res.total_returns.iloc[1:], 0.0002)


def test_positions_drift_between_trades():
    idx = pd.bdate_range("2024-01-01", periods=3)
    rets = pd.DataFrame({"A": [0.0, 0.10, 0.0], "B": [0.0, 0.0, 0.0]}, index=idx)
    res = simulate_target_schedule(rets, pd.Series(0.0, index=idx), {idx[0]: {"A": 0.5, "B": 0.5}},
                                   cost_bps_roundtrip=0.0)
    assert np.isclose(res.weights.loc[idx[1], "A"], 0.55 / 1.05)


def test_ledoit_wolf_size_is_reasonable():
    rng = np.random.default_rng(3)
    rejections, zmax = 0, 0.0
    for _ in range(200):
        common = rng.normal(0, 0.008, 750)
        a = pd.Series(0.0003 + common + rng.normal(0, 0.006, 750))
        b = pd.Series(0.0003 + common + rng.normal(0, 0.006, 750))
        out = ledoit_wolf_sharpe_test(a, b)
        rejections += out["p_value"] < 0.05
        zmax = max(zmax, abs(out["z"]))
    assert 0.0 <= rejections / 200 <= 0.12   # nominal 5%
    assert zmax < 5.0                         # the pre-fix test produced |z| > 100


def test_ledoit_wolf_detects_large_difference():
    rng = np.random.default_rng(4)
    common = rng.normal(0, 0.01, 2000)
    a = pd.Series(0.0015 + common + rng.normal(0, 0.003, 2000))
    b = pd.Series(-0.0005 + common + rng.normal(0, 0.003, 2000))
    assert ledoit_wolf_sharpe_test(a, b)["p_value"] < 0.01


def test_paired_bootstrap_and_qvalues():
    rng = np.random.default_rng(5)
    a = pd.Series(rng.normal(0.0005, 0.01, 600))
    b = pd.Series(rng.normal(0.0005, 0.01, 600))
    out = paired_bootstrap_sharpe_diff(a, b, n_boot=300)
    assert out["ci_lo"] <= out["diff"] <= out["ci_hi"]
    q = bh_qvalues([0.001, 0.02, 0.03, 0.5])
    assert q[0] <= q[1] <= q[2] <= q[3] <= 1.0


def test_cbs_zero_when_already_worse():
    idx = pd.bdate_range("2023-01-02", periods=500)
    rng = np.random.default_rng(6)
    ens = pd.Series(rng.normal(0.0006, 0.01, 500), index=idx)
    coord = ens - 0.0004
    t = pd.Series(0.05, index=idx)
    assert coordination_breakeven_spread(coord, ens, t, t) == 0.0
    assert sharpe(net_excess_at_cost(ens, t, 30)) < sharpe(ens)


def test_alfred_point_in_time_uses_release_dates():
    import data.fetchers.alfred as A
    idx = pd.date_range("2022-01-01", periods=6, freq="MS")
    s = pd.Series(np.arange(6.0), index=idx, name="CPIAUCSL")
    s.attrs["release"] = pd.Series(idx + pd.Timedelta(days=42), index=idx)
    out = A._point_in_time(s, "CPIAUCSL", date(2022, 1, 1), date(2022, 3, 10))
    # Feb-2022 observation released 2022-03-15 must not be visible on 2022-03-10
    assert out.index.max() == pd.Timestamp("2022-01-01")
    # without stored release dates the conservative lag is applied
    s2 = pd.Series(np.arange(6.0), index=idx, name="GDP")
    assert A._point_in_time(s2, "GDP", date(2022, 1, 1), date(2022, 4, 1)).empty
    out2 = A._point_in_time(s2, "GDP", date(2022, 1, 1), date(2022, 5, 10))
    assert list(out2.index) == [pd.Timestamp("2022-01-01")]


def test_eia_and_noaa_respect_publication_lags():
    from agents.fundamentals_carry import FundamentalsCarryAgent
    ag = FundamentalsCarryAgent(None, ("USO",))
    weeks = pd.date_range("2023-01-06", periods=60, freq="W-FRI")
    vals = np.r_[np.zeros(59), 100.0]           # spike in the week ending 2024-02-23
    ag._crude_inv = pd.Series(vals, index=weeks)
    assert ag._uso_inventory_z(weeks[-1].date()) == 0.0 or abs(ag._uso_inventory_z(weeks[-1].date())) < 1e-9
    assert ag._uso_inventory_z((weeks[-1] + pd.Timedelta(days=6)).date()) > 1.0   # visible after release
    months = pd.date_range("2021-01-01", periods=30, freq="MS")
    ag._hdd = pd.Series(np.r_[np.zeros(29), 50.0], index=months)               # spike in the last month
    assert ag._hdd_z((months[-1] + pd.Timedelta(days=20)).date()) == 0.0        # month not over yet
    assert ag._hdd_z((months[-1] + pd.DateOffset(months=1) + pd.Timedelta(days=10)).date()) > 1.0


def test_json_object_parsing_tolerates_fences():
    from agents._llm_helpers import parse_json_object
    assert parse_json_object('```json\n{"direction": 1, "conviction": 0.4}\n```')["direction"] == 1
    assert parse_json_object("no json here") is None


def test_ensemble_uses_the_manager_vote():
    from agents.base_agent import AgentDecision, InstrumentView
    from agents.portfolio_manager import PortfolioManagerAgent
    from coordination.independent_ensemble import IndependentEnsembleProtocol

    class Stub:
        def __init__(self, name, d, c):
            self.name, self.d, self.c = name, d, c
        def run(self, as_of):
            return AgentDecision(agent_name=self.name, as_of=as_of,
                                 views=[InstrumentView(instrument="SPY", direction=self.d, conviction=self.c, horizon="1w")])
    agents = {"macro_regime": Stub("macro_regime", 1, 0.4), "narrative_event": Stub("narrative_event", -1, 0.3),
              "technical_trend": Stub("technical_trend", 1, 0.9), "risk_correlation": Stub("risk_correlation", -1, 1.0)}
    ens = IndependentEnsembleProtocol(agents, ("SPY",)).run(date(2024, 1, 5)).final_views[0]
    pm = PortfolioManagerAgent(None, ("SPY",))
    pm.set_inputs([a.run(date(2024, 1, 5)) for a in agents.values()])
    ref = pm.run(date(2024, 1, 5)).views[0]
    assert (ens.direction, round(ens.conviction, 9)) == (ref.direction, round(ref.conviction, 9))


def test_macro_regime_neutral_when_inputs_missing():
    from agents.macro_regime import MacroRegimeAgent
    assert MacroRegimeAgent._classify_regime({}) == "unknown"
    assert MacroRegimeAgent._classify_regime(
        {"industrial_production_yoy_pct_chg_3m": -0.5, "cpi_yoy_pct_chg_3m": 0.2}) == "growth_down_x_infl_up"


def test_alfred_rejects_implausible_release_dates():
    import data.fetchers.alfred as A
    idx = pd.date_range("2022-01-01", periods=12, freq="MS")
    good = pd.Series(idx + pd.Timedelta(days=12), index=idx)
    bad = pd.Series(pd.Timestamp("2026-10-01"), index=idx)    # today's vintage, not first release
    assert A._release_dates_plausible("CPIAUCSL", good)
    assert not A._release_dates_plausible("CPIAUCSL", bad)


def test_claude_client_caches_and_replays(monkeypatch, tmp_path):
    import llm.base as B
    from llm.claude_client import ClaudeClient

    monkeypatch.setattr(B, "LLM_CACHE_ROOT", tmp_path)
    calls = []

    class Msg:
        content = [type("T", (), {"type": "text", "text": '{"direction": 1}'})()]
        model, id, stop_reason = "claude-opus-5-5", "msg_1", "end_turn"
        usage = None

    class FakeMessages:
        def create(self, **kw):
            calls.append(kw)
            return Msg()

    c = ClaudeClient(model="claude-opus-5-5", api_key="test")
    c._client = type("C", (), {"messages": FakeMessages()})()
    r1 = c.complete(system="s", user="u", max_tokens=50)
    r2 = c.complete(system="s", user="u", max_tokens=50)
    assert r1.text == '{"direction": 1}' and not r1.cached and r2.cached
    assert len(calls) == 1 and "temperature" not in calls[0] and calls[0]["thinking"] == {"type": "adaptive"} and calls[0]["output_config"] == {"effort": "low"}
    assert ClaudeClient(model="claude-sonnet-5-5", api_key="x").model == "claude-sonnet-5-5"
    from llm.claude_client import thinking_config
    assert thinking_config("claude-sonnet-5-5") == {"type": "between_tools"}
    assert ClaudeClient(model="claude-opus-5-5", offline=True).model == "claude-opus-5-5"


def test_gdelt_file_aggregation_matches_filters():
    from data.fetchers.gdelt_files import aggregate_rows
    filt = {"BTC": {"themes": ["TECH_CRYPTOCURRENCY"], "keywords": ["bitcoin"]},
            "USO": {"themes": ["ECON_PETROLEUM"], "keywords": ["oil price"]}}
    hdr = "DATE\tNUMARTS\tCOUNTS\tTHEMES\tLOCATIONS\tPERSONS\tORGANIZATIONS\tTONE\tCAMEOEVENTIDS\tSOURCES\tSOURCEURLS\n"
    rows = [
        "20230315\t3\t\tTECH_CRYPTOCURRENCY;X\t\t\t\t-2.0,1,3,4,5,6\t\ta.com\thttps://a.com/x\n",
        "20230315\t1\t\tY\t\t\t\t4.0,1,1,1,1,1\t\tb.com\thttps://b.com/Bitcoin-rally\n",
        "20230315\t5\t\tZ\t\t\t\t1.0,1,1,1,1,1\t\tc.com\thttps://c.com/other\n",
    ]
    out = aggregate_rows([hdr] + rows, filt)
    assert out["n_BTC"] == 4 and abs(out["tone_BTC"] - (3 * -2.0 + 4.0) / 4) < 1e-12
    assert out["n_USO"] == 0


def test_macro_payload_labels_levels_correctly():
    from agents.macro_regime import OUT_NAME
    assert OUT_NAME["cpi_yoy"] == "cpi_index"  # level series must not be called *_yoy


def test_multiround_debate_builds_transcript_and_adjudicates():
    from agents.base_agent import AgentDecision, InstrumentView
    from coordination.debate import MultiRoundDebateProtocol
    from llm.base import LLMResponse

    class Stub:
        def __init__(self, name, d):
            self.name, self.d = name, d
        def run(self, as_of):
            return AgentDecision(agent_name=self.name, as_of=as_of, views=[
                InstrumentView(instrument="SPY", direction=self.d, conviction=0.5, horizon="1w", factors={"x": 1.0})])

    class FakeLLM:
        model = "fake"
        def __init__(self):
            self.calls = []
        def complete(self, *, system, user, max_tokens=2048, **kw):
            self.calls.append(system)
            text = '{"direction": 1, "conviction": 0.6, "rationale": "ok"}' if "adjudicator" in system else "argument"
            return LLMResponse(text=text, model="fake", prompt_hash="h")

    llm = FakeLLM()
    agents = {"a": Stub("a", 1), "b": Stub("b", -1)}
    v = MultiRoundDebateProtocol(agents, llm, ("SPY",)).run(date(2024, 1, 5)).final_views[0]
    assert (v.direction, v.conviction) == (1, 0.6)
    assert sum("bull researcher" in c for c in llm.calls) == 2 and sum("bear researcher" in c for c in llm.calls) == 2
    assert json.loads if True else None


def test_anonymizer_round_trip():
    from llm.anonymize import mask, unmask
    t = '{"instrument": "SPY", "as_of": "2023-03-17", "note": "IEF and BTC-USD"}'
    m = mask(t)
    assert "SPY" not in m and "2023-03-17" not in m and "IEF" not in m
    assert unmask(m).replace("[date]", "2023-03-17") == t

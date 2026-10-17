"""Fundamentals/Carry Agent (spec §4.1).

Per-instrument factor families:
- Equity indices: aggregated P/E, FCF yield, revenue growth (EDGAR 10-K/Q)
- Bonds: real yields, term premium, slope (ALFRED) — wired here
- Gold: real-yield-vs-gold, USD context — wired here
- Oil: EIA inventories + NOAA weather (HDD, hurricanes)
- BTC: on-chain (active addresses, exchange flows, MVRV, hashrate)
- QQQ: GitHub Archive aggregate engineering velocity
- Earnings transcripts: per-quarter guidance/sentiment

Equity index aggregation across constituents is non-trivial (EDGAR XBRL
parsing across hundreds of CIKs); kept stubbed at zero. Bond/gold factors
are wired off ALFRED first-release series and produce non-zero anchor views
when the LLM is unavailable.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from agents._llm_helpers import views_from_llm_or_anchor
from agents.base_agent import BaseAgent, InstrumentView
from data.fetchers.alfred import HEADLINE_SERIES, first_release, prefetch_window
from data.fetchers.blockchain_info import fetch_panel as btc_panel
from data.fetchers.blockchain_info import mvrv_proxy as btc_mvrv_proxy
from data.fetchers.edgar_aggregates import index_pe_yoy
from data.fetchers.eia import crude_oil_inventory_us
from data.fetchers.github_archive import qqq_engineering_velocity
from data.fetchers.noaa import heating_degree_days_us, named_storms_active
from pact_logging import get_logger

log = get_logger(__name__)

EQUITY_INDICES = ("SPY", "QQQ", "IWM", "EEM")
# Subset for which the EDGAR XBRL aggregation produces a credible signal.
# IWM (Russell 2000) is excluded: 30-name constituent list covers only ~5%
# of index weight, making the aggregate unreliable. SPY/QQQ/EEM are
# credible enough — the cap-weighted earnings-yield aggregation in
# edgar_aggregates.py handles loss-making constituents (common in EEM)
# without blowing up the inverse-P/E ratio.
# Equity-index XBRL aggregation is disabled: its TTM sums mix year-to-date and
# quarterly facts and its market caps mix split-adjusted prices with as-reported
# share counts, and the constituent lists are fixed as of 2025 (look-ahead in
# selection). Re-enable only after those are fixed and validated.
EQUITY_INDICES_AGGREGATABLE: tuple[str, ...] = ()
EIA_RELEASE_LAG_DAYS = 6     # weekly period ends Friday; WPSR released the following Wed (Thu after holidays)
NOAA_RELEASE_LAG_DAYS = 10   # monthly GSOM value usable only after the month ends (+ publication lag)
BONDS = ("IEF", "SHY")

SYSTEM_PROMPT = (
    "You are the Fundamentals/Carry Agent. Inputs are per-instrument factor "
    "blobs (real yield, slope, P/E, etc) and a deterministic anchor view. "
    "Refine the anchor — agree, downgrade, or flip — based on the factors. "
    "Never invent factors. Horizon is typically 1q. Output strict JSON: a "
    'list of {"instrument": SYMBOL, "direction": -1|0|1, "conviction": 0..1, '
    '"horizon": "1q", "rationale": "<=240 chars cite which factor"}.'
)


class FundamentalsCarryAgent(BaseAgent):
    name = "fundamentals_carry"

    def __init__(
        self,
        llm_client,
        universe: tuple[str, ...],
        enable_altdata: bool = False,
        cell_window: tuple[date, date] | None = None,
    ):
        super().__init__(llm_client, universe)
        self.enable_altdata = enable_altdata
        self.cell_window = cell_window
        self._crude_inv: pd.Series | None = None
        self._hdd: pd.Series | None = None
        self._storms: pd.Series | None = None
        self._gh_velocity: pd.Series | None = None
        self._btc_panel: pd.DataFrame | None = None

    def extract_factors(self, as_of: date) -> dict[str, dict[str, float]]:
        macro = self._fetch_rates(as_of)

        out: dict[str, dict[str, float]] = {}
        for sym in self.universe:
            f: dict[str, float] = {}
            if sym in EQUITY_INDICES:
                # Equity-index fundamentals via SEC companyfacts XBRL aggregation
                # (data/fetchers/edgar_aggregates.py). Cached per (symbol, as_of)
                # so per-rebalance lookups are fast after first compute.
                # IWM is excluded — see EQUITY_INDICES_AGGREGATABLE.
                if sym in EQUITY_INDICES_AGGREGATABLE:
                    try:
                        agg = index_pe_yoy(sym, as_of, cell_window=self.cell_window)
                        f["agg_pe"] = float(agg.get("agg_pe", 0.0)) if pd.notna(agg.get("agg_pe", 0.0)) else 0.0
                        f["fwd_earnings_yield"] = float(agg.get("fwd_earnings_yield", 0.0))
                        f["rev_growth_yoy"] = float(agg.get("rev_growth_yoy", 0.0))
                        f["fcf_yield"] = float(agg.get("fcf_yield", 0.0))
                    except Exception as e:
                        log.warning("edgar aggregate failed sym=%s: %s", sym, type(e).__name__)
                        f.update({"agg_pe": 0.0, "fwd_earnings_yield": 0.0, "rev_growth_yoy": 0.0, "fcf_yield": 0.0})
                else:
                    # IWM: equity-fundamentals branch deliberately zeroed
                    # (Russell 2000 needs a 200+ name aggregate to be credible).
                    f.update({"agg_pe": 0.0, "fwd_earnings_yield": 0.0, "rev_growth_yoy": 0.0, "fcf_yield": 0.0})
                f["transcript_guidance_score"] = 0.0  # not yet wired
                if self.enable_altdata and sym == "QQQ":
                    f["github_eng_velocity_log_abnormal"] = self._qqq_eng_velocity_z(as_of)
            elif sym in BONDS:
                # Real yield (10y or 2y minus CPI YoY %), nominal level, slope.
                if sym == "IEF":
                    nominal = macro["ten_year"]
                else:
                    nominal = macro["two_year"]
                f.update({
                    "nominal_yield": nominal,
                    "real_yield": _real_yield(nominal, macro["cpi_yoy"]),
                    "slope_2s10s": macro["slope_2s10s"],
                    "fed_funds": macro["fed_funds"],
                })
            elif sym == "GLD":
                f.update({
                    "real_yield_10y": _real_yield(macro["ten_year"], macro["cpi_yoy"]),
                    "fed_funds": macro["fed_funds"],
                    "cpi_yoy": macro["cpi_yoy"],
                })
            elif sym == "USO":
                f["eia_inventory_z"] = self._uso_inventory_z(as_of)
                if self.enable_altdata:
                    f["noaa_hdd_z"] = self._hdd_z(as_of)
                    f["hurricane_active"] = self._hurricane_active(as_of)
            elif sym == "BTC":
                if self.enable_altdata:
                    f.update(self._btc_factors(as_of))
            elif sym == "UUP":
                f.update({"fed_funds": macro["fed_funds"], "cpi_yoy": macro["cpi_yoy"]})
            out[sym] = f
        return out

    def decide(
        self,
        as_of: date,
        factors_by_instrument: dict[str, dict[str, float]],
    ) -> list[InstrumentView]:
        anchor = [
            self._anchor_view(sym, f) for sym, f in factors_by_instrument.items()
        ]

        return views_from_llm_or_anchor(
            self.llm,
            system_prompt=SYSTEM_PROMPT,
            user_payload={
                "as_of": as_of.isoformat(),
                "factors": factors_by_instrument,
                "anchor": [
                    {"instrument": v.instrument, "direction": v.direction, "conviction": v.conviction, "rationale": v.rationale}
                    for v in anchor
                ],
            },
            universe=self.universe,
            default_horizon="1q",
            anchor_views=anchor,
            agent_name=self.name,
        )

    @staticmethod
    def _anchor_view(sym: str, f: dict[str, float]) -> InstrumentView:
        if sym in BONDS:
            return _bond_anchor(sym, f)
        if sym == "GLD":
            return _gold_anchor(f)
        if sym == "USO":
            return _oil_anchor(f)
        return InstrumentView(
            instrument=sym, direction=0, conviction=0.0, horizon="1q",
            factors=f, rationale="fundamentals anchor: factors not wired",
        )

    def _hdd_z(self, as_of: date) -> float:
        """Z-score of latest monthly HDD vs trailing 36-month mean."""
        if self._hdd is None:
            try:
                start = (self.cell_window[0] if self.cell_window else as_of) - timedelta(days=400)
                end = self.cell_window[1] if self.cell_window else as_of
                self._hdd = heating_degree_days_us(start, end)
            except Exception as e:
                log.warning("noaa hdd fetch failed: %s", type(e).__name__)
                self._hdd = pd.Series(dtype=float)
        s = self._hdd
        if s is None or len(s) < 12:
            return 0.0
        # monthly values are dated at the start of the month: usable after month end + lag
        avail = s.index + pd.DateOffset(months=1) + pd.Timedelta(days=NOAA_RELEASE_LAG_DAYS)
        pit = s.loc[avail <= pd.Timestamp(as_of)]
        if len(pit) < 12:
            return 0.0
        latest = float(pit.iloc[-1])
        trailing = pit.iloc[-min(36, len(pit)):]
        mu, sd = float(trailing.mean()), float(trailing.std())
        return (latest - mu) / sd if sd > 0 else 0.0

    def _hurricane_active(self, as_of: date) -> float:
        """Indicator: is `as_of` inside Atlantic hurricane season + a named storm active?"""
        if self._storms is None:
            try:
                start = (self.cell_window[0] if self.cell_window else as_of) - timedelta(days=60)
                end = self.cell_window[1] if self.cell_window else as_of
                self._storms = named_storms_active(start, end)
            except Exception as e:
                log.warning("noaa storms fetch failed: %s", type(e).__name__)
                self._storms = pd.Series(dtype=float)
        s = self._storms
        if s is None or s.empty:
            return 0.0
        pit = s.loc[s.index <= pd.Timestamp(as_of)]
        return float(pit.iloc[-1]) if not pit.empty else 0.0

    def _btc_factors(self, as_of: date) -> dict[str, float]:
        """BTC on-chain factors derived from blockchain.info chart series.

        Returns active-addresses log-abnormal, hashrate log-abnormal,
        transaction-volume z (proxy for exchange netflow), and an MVRV
        proxy (price/200d MA). All standardized over a trailing window
        relative to `as_of` for point-in-time hygiene.
        """
        if self._btc_panel is None:
            try:
                start = (self.cell_window[0] if self.cell_window else as_of) - timedelta(days=400)
                end = self.cell_window[1] if self.cell_window else as_of
                self._btc_panel = btc_panel(start, end)
            except Exception as e:
                log.warning("blockchain_info fetch failed: %s", type(e).__name__)
                self._btc_panel = pd.DataFrame()

        f = {
            "active_addresses_log_abnormal": 0.0,
            "exchange_netflow_z": 0.0,
            "mvrv": 0.0,
            "hashrate_log_abnormal": 0.0,
        }
        if self._btc_panel is None or self._btc_panel.empty:
            return f
        as_of_ts = pd.Timestamp(as_of)
        pit = self._btc_panel.loc[self._btc_panel.index <= as_of_ts]
        if len(pit) < 90:
            return f

        def _log_abnormal(s: pd.Series) -> float:
            tail = np.log1p(s.tail(90))
            if len(tail) < 30 or tail.std() == 0:
                return 0.0
            return float((tail.iloc[-1] - tail.iloc[:-7].mean()) / tail.iloc[:-7].std())

        if "active_addresses" in pit.columns:
            f["active_addresses_log_abnormal"] = _log_abnormal(pit["active_addresses"])
        if "hash_rate" in pit.columns:
            f["hashrate_log_abnormal"] = _log_abnormal(pit["hash_rate"])
        if "transactions" in pit.columns:
            tx = pit["transactions"].tail(90)
            mu, sd = float(tx.mean()), float(tx.std())
            f["exchange_netflow_z"] = (float(tx.iloc[-1]) - mu) / sd if sd > 0 else 0.0
        if "market_price_usd" in pit.columns:
            mvrv = btc_mvrv_proxy(pit["market_price_usd"], window=200)
            if not mvrv.empty and pd.notna(mvrv.iloc[-1]):
                f["mvrv"] = float(mvrv.iloc[-1])
        return f

    def _qqq_eng_velocity_z(self, as_of: date) -> float:
        """Z-score of trailing-7d GitHub event volume vs trailing 90d, log-scaled."""
        if self._gh_velocity is None:
            try:
                start = (self.cell_window[0] if self.cell_window else as_of) - timedelta(days=120)
                end = self.cell_window[1] if self.cell_window else as_of
                self._gh_velocity = qqq_engineering_velocity(start, end)
            except Exception as e:
                log.warning("github_archive fetch failed: %s", type(e).__name__)
                self._gh_velocity = pd.Series(dtype=float)
        s = self._gh_velocity
        if s is None or len(s) < 30:
            return 0.0
        pit = s.loc[s.index <= pd.Timestamp(as_of)]
        if len(pit) < 30:
            return 0.0
        recent = float(np.log1p(pit.tail(7).sum()))
        prior = np.log1p(pit.iloc[-90:-7].rolling(7).sum().dropna())
        if len(prior) < 5:
            return 0.0
        mu, sd = float(prior.mean()), float(prior.std())
        return (recent - mu) / sd if sd > 0 else 0.0

    def _uso_inventory_z(self, as_of: date) -> float:
        """Z-score of latest weekly U.S. crude inventory vs trailing 52-week mean.

        Positive z = stocks above trend (bearish oil); negative z = below
        trend (bullish oil). Caches the full inventory series across calls.
        """
        if self._crude_inv is None:
            if self.cell_window:
                start, end = self.cell_window
                start = start - timedelta(days=400)
            else:
                start = as_of - timedelta(days=400)
                end = as_of
            try:
                df = crude_oil_inventory_us(start, end)
                if df.empty:
                    self._crude_inv = pd.Series(dtype=float)
                else:
                    s = pd.Series(
                        data=df["value"].values,
                        index=pd.to_datetime(df["period"]),
                    ).sort_index()
                    self._crude_inv = s
            except Exception as e:
                log.warning("eia crude inventory fetch failed: %s", type(e).__name__)
                self._crude_inv = pd.Series(dtype=float)

        s = self._crude_inv
        if s is None or len(s) < 8:
            return 0.0
        as_of_ts = pd.Timestamp(as_of)
        # EIA `period` is the week-ending date; the report is published days later
        pit = s.loc[s.index + pd.Timedelta(days=EIA_RELEASE_LAG_DAYS) <= as_of_ts]
        if len(pit) < 8:
            return 0.0
        latest = float(pit.iloc[-1])
        trailing = pit.iloc[-min(52, len(pit)):]
        mu, sd = float(trailing.mean()), float(trailing.std())
        if sd == 0:
            return 0.0
        return (latest - mu) / sd

    def _fetch_rates(self, as_of: date) -> dict[str, float]:
        if self.cell_window and not getattr(self, "_alfred_prefetched", False):
            cell_start = self.cell_window[0] - timedelta(days=560)
            cell_end = self.cell_window[1]
            prefetch_window(
                ("DGS10", "DGS2", "DFEDTARU", "CPIAUCSL"),
                cell_start, cell_end,
            )
            self._alfred_prefetched = True
        start = as_of - timedelta(days=560)
        out: dict[str, float] = {}
        for label in ("ten_year", "two_year", "fed_funds"):
            try:
                s = first_release(HEADLINE_SERIES[label], start, as_of)
                out[label] = float(s.iloc[-1]) if len(s) else 0.0
            except Exception as e:
                log.warning("fundamentals: ALFRED fetch failed series=%s: %s", label, type(e).__name__)
                out[label] = 0.0
        # cpi_yoy: CPIAUCSL is the level series; compute YoY % from latest vs ~252 days back.
        try:
            cpi = first_release(HEADLINE_SERIES["cpi_yoy"], start, as_of)
            cpi = cpi[~cpi.index.duplicated(keep="last")]
            if len(cpi) >= 13:   # latest published month vs the same month a year earlier
                out["cpi_yoy"] = float((cpi.iloc[-1] / cpi.iloc[-13] - 1.0) * 100.0)
            else:
                out["cpi_yoy"] = 0.0
        except Exception as e:
            log.warning("fundamentals: ALFRED CPI fetch failed: %s", type(e).__name__)
            out["cpi_yoy"] = 0.0
        out["slope_2s10s"] = out["ten_year"] - out["two_year"]
        return out


def _real_yield(nominal: float, cpi_yoy_pct: float) -> float:
    """Real yield ≈ nominal − CPI YoY %. Falls back to nominal when CPI missing."""
    if cpi_yoy_pct in (0.0, None) or cpi_yoy_pct != cpi_yoy_pct:  # NaN check
        return nominal
    return nominal - cpi_yoy_pct


def _bond_anchor(sym: str, f: dict[str, float]) -> InstrumentView:
    """Rising real yields → short bonds; deeply negative real yields → long."""
    real_yield = f.get("real_yield", 0.0)
    if real_yield > 1.5:
        direction, conv = -1, 0.4
        why = f"real_yield={real_yield:.2f}% > 1.5% → bonds expensive"
    elif real_yield < -0.5:
        direction, conv = 1, 0.4
        why = f"real_yield={real_yield:.2f}% < -0.5% → bonds cheap"
    else:
        direction, conv = 0, 0.0
        why = f"real_yield={real_yield:.2f}% neutral"
    return InstrumentView(
        instrument=sym, direction=direction, conviction=conv, horizon="1q",
        factors=f, rationale=why,
    )


def _oil_anchor(f: dict[str, float]) -> InstrumentView:
    """High inventories → short oil; low inventories → long oil."""
    z = f.get("eia_inventory_z", 0.0)
    if z > 1.0:
        direction, conv = -1, min(0.3 + (z - 1.0) * 0.1, 0.5)
        why = f"crude inventory z={z:.2f} > 1σ → bearish oil"
    elif z < -1.0:
        direction, conv = 1, min(0.3 + (abs(z) - 1.0) * 0.1, 0.5)
        why = f"crude inventory z={z:.2f} < -1σ → bullish oil"
    else:
        direction, conv = 0, 0.0
        why = f"crude inventory z={z:.2f} (within 1σ band)"
    return InstrumentView(
        instrument="USO", direction=direction, conviction=conv, horizon="1q",
        factors=f, rationale=why,
    )


def _gold_anchor(f: dict[str, float]) -> InstrumentView:
    """Falling real yields → long gold (textbook)."""
    real_yield = f.get("real_yield_10y", 0.0)
    if real_yield < 0.5:
        direction, conv = 1, 0.35
        why = f"10y real_yield={real_yield:.2f}% low → gold supportive"
    elif real_yield > 2.0:
        direction, conv = -1, 0.35
        why = f"10y real_yield={real_yield:.2f}% high → gold headwind"
    else:
        direction, conv = 0, 0.0
        why = f"10y real_yield={real_yield:.2f}% neutral"
    return InstrumentView(
        instrument="GLD", direction=direction, conviction=conv, horizon="1q",
        factors=f, rationale=why,
    )

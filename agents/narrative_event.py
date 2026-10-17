"""Narrative/Event Agent (spec §4.1).

Primary inputs: GDELT GKG (BigQuery, or the free daily files when no GCP project is set).
Secondary inputs: Wikipedia, Google Trends, Polymarket — gated by
`enable_secondary` flag, stubbed at zero until per-source fetchers exist.

Architecture: when constructed with `cell_window=(start, end)`, the agent
pre-fetches the GDELT daily panel for the whole window in a single batched
BigQuery call (cached locally). Subsequent `extract_factors(as_of)` calls
slice the panel up to `as_of` — point-in-time clean, zero per-rebalance
network cost.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from agents._llm_helpers import views_from_llm_or_anchor
from agents.base_agent import BaseAgent, InstrumentView
from data.fetchers.gdelt import GDELT_V2_START, fetch_panel as gdelt_panel
from data.fetchers.google_trends import search_interest_for_instrument
from data.fetchers.polymarket import POLYMARKET_START, event_prob_panel
from data.fetchers.wikipedia import pageviews_for_instrument
from pact_logging import get_logger

log = get_logger(__name__)

# Only factors that are actually computed are sent to the LLM. The original
# list also had geo concentration, novelty, EDGAR 8-K, earnings-surprise and
# guidance fields that were never wired (always 0.0, which an LLM would read
# as "no filings / no news"), and a theme-intensity field identical to volume.
PRIMARY_FACTORS: tuple[str, ...] = (
    "gdelt_event_volume_log",
    "gdelt_tone",
    "gdelt_tone_surprise_wow",
)

SECONDARY_FACTORS: tuple[str, ...] = (
    "wiki_abnormal_attention",
    "wiki_attention_persistence",
    "gtrends_abnormal_level",
    "gtrends_momentum",
    "polymarket_event_prob",
    "polymarket_prob_change",
)

SYSTEM_PROMPT = (
    "You are the Narrative/Event Agent. Inputs are per-instrument factor "
    "blobs from GDELT news and (optionally) attention data. Map them "
    "into per-instrument views over a 1-week horizon. If all factors for an "
    "instrument are zero or missing, emit direction=0, conviction=0. Never "
    "invent factors. Output strict JSON: a list of "
    '{"instrument": SYMBOL, "direction": -1|0|1, "conviction": 0..1, '
    '"horizon": "1w", "rationale": "<=240 chars cite the strongest factor"}.'
)


class NarrativeEventAgent(BaseAgent):
    name = "narrative_event"

    def __init__(
        self,
        llm_client,
        universe: tuple[str, ...],
        enable_secondary: Iterable[str] = (),
        cell_window: tuple[date, date] | None = None,
        gdelt_lookback_days: int = 60,
    ):
        super().__init__(llm_client, universe)
        self.enabled_secondary = set(enable_secondary)
        self.cell_window = cell_window
        self.gdelt_lookback_days = gdelt_lookback_days
        self._gdelt_panel: pd.DataFrame | None = None
        self._wiki_panel: dict[str, pd.Series] | None = None
        self._gtrends_panel: dict[str, pd.Series] | None = None
        self._polymarket_panel: pd.DataFrame | None = None

    def _ensure_gdelt_panel(self, as_of: date) -> pd.DataFrame | None:
        """Fetch GDELT daily panel covering the cell window (or `as_of`-only fallback).

        Returns None when the requested window is fully before GDELT V2 starts
        (pre-2015-02-18) — the agent then falls back to zero anchors.
        """
        if self._gdelt_panel is not None:
            return self._gdelt_panel

        if self.cell_window:
            start, end = self.cell_window
            # Need a lookback before the cell start for surprise/WoW.
            start = start - timedelta(days=self.gdelt_lookback_days)
        else:
            end = as_of
            start = as_of - timedelta(days=self.gdelt_lookback_days * 4)

        if end < GDELT_V2_START:
            log.warning("narrative: cell window ends before GDELT v2 start; skipping fetch")
            self._gdelt_panel = pd.DataFrame()
            return None
        if start < GDELT_V2_START:
            start = GDELT_V2_START

        try:
            panel = gdelt_panel(
                start=start,
                end=end,
                instruments=[s for s in self.universe if s != "SHY"],  # SHY: thin filter, skip
            )
            panel["day"] = pd.to_datetime(panel["day"])
            panel = panel.set_index("day").sort_index()
            self._gdelt_panel = panel
            log.info(
                "narrative gdelt panel ready rows=%d range=%s..%s",
                len(panel), panel.index.min().date() if len(panel) else None,
                panel.index.max().date() if len(panel) else None,
            )
            return panel
        except Exception as e:
            log.warning("narrative gdelt fetch failed (%s); falling back to zero factors", type(e).__name__)
            self._gdelt_panel = pd.DataFrame()
            return None

    def extract_factors(self, as_of: date) -> dict[str, dict[str, float]]:
        panel = self._ensure_gdelt_panel(as_of)
        wiki = self._ensure_secondary("wikipedia") if self._secondary_active("wiki_") else None
        gtrends = self._ensure_secondary("gtrends") if self._secondary_active("gtrends_") else None
        polymarket = self._ensure_secondary("polymarket") if self._secondary_active("polymarket_") else None
        as_of_ts = pd.Timestamp(as_of)

        out: dict[str, dict[str, float]] = {}
        for sym in self.universe:
            f = {k: 0.0 for k in PRIMARY_FACTORS}
            for k in SECONDARY_FACTORS:
                if k in self.enabled_secondary:
                    f[k] = 0.0

            # GDELT primary
            if panel is not None and not panel.empty:
                n_col = f"n_{sym}"
                tone_col = f"tone_{sym}"
                if n_col in panel.columns:
                    # a UTC day's GDELT file is complete only after the day ends
                    # (published the next morning): use days < as_of
                    pit = panel.loc[panel.index < as_of_ts]
                    if len(pit) >= 7:
                        recent_7d = pit.tail(7)
                        n_recent = float(recent_7d[n_col].sum())
                        tone_recent = float(recent_7d[tone_col].mean()) if tone_col in pit.columns else 0.0
                        f["gdelt_event_volume_log"] = float(np.log1p(n_recent))
                        f["gdelt_tone"] = tone_recent if pd.notna(tone_recent) else 0.0
                        if len(pit) >= 35:
                            prior = pit.iloc[-35:-7]
                            prior_tone = float(prior[tone_col].mean()) if tone_col in prior.columns else 0.0
                            prior_std = float(prior[tone_col].std()) if tone_col in prior.columns else 0.0
                            if prior_std > 0:
                                f["gdelt_tone_surprise_wow"] = (tone_recent - prior_tone) / prior_std

            # Wikipedia secondary
            if wiki and "wiki_abnormal_attention" in self.enabled_secondary:
                s = wiki.get(sym)
                if s is not None and not s.empty:
                    # pageviews for day d are complete only after d ends: use days < as_of
                    pit = s.loc[s.index < as_of_ts]
                    if len(pit) >= 30:
                        recent = pit.tail(7).mean()
                        prior = pit.iloc[-30:-7]
                        mu, sd = float(prior.mean()), float(prior.std())
                        if sd > 0:
                            f["wiki_abnormal_attention"] = (recent - mu) / sd
                        if "wiki_attention_persistence" in self.enabled_secondary:
                            f["wiki_attention_persistence"] = float(pit.tail(28).autocorr(lag=1) or 0.0)

            # Google Trends secondary
            if gtrends and "gtrends_abnormal_level" in self.enabled_secondary:
                s = gtrends.get(sym)
                if s is not None and not s.empty:
                    # weekly points are labelled by week start: use completed weeks only
                    pit = s.loc[s.index + pd.Timedelta(days=7) <= as_of_ts]
                    if len(pit) >= 13:
                        recent = float(pit.tail(2).mean())
                        prior = pit.iloc[-13:-2]
                        mu, sd = float(prior.mean()), float(prior.std())
                        if sd > 0:
                            f["gtrends_abnormal_level"] = (recent - mu) / sd
                        if "gtrends_momentum" in self.enabled_secondary and len(pit) >= 4:
                            # relative change: invariant to Trends' window-wide 0-100 rescaling
                            base = float(pit.iloc[-4])
                            f["gtrends_momentum"] = float(pit.iloc[-1] / base - 1.0) if base > 0 else 0.0

            # Polymarket secondary (Window B only)
            if polymarket is not None and not polymarket.empty and "polymarket_event_prob" in self.enabled_secondary:
                if sym in polymarket.columns:
                    pit = polymarket[sym].loc[polymarket.index <= as_of_ts - pd.Timedelta(hours=48)].dropna()
                    if not pit.empty:
                        f["polymarket_event_prob"] = float(pit.iloc[-1])
                        if "polymarket_prob_change" in self.enabled_secondary and len(pit) >= 7:
                            f["polymarket_prob_change"] = float(pit.iloc[-1] - pit.iloc[-7])
            out[sym] = f
        return out

    def _secondary_active(self, prefix: str) -> bool:
        return any(k.startswith(prefix) for k in self.enabled_secondary)

    def _ensure_secondary(self, source: str):
        """Lazily fetch + cache a secondary panel for the cell window."""
        if not self.cell_window:
            return None
        start = self.cell_window[0] - timedelta(days=60)
        end = self.cell_window[1]

        if source == "wikipedia":
            if self._wiki_panel is None:
                self._wiki_panel = {}
                for sym in self.universe:
                    try:
                        self._wiki_panel[sym] = pageviews_for_instrument(sym, start, end)
                    except Exception as e:
                        log.warning("wikipedia fetch failed sym=%s: %s", sym, type(e).__name__)
                        self._wiki_panel[sym] = pd.Series(dtype=float)
            return self._wiki_panel

        if source == "gtrends":
            if self._gtrends_panel is None:
                self._gtrends_panel = {}
                for sym in self.universe:
                    try:
                        self._gtrends_panel[sym] = search_interest_for_instrument(sym, start, end)
                    except Exception as e:
                        log.warning("gtrends fetch failed sym=%s: %s", sym, type(e).__name__)
                        self._gtrends_panel[sym] = pd.Series(dtype=float)
            return self._gtrends_panel

        if source == "polymarket":
            # Window B only; spec §6.4.
            if self.cell_window[1] < POLYMARKET_START:
                self._polymarket_panel = pd.DataFrame()
                return self._polymarket_panel
            if self._polymarket_panel is None:
                pm_start = max(start, POLYMARKET_START)
                try:
                    self._polymarket_panel = event_prob_panel(self.universe, pm_start, end)
                except Exception as e:
                    log.warning("polymarket fetch failed: %s", type(e).__name__)
                    self._polymarket_panel = pd.DataFrame()
            return self._polymarket_panel
        return None

    def decide(
        self,
        as_of: date,
        factors_by_instrument: dict[str, dict[str, float]],
    ) -> list[InstrumentView]:
        anchor = [
            self._anchor_view(sym, f) for sym, f in factors_by_instrument.items()
        ]

        if all(_all_zero(f) for f in factors_by_instrument.values()):
            return anchor

        return views_from_llm_or_anchor(
            self.llm,
            system_prompt=SYSTEM_PROMPT,
            user_payload={
                "as_of": as_of.isoformat(),
                "factors": factors_by_instrument,
                "enabled_secondary": sorted(self.enabled_secondary),
            },
            universe=self.universe,
            default_horizon="1w",
            anchor_views=anchor,
            agent_name=self.name,
        )

    @staticmethod
    def _anchor_view(sym: str, f: dict[str, float]) -> InstrumentView:
        """Anchor view from GDELT factors when no LLM is available.

        Tone surprise crossing thresholds → directional view.
        """
        surprise = f.get("gdelt_tone_surprise_wow", 0.0)
        tone = f.get("gdelt_tone", 0.0)
        # Strong negative tone surprise → bearish; strong positive → bullish.
        if surprise > 0.5 and tone > 0:
            direction, conviction = 1, min(0.3 + abs(surprise) * 0.1, 0.6)
            why = f"GDELT tone surprise +{surprise:.2f} σ; tone={tone:.2f}"
        elif surprise < -0.5 and tone < 0:
            direction, conviction = -1, min(0.3 + abs(surprise) * 0.1, 0.6)
            why = f"GDELT tone surprise {surprise:.2f} σ; tone={tone:.2f}"
        else:
            direction, conviction = 0, 0.0
            why = f"GDELT tone={tone:.2f} surprise={surprise:.2f} (below threshold)"
        return InstrumentView(
            instrument=sym, direction=direction, conviction=conviction, horizon="1w",
            factors=f, rationale=why,
        )


def _all_zero(f: dict[str, float]) -> bool:
    return all(v == 0.0 for v in f.values())

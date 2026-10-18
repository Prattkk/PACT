"""Shared LLM-call helpers for agents (spec §4.2).

Every LLM-driven agent follows the same pattern:
1. Build a deterministic anchor view from factors (always works, even with no LLM).
2. If an LLM is available, ask it to refine/replace the anchor.
3. If the LLM call fails or returns unparseable output, fall back to the anchor.

`views_from_llm_or_anchor` encapsulates step 2-3 so the agents stay short.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agents.base_agent import Horizon, InstrumentView
from pact_logging import get_logger

log = get_logger(__name__)

# Process-wide tallies of LLM fallbacks, keyed "agent:reason". The engine
# snapshots these before/after a backtest and stores the difference in the
# cell metadata, so the share of decisions that silently fell back to the
# deterministic anchor is reported rather than hidden.
LLM_STATS: Counter[str] = Counter()


def parse_view_list(
    text: str,
    universe: tuple[str, ...],
    default_horizon: Horizon,
) -> list[InstrumentView]:
    """Parse an LLM JSON list into InstrumentViews. Returns [] on failure."""
    data = _try_load_json(text)
    if not isinstance(data, list):
        return []
    out: list[InstrumentView] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        entry.setdefault("horizon", default_horizon)
        try:
            out.append(InstrumentView(**entry))
        except Exception:
            continue
    return out


def parse_json_object(text: str) -> dict | None:
    """Parse one JSON object from LLM output, tolerating code fences and prose."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(text[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _log_failure(component: str, exc: Exception) -> None:
    import os

    path = os.environ.get("PACT_FAILURE_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps({"protocol": "agent", "component": component,
                                 "type": type(exc).__name__, "msg": str(exc)[:300]}) + "\n")


def tally(component: str, outcome: str, n: int = 1) -> None:
    """Record an LLM call outcome: 'calls', 'call_failed', 'unparseable', 'symbols_filled_from_anchor'."""
    LLM_STATS[f"{component}:{outcome}"] += n


def _try_load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to recover a JSON array embedded in prose.
    i = text.find("[")
    j = text.rfind("]")
    if i >= 0 and j > i:
        try:
            return json.loads(text[i : j + 1])
        except json.JSONDecodeError:
            return None
    return None


def views_from_llm_or_anchor(
    llm,
    *,
    system_prompt: str,
    user_payload: dict,
    universe: tuple[str, ...],
    default_horizon: Horizon,
    anchor_views: list[InstrumentView],
    agent_name: str = "agent",
    max_tokens: int = 1500,
) -> list[InstrumentView]:
    """Try LLM; on any failure fall back to `anchor_views`.

    Always returns one view per universe symbol — fills missing symbols from
    the anchor so downstream protocols never see a partial output.
    """
    if llm is None:
        return anchor_views
    LLM_STATS[f"{agent_name}:calls"] += 1
    try:
        resp = llm.complete(
            system=system_prompt,
            user=json.dumps(user_payload, default=float),
            max_tokens=max_tokens,
        )
    except Exception as e:
        LLM_STATS[f"{agent_name}:call_failed"] += 1
        log.warning("agent=%s llm.complete failed (%s); falling back to anchor", agent_name, type(e).__name__)
        _log_failure(agent_name, e)
        return anchor_views

    parsed = parse_view_list(resp.text, universe, default_horizon)
    if not parsed:
        LLM_STATS[f"{agent_name}:unparseable"] += 1
        log.warning("agent=%s llm output unparseable; falling back to anchor", agent_name)
        return anchor_views

    by_anchor_f = {v.instrument: v.factors for v in anchor_views}
    parsed = [v if v.factors else v.model_copy(update={"factors": by_anchor_f.get(v.instrument, {})})
              for v in parsed]   # keep the factors so downstream protocols (e.g. debate) can read them
    seen = {v.instrument for v in parsed}
    n_missing = sum(1 for s in universe if s.upper() not in seen)
    if n_missing:
        LLM_STATS[f"{agent_name}:symbols_filled_from_anchor"] += n_missing
    by_anchor = {v.instrument: v for v in anchor_views}
    for sym in universe:
        s = sym.upper()
        if s not in seen and s in by_anchor:
            parsed.append(by_anchor[s])
    log.debug("agent=%s llm produced %d views (anchor=%d)", agent_name, len(parsed), len(anchor_views))
    return parsed

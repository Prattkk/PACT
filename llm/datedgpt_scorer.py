"""Contamination-clean regime: DatedGPT instruct models as log-probability scorers.

DatedGPT (Yan et al. 2026) releases one ~1.3B-parameter Llama-architecture
instruct model per year, trained only on data available up to the end of that
year. A decision dated in year Y uses the year Y-1 vintage, so the model cannot
have seen any outcome in the test window.

These models cannot follow the structured JSON prompts used by the Claude
regimes (their free-form output is incoherent; see the probe in
scripts/probe_datedgpt.py), so the client does not generate text. For each
instrument in a specialist's payload it renders a short question with the same
factors (and the rule-based anchor, where the specialist has one) and reads the
log-probabilities of the answers "Long" and "Short". The log-odds are
standardised across the instruments in the same call (removing the model's
instrument-independent bias towards "Long" using only same-date information);
direction = sign(z) when |z| > 0.5 and conviction = min(|z| / 2, 1). The result
is returned as the JSON view list the agents already parse.

Only multi-instrument specialist calls are supported (macro, narrative,
fundamentals, cross-asset), which covers the ensemble, pipeline, placebo
pipeline and hierarchical protocols. The technical agent's LLM call only writes
a rationale and gets a fixed string. Scoring is deterministic, so this regime
has no sampling variance.
"""

from __future__ import annotations

import json
import os
import threading
from statistics import mean, pstdev

from llm.base import LLMClient, LLMResponse, canonicalize_prompt, prompt_hash, read_cache, write_cache
from pact_logging import get_logger

log = get_logger(__name__)

REPO = "datedgpt/datedgpt-{y}-instruct"
# Hugging Face commit SHAs pinned for reproducibility
REVISIONS = {2015: "09dcadf9e2c091ec5b9ab5edf58c284cd3250469",   # 2015-2020: unused-period run (pre-registration v2)
             2016: "8d66e16b20f44fa6c0fb6083f91789af1523a989",
             2017: "aa95ea00fca700811f3d14e9991ef59101182376",
             2018: "05ad5bb33bb859ee5f979cca96812eb5a14384a7",
             2019: "a1d6689fd350fff2b969860762b89f25df899f57",
             2020: "407cf0c746be710a268e0ed54c012d43d0453b6c",
             2021: "ac1b847bf836e2168af1a468f487b1f39e8c75c5",
             2022: "3fde1bcb0e69c363a786c93320e76b32d34ea5cc",
             2023: "7314fa3e58a921c9b0cd3e320cf3f44145b94e5c"}
SCORER_VERSION = "scorer-v1"
SCORER_V2 = "scorer-v2-fp32"   # controlled-experiment questions (pre-registration v3)
SHORT = {"macro_regime": "macro", "narrative_event": "news", "fundamentals_carry": "fundamentals",
         "technical_trend": "technical", "cross_asset_transmission": "cross-asset"}
DESCRIPTIONS = {
    "SPY": "US large-cap stock index ETF", "QQQ": "US technology-heavy stock index ETF",
    "IWM": "US small-cap stock index ETF", "EEM": "emerging-markets stock ETF",
    "IEF": "7-10 year US Treasury bond ETF", "SHY": "1-3 year US Treasury bond ETF",
    "GLD": "gold ETF", "USO": "crude oil ETF", "UUP": "US dollar index ETF", "BTC": "bitcoin",
}
AGENTS = {  # system-prompt prefix -> (payload key holding per-instrument data, label, horizon)
    "You are the Cross-Asset Market": ("factors", "Recent return and correlations with major drivers", "month"),
    "You are the Macro Regime": ("macro", "Macroeconomic data (first releases)", "quarter"),
    "You are the Narrative/Event": ("factors", "News coverage and tone indicators", "week"),
    "You are the Fundamentals/Carry": ("factors", "Fundamental and carry indicators", "quarter"),
    "You are the Cross-Asset": ("upstream_views", "Views of the macro and news analysts (direction -1/0/1, conviction)", "month"),
}


class DatedGPTScorer(LLMClient):
    def __init__(self, offline: bool = False, revisions: dict[int, str] | None = None) -> None:
        self.model = f"datedgpt-instruct-{SCORER_VERSION}"
        self.offline = offline
        self.revisions = revisions or dict(REVISIONS)
        self._loaded: tuple[int, object, object] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ model
    def _model(self, vintage: int, fp32: bool = False):
        if self._loaded and self._loaded[0] == (vintage, fp32):
            return self._loaded[1], self._loaded[2]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._loaded = None   # one model in memory at a time
        repo = REPO.format(y=vintage)
        rev = self.revisions.get(vintage)
        log.info("loading %s@%s fp32=%s", repo, rev, fp32)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16,
                                                     low_cpu_mem_usage=True).eval()
        if fp32:
            model = model.float()
        self._loaded = ((vintage, fp32), tok, model)
        return tok, model

    def _log_odds(self, vintage: int, question: str, fp32: bool = False) -> float:
        import torch

        tok, model = self._model(vintage, fp32)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": question + "\nAnswer with one word: Long or Short."}], tokenize=False)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            lp = torch.log_softmax(model(**ids).logits[0, -1].float(), -1)
        long_id = tok("Long", add_special_tokens=False)["input_ids"][0]
        short_id = tok("Short", add_special_tokens=False)["input_ids"][0]
        return float(lp[long_id] - lp[short_id])

    # --------------------------------------------------------------- prompts
    @staticmethod
    def kind(system: str) -> str:
        """'v1' for the frozen specialist questions; 'v2' for the controlled-experiment questions
        (pre-registration v3: cross-asset market evidence, revision, judge), scored in float32."""
        if (system.startswith("You are the Cross-Asset Market") or "second, final assessment" in system
                or system.startswith("You are the portfolio adjudicator")):
            return "v2"
        return "v1"

    @staticmethod
    def _assess(v: dict | None) -> str:
        if not v:
            return "no view"
        d = {1: "long", -1: "short"}.get(int(v.get("direction", 0)), "neutral")
        return d if d == "neutral" else f"{d} (conviction {float(v.get('conviction', 0)):.2f})"

    @classmethod
    def questions(cls, system: str, payload: dict) -> dict[str, str]:
        if "second, final assessment" in system:
            horizon = "quarter" if '"1q"' in system else ("month" if '"1m"' in system else "week")
            out = {}
            for sym, desc in DESCRIPTIONS.items():
                facts = payload.get("evidence", {}).get(sym)
                if not facts:
                    continue
                q = f"Asset: {sym} ({desc}).\nYour evidence: " + ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in facts.items()) + "."
                q += f"\nYour first assessment: {cls._assess(payload.get('first_view', {}).get(sym))}."
                peers = payload.get("peer_views", {}).get(sym)
                if peers is not None:
                    q += "\nOther analysts' first assessments: " + "; ".join(
                        f"{SHORT.get(a, a)} {cls._assess(v)}" for a, v in peers.items()) + "."
                q += f"\nOver the next {horizon}, should an investor be long or short this asset?"
                out[sym] = q
            return out
        if system.startswith("You are the portfolio adjudicator"):
            out = {}
            for sym, desc in DESCRIPTIONS.items():
                rows = payload.get("instruments", {}).get(sym)
                if not rows:
                    continue
                q = f"Asset: {sym} ({desc}).\nAnalysts' final assessments: " + "; ".join(
                    f"{SHORT.get(r['agent'], r['agent'])} {cls._assess(r)}" for r in rows) + "."
                q += "\nOver the next week, should an investor be long or short this asset?"
                out[sym] = q
            return out
        spec = next((v for k, v in AGENTS.items() if system.startswith(k)), None)
        if spec is None:
            raise NotImplementedError("clean regime supports multi-instrument specialist calls only")
        key, label, horizon = spec
        anchors = {a["instrument"]: a for a in payload.get("anchor", [])}
        out = {}
        for sym, desc in DESCRIPTIONS.items():
            data = payload[key] if key == "macro" else payload.get(key, {}).get(sym)
            if not data:
                continue
            facts = ", ".join(f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
                              for k, v in data.items())
            q = f"Asset: {sym} ({desc}).\n{label}: {facts}."
            a = anchors.get(sym)
            if a is not None and a.get("direction", 0) != 0:
                q += f"\nA rule-based model suggests {'long' if a['direction'] > 0 else 'short'} " \
                     f"(conviction {a.get('conviction', 0):.2f})."
            q += f"\nOver the next {horizon}, should an investor be long or short this asset?"
            out[sym] = q
        return out

    # ------------------------------------------------------------- interface
    def complete(self, *, system: str, user: str, temperature: float = 0.0, seed: int | None = 42,
                 max_tokens: int = 2048) -> LLMResponse:
        if system.startswith("You are a terse technical-analysis explainer"):
            return LLMResponse(text="rule-based anchor (no LLM rationale in the clean regime)",
                               model=self.model, prompt_hash="na", cached=True)
        payload = json.loads(user)
        vintage = int(payload["as_of"][:4]) - 1
        v2 = self.kind(system) == "v2"
        version = SCORER_V2 if v2 else SCORER_VERSION
        model_key = f"datedgpt-{vintage}-instruct@{self.revisions.get(vintage, 'main')}|{version}"
        canonical = canonicalize_prompt(system=system, user=user, model=model_key, temperature=None, seed=None)
        h = prompt_hash(canonical)
        hit = read_cache(self.model, h)
        if hit is not None:
            return LLMResponse(text=hit["response"]["text"], model=self.model, prompt_hash=h,
                               raw=hit["response"], cached=True)
        if self.offline:
            raise KeyError(f"Cache miss in offline mode for prompt hash {h}.")
        qs = self.questions(system, payload)
        with self._lock:
            scores = {s: self._log_odds(vintage, q, fp32=v2) for s, q in qs.items()}
        views = []
        if len(scores) >= 3:
            mu, sd = mean(scores.values()), pstdev(scores.values())
            for s, v in scores.items():
                z = (v - mu) / sd if sd > 0 else 0.0
                d = (1 if z > 0 else -1) if abs(z) > 0.5 else 0
                views.append({"instrument": s, "direction": d, "conviction": round(min(abs(z) / 2, 1.0), 4),
                              "horizon": "1w", "rationale": f"DatedGPT-{vintage} log-odds z={z:+.2f}"})
        text = json.dumps(views)
        payload_out = {"text": text, "log_odds": scores, "vintage": vintage, "scorer": version}
        write_cache(self.model, h, canonical, payload_out)
        return LLMResponse(text=text, model=self.model, prompt_hash=h, raw=payload_out, cached=False)


class NameOnlyScorer(DatedGPTScorer):
    """Diagnostic (pre-registration v2, C1): the same scorer asked with the asset description
    only, no factors and no anchor, so its views are the vintage's fixed instrument tilt."""

    def __init__(self, offline: bool = False) -> None:
        super().__init__(offline=offline)
        self.model = f"datedgpt-instruct-nameonly-{SCORER_VERSION}"
        self._memo: dict[tuple[int, str], float] = {}

    @staticmethod
    def questions(system: str, payload: dict) -> dict[str, str]:
        spec = next((v for k, v in AGENTS.items() if system.startswith(k)), None)
        if spec is None:
            raise NotImplementedError("clean regime supports multi-instrument specialist calls only")
        full = DatedGPTScorer.questions(system, payload)          # same instrument set as the real call
        return {s: f"Asset: {s} ({DESCRIPTIONS[s]}).\nOver the next {spec[2]}, should an investor be long or short this asset?"
                for s in full}

    def _log_odds(self, vintage: int, question: str, fp32: bool = False) -> float:
        if (vintage, question) not in self._memo:
            self._memo[(vintage, question)] = super()._log_odds(vintage, question, fp32)
        return self._memo[(vintage, question)]

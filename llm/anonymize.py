"""Identifier masking for the contamination test.

Wraps any LLM client: before a call, tickers are replaced by asset-class
aliases (EQUITY_1, BOND_1, ...) and ISO dates by "[date]"; after the call the
aliases in the response are mapped back to tickers so the parsers see the
usual symbols. Asset classes are kept because they carry legitimate priors
(e.g. bonds fall when yields rise); names and dates are what would let a model
recall realised 2022-2024 outcomes. Numeric factor values are unchanged and
can still hint at the period, so this masks direct identifiers only.
"""

from __future__ import annotations

import re

from llm.base import LLMClient, LLMResponse

ALIASES: dict[str, str] = {
    "SPY": "EQUITY_1", "QQQ": "EQUITY_2", "IWM": "EQUITY_3", "EEM": "EQUITY_4",
    "IEF": "BOND_1", "SHY": "BOND_2", "GLD": "COMMODITY_1", "USO": "COMMODITY_2",
    "UUP": "FX_1", "BTC": "CRYPTO_1", "VIX": "VOLINDEX_1",
}
_REVERSE = {v: k for k, v in ALIASES.items()}
_TICKER_RE = re.compile(r"\b(" + "|".join(ALIASES) + r")\b")
_ALIAS_RE = re.compile(r"\b(" + "|".join(_REVERSE) + r")\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def mask(text: str) -> str:
    return _DATE_RE.sub("[date]", _TICKER_RE.sub(lambda m: ALIASES[m.group(1)], text))


def unmask(text: str) -> str:
    return _ALIAS_RE.sub(lambda m: _REVERSE[m.group(1)], text)


class AnonymizingClient(LLMClient):
    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.model = inner.model

    def complete(self, *, system: str, user: str, temperature: float = 0.0,
                 seed: int | None = 42, max_tokens: int = 2048) -> LLMResponse:
        r = self.inner.complete(system=mask(system), user=mask(user), temperature=temperature,
                                seed=seed, max_tokens=max_tokens)
        return LLMResponse(text=unmask(r.text), model=r.model, prompt_hash=r.prompt_hash,
                           raw=r.raw, cached=r.cached)

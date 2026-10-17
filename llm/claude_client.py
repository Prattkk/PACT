"""Frontier path on the Anthropic Messages API — pinned model ID, fully cached.

Drop-in replacement for `GPT4oClient` (same `complete` signature and the same
SHA-256 prompt cache in `llm/base.py`).

Reproducibility notes (report these in the paper):
- The model ID is pinned (`claude-opus-5-5` by default; override with
  `PACT_CLAUDE_MODEL`). The API's returned `model` and message `id` are
  stored with every cached response.
- The Messages API exposes no temperature, top-p or seed for this model, so
  sampling is the API default; thinking is adaptive at effort "low" (this
  model does not accept thinking=disabled).
  The cache key therefore uses temperature=None and seed=None, and replaying
  from the cache is what makes a rerun exact.
- Every training-data cutoff of current Claude models is after 2024, so
  all LLM cells on the 2022-2024 window are in-sample for the model.
"""

from __future__ import annotations

import os
import time
from typing import Any

from llm.base import (
    LLMClient,
    LLMResponse,
    canonicalize_prompt,
    prompt_hash,
    read_cache,
    write_cache,
)
from pact_logging import get_logger

log = get_logger(__name__)

DEFAULT_MODEL = os.environ.get("PACT_CLAUDE_MODEL", "claude-opus-5-5")
# Opus 5.5 does not accept thinking=disabled; pin adaptive thinking at the
# lowest effort so answers stay short and the configuration is fixed.
THINKING = {"type": "adaptive"}
EFFORT = os.environ.get("PACT_CLAUDE_EFFORT", "low")
# Thinking tokens share max_tokens, so short requests get a floor; billing is
# per token used, and prompts still ask for short answers.
MIN_MAX_TOKENS = 4096


class ClaudeClient(LLMClient):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        offline: bool = False,
    ) -> None:
        self.model = model
        self.offline = offline
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if self.offline:
                raise RuntimeError("ClaudeClient is offline; every prompt must be a cache hit.")
            if not self._api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set (put it in the gitignored .env).")
            from anthropic import Anthropic

            # plain float timeout: newer SDKs reject httpx.Timeout objects
            self._client = Anthropic(api_key=self._api_key, timeout=180.0, max_retries=6)
        return self._client

    def _create(self, *, system: str, user: str, max_tokens: int):
        return self._ensure_client().messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max(max_tokens, MIN_MAX_TOKENS),
            thinking=THINKING,
            output_config={"effort": EFFORT},
        )

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        seed: int | None = 42,  # accepted for interface parity; the API has no seed
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # temperature is accepted for interface parity but cannot be sent
        canonical = canonicalize_prompt(system=system, user=user, model=f"{self.model}|effort={EFFORT}",
                                        temperature=None, seed=None)
        h = prompt_hash(canonical)

        hit = read_cache(self.model, h)
        if hit is not None:
            resp = hit["response"]
            return LLMResponse(text=resp["text"], model=self.model, prompt_hash=h, raw=resp, cached=True)

        if self.offline:
            log.error("offline cache miss model=%s hash=%s", self.model, h[:12])
            raise KeyError(f"Cache miss in offline mode for prompt hash {h}.")

        log.info("claude api call model=%s hash=%s", self.model, h[:12])
        t0 = time.monotonic()
        msg = self._create(system=system, user=user, max_tokens=max_tokens)
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        usage = msg.usage.model_dump() if getattr(msg, "usage", None) else None
        payload: dict[str, Any] = {
            "text": text,
            "api_model": getattr(msg, "model", None),
            "message_id": getattr(msg, "id", None),
            "usage": usage,
            "finish_reason": getattr(msg, "stop_reason", None),
            "sampling": "API default (no temperature parameter)",
            "thinking": THINKING["type"],
            "effort": EFFORT,
            "max_tokens_sent": max(max_tokens, MIN_MAX_TOKENS),
            "latency_s": round(time.monotonic() - t0, 3),
        }
        write_cache(self.model, h, canonical, payload)
        log.info(
            "claude api ok model=%s hash=%s in=%s out=%s stop=%s",
            payload["api_model"], h[:12],
            (usage or {}).get("input_tokens"), (usage or {}).get("output_tokens"),
            payload["finish_reason"],
        )
        return LLMResponse(text=text, model=self.model, prompt_hash=h, raw=payload, cached=False)

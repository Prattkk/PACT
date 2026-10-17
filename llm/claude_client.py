"""Frontier path on the Anthropic Messages API — pinned model ID, fully cached.

Drop-in replacement for `GPT4oClient` (same `complete` signature and the same
SHA-256 prompt cache in `llm/base.py`).

Reproducibility notes (report these in the paper):
- The model ID is pinned (`claude-opus-5-5` by default; override with
  `PACT_CLAUDE_MODEL`). The API's returned `model` and message `id` are
  stored with every cached response.
- temperature=0. The Messages API has no `seed`, so the cache key uses
  seed=None; replaying from the cache is what makes a rerun exact.
- If the API refuses an explicit temperature for this model, the call is
  retried without it and `temperature_sent=False` is recorded in the cache
  record, so the paper can say exactly what was sent.
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
        self._send_temperature = True

    def _ensure_client(self):
        if self._client is None:
            if self.offline:
                raise RuntimeError("ClaudeClient is offline; every prompt must be a cache hit.")
            if not self._api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set (put it in the gitignored .env).")
            import httpx
            from anthropic import Anthropic

            self._client = Anthropic(
                api_key=self._api_key,
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
                max_retries=5,
            )
        return self._client

    def _create(self, *, system: str, user: str, temperature: float, max_tokens: int):
        client = self._ensure_client()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        if self._send_temperature:
            kwargs["temperature"] = temperature
        try:
            return client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — narrow below
            from anthropic import BadRequestError

            if (
                isinstance(exc, BadRequestError)
                and self._send_temperature
                and "temperature" in str(exc).lower()
            ):
                log.warning("model %s refuses explicit temperature; retrying without it", self.model)
                self._send_temperature = False
                kwargs.pop("temperature", None)
                return client.messages.create(**kwargs)
            raise

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        seed: int | None = 42,  # accepted for interface parity; the API has no seed
        max_tokens: int = 2048,
    ) -> LLMResponse:
        canonical = canonicalize_prompt(
            system=system, user=user, model=self.model, temperature=temperature, seed=None
        )
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
        msg = self._create(system=system, user=user, temperature=temperature, max_tokens=max_tokens)
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        usage = msg.usage.model_dump() if getattr(msg, "usage", None) else None
        payload: dict[str, Any] = {
            "text": text,
            "api_model": getattr(msg, "model", None),
            "message_id": getattr(msg, "id", None),
            "usage": usage,
            "finish_reason": getattr(msg, "stop_reason", None),
            "temperature_sent": self._send_temperature,
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

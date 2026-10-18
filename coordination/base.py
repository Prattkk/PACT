"""Coordination protocol interface.

Each cell of the headline 7×2 ablation matrix (spec §6.2) is a subclass
of `CoordinationProtocol`. The harness calls `run(as_of)` and gets back
the same `CoordinationResult` shape regardless of protocol.
"""

from __future__ import annotations

import json
import os

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from agents.base_agent import AgentDecision, InstrumentView
from pact_logging import get_logger

log = get_logger(__name__)


@dataclass
class CoordinationResult:
    protocol: str
    as_of: date
    final_views: list[InstrumentView]
    per_agent_decisions: list[AgentDecision] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class CoordinationProtocol(ABC):
    name: str = "base"

    def __init__(self, agents: dict[str, object]) -> None:
        # `agents` keyed by agent name; protocol picks what it needs.
        self.agents = agents
        # Count every swallowed exception so silent fallbacks are visible in
        # cell metadata (an agent that fails is otherwise indistinguishable
        # from an agent that is flat).
        self.failures: Counter[str] = Counter()

    def _record_failure(self, component: str, exc: Exception) -> None:
        self.failures[component] += 1
        path = os.environ.get("PACT_FAILURE_LOG")
        if path:
            with open(path, "a") as fh:
                fh.write(json.dumps({"protocol": self.name, "component": component,
                                     "type": type(exc).__name__, "msg": str(exc)[:300]}) + "\n")
        log.warning("protocol=%s component=%s failed: %s: %s",
                    self.name, component, type(exc).__name__, str(exc)[:200])

    @abstractmethod
    def run(self, as_of: date) -> CoordinationResult: ...

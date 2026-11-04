from coordination.base import CoordinationProtocol, CoordinationResult
from coordination.debate import DebateProtocol, MultiRoundDebateProtocol
from coordination.deterministic_only import DeterministicOnlyProtocol
from coordination.hierarchical import HierarchicalProtocol
from coordination.independent_ensemble import IndependentEnsembleProtocol
from coordination.llm_plus_anchor import LLMPlusAnchorProtocol
from coordination.sequential_pipeline import PlaceboPipelineProtocol, SequentialPipelineProtocol, SEEDED_PLACEBOS
from coordination.single_agent import SelfConsistentMonolithProtocol, SingleAgentProtocol

REGISTRY = {
    "single_agent": SingleAgentProtocol,
    "single_agent_sc": SelfConsistentMonolithProtocol,
    "independent_ensemble": IndependentEnsembleProtocol,
    "sequential_pipeline": SequentialPipelineProtocol,
    "pipeline_placebo": PlaceboPipelineProtocol,
    "hierarchical": HierarchicalProtocol,
    "debate": DebateProtocol,
    "debate_multiround": MultiRoundDebateProtocol,
    "deterministic_only": DeterministicOnlyProtocol,
    "llm_plus_anchor": LLMPlusAnchorProtocol,
    **SEEDED_PLACEBOS,          # diagnostic placebos with random derangements (pre-registration v2)
}

__all__ = [
    "CoordinationProtocol",
    "CoordinationResult",
    "DebateProtocol",
    "MultiRoundDebateProtocol",
    "DeterministicOnlyProtocol",
    "HierarchicalProtocol",
    "IndependentEnsembleProtocol",
    "LLMPlusAnchorProtocol",
    "REGISTRY",
    "SequentialPipelineProtocol",
    "SingleAgentProtocol",
]

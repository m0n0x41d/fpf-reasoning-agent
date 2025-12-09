"""
FPF Reasoning Module

SGR Cascade approach: One LLM, different Pydantic schemas per phase.
Python orchestration controls phase order and gate enforcement.
"""

from .schemas.abduction import (
    AnomalyAnalysis,
    HypothesisCandidate,
    AbductionOutput,
    ProblemType,
    ABDUCTION_SYSTEM_PROMPT,
)
from .schemas.deduction import (
    LogicalPremise,
    DerivationStep,
    TestablePrediction,
    DeductionOutput,
    PremiseSource,
    InferenceType,
    DEDUCTION_SYSTEM_PROMPT,
)
from .schemas.induction import (
    EvidenceItem,
    PredictionTest,
    InductionOutput,
    EvidenceSourceType,
    PredictionMatch,
    InductionVerdict,
    INDUCTION_SYSTEM_PROMPT,
)
from .schemas.synthesis import (
    SynthesisOutput,
    SYNTHESIS_SYSTEM_PROMPT,
)
from .lifecycle import (
    LifecycleManager,
    LifecycleState,
    TransitionRule,
    check_can_transition,
    get_next_state,
)
from .adi_executor import (
    ADIExecutor,
    ExecutionContext,
    PhaseResult,
    CycleResult,
    run_adi_cycle,
)

__all__ = [
    # Abduction
    "AnomalyAnalysis",
    "HypothesisCandidate",
    "AbductionOutput",
    "ProblemType",
    "ABDUCTION_SYSTEM_PROMPT",
    # Deduction
    "LogicalPremise",
    "DerivationStep",
    "TestablePrediction",
    "DeductionOutput",
    "PremiseSource",
    "InferenceType",
    "DEDUCTION_SYSTEM_PROMPT",
    # Induction
    "EvidenceItem",
    "PredictionTest",
    "InductionOutput",
    "EvidenceSourceType",
    "PredictionMatch",
    "InductionVerdict",
    "INDUCTION_SYSTEM_PROMPT",
    # Synthesis
    "SynthesisOutput",
    "SYNTHESIS_SYSTEM_PROMPT",
    # Lifecycle
    "LifecycleManager",
    "LifecycleState",
    "TransitionRule",
    "check_can_transition",
    "get_next_state",
    # Executor
    "ADIExecutor",
    "ExecutionContext",
    "PhaseResult",
    "CycleResult",
    "run_adi_cycle",
]

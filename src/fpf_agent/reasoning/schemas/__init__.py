"""
SGR Schemas for ADI Reasoning Phases (FPF B.5)

Each schema forces the LLM through structured reasoning steps.
SGR Cascade: Same LLM, different Pydantic schemas per phase.
"""

from .abduction import (
    AnomalyAnalysis,
    HypothesisCandidate,
    AbductionOutput,
    ProblemType,
)
from .deduction import (
    LogicalPremise,
    DerivationStep,
    TestablePrediction,
    DeductionOutput,
    PremiseSource,
    InferenceType,
    PredictionCriticality,
)
from .induction import (
    EvidenceItem,
    PredictionTest,
    InductionOutput,
    EvidenceSourceType,
    PredictionMatch,
    InductionVerdict,
)
from .synthesis import SynthesisOutput

__all__ = [
    # Abduction
    "AnomalyAnalysis",
    "HypothesisCandidate",
    "AbductionOutput",
    "ProblemType",
    # Deduction
    "LogicalPremise",
    "DerivationStep",
    "TestablePrediction",
    "DeductionOutput",
    "PremiseSource",
    "InferenceType",
    "PredictionCriticality",
    # Induction
    "EvidenceItem",
    "PredictionTest",
    "InductionOutput",
    "EvidenceSourceType",
    "PredictionMatch",
    "InductionVerdict",
    # Synthesis
    "SynthesisOutput",
]

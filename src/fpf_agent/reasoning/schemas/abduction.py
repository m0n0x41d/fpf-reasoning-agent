"""
SGR Schemas for Abduction Phase (FPF B.5.2)

Abduction: Generate plausible hypotheses that could EXPLAIN the given problem.

The schema forces the LLM through:
1. Anomaly analysis - understand what needs explaining
2. Candidate generation - produce 2-5 competing hypotheses
3. Selection - choose best with explicit rationale
4. Documentation - record alternatives for future reference
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class ProblemType(str, Enum):
    """Classification of the problem/anomaly to be explained."""

    EXPLANATORY_GAP = "explanatory_gap"
    """We don't know WHY something is the case."""

    PREDICTION_FAILURE = "prediction_failure"
    """Expected X, got Y - prediction didn't match observation."""

    INCONSISTENCY = "inconsistency"
    """A contradicts B - need to resolve conflict."""

    NOVEL_PHENOMENON = "novel_phenomenon"
    """Something new observed that existing knowledge doesn't cover."""

    OPTIMIZATION_NEED = "optimization_need"
    """Current solution works but could be better."""

    DESIGN_QUESTION = "design_question"
    """Need to choose between alternatives."""


class AnomalyAnalysis(BaseModel):
    """
    Step 1: Understand what needs explaining.

    Forces explicit articulation of WHY existing knowledge is insufficient.
    """

    problem_type: ProblemType = Field(
        description="Classification of the problem type"
    )

    what_needs_explaining: str = Field(
        min_length=10,
        description="The specific phenomenon or gap that needs a hypothesis. "
        "Be precise about WHAT is unexplained.",
    )

    why_existing_insufficient: str = Field(
        min_length=10,
        description="Why current knowledge cannot explain this. "
        "What makes existing explanations inadequate?",
    )

    key_observations: list[str] = Field(
        default_factory=list,
        min_length=1,
        description="Specific observations or facts that any valid hypothesis must account for.",
    )

    constraints: list[str] = Field(
        default_factory=list,
        description="Constraints any valid hypothesis must satisfy "
        "(e.g., must be consistent with X, must not violate Y).",
    )

    scope_boundaries: str = Field(
        default="",
        description="What is explicitly OUT of scope for this hypothesis. "
        "Helps prevent over-generalization.",
    )


class HypothesisCandidate(BaseModel):
    """
    A single hypothesis candidate.

    Each hypothesis must be:
    - Falsifiable (can be tested)
    - Mechanistic (explains HOW, not just WHAT)
    - Explicit about assumptions
    """

    statement: str = Field(
        min_length=10,
        description="Clear, falsifiable statement of the hypothesis. "
        "Should be specific enough to derive testable predictions.",
    )

    mechanism: str = Field(
        min_length=10,
        description="HOW this hypothesis explains the phenomenon. "
        "What is the causal or logical chain?",
    )

    testable_predictions: list[str] = Field(
        min_length=1,
        description="At least one prediction that could be tested. "
        "If hypothesis is true, what observable consequences follow?",
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description="Key assumptions this hypothesis relies on. "
        "What must be true for this hypothesis to hold?",
    )

    potential_issues: list[str] = Field(
        default_factory=list,
        description="Known weaknesses, edge cases, or concerns with this hypothesis.",
    )

    distinguishing_features: str = Field(
        default="",
        description="What makes this hypothesis different from alternatives? "
        "How would we distinguish it from competing explanations?",
    )


class AbductionOutput(BaseModel):
    """
    Complete output of abduction phase.

    SGR Cascade: Forces LLM through structured reasoning steps.

    FPF B.5.2 Compliance:
    - Must produce multiple competing hypotheses (not variations of one idea)
    - Must document alternatives (they may be useful later)
    - Must have testable predictions to pass gate to Deduction
    """

    # Step 1: Problem analysis
    anomaly: AnomalyAnalysis = Field(
        description="Analysis of the problem/anomaly to be explained"
    )

    # Step 2: Generate candidates (2-5)
    candidates: Annotated[
        list[HypothesisCandidate],
        Field(
            min_length=2,
            max_length=5,
            description="2-5 competing hypotheses. These should be genuinely different "
            "explanations, not minor variations of the same idea.",
        ),
    ]

    # Step 3: Selection
    selection_criteria: list[str] = Field(
        min_length=1,
        description="Criteria used to rank hypotheses "
        "(e.g., simplicity, explanatory power, testability, consistency).",
    )

    best_index: Annotated[
        int,
        Field(
            ge=0,
            description="Index of selected best hypothesis (0-indexed into candidates list).",
        ),
    ]

    selection_rationale: str = Field(
        min_length=10,
        description="Why this hypothesis was selected over others. "
        "Must reference the selection criteria.",
    )

    # Step 4: Document alternatives
    alternatives_summary: str = Field(
        min_length=10,
        description="Brief summary of rejected alternatives and why they were less preferred. "
        "These may be revisited if the selected hypothesis fails.",
    )

    # Metadata for FPF compliance
    plausibility_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Estimated plausibility of the best hypothesis [0-1]. "
            "This is a rough estimate based on fit to evidence and mechanism clarity.",
        ),
    ]

    confidence_in_selection: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Confidence that the selected hypothesis is better than alternatives [0-1]. "
            "Low confidence suggests we should test multiple hypotheses.",
        ),
    ]

    @model_validator(mode="after")
    def validate_best_index_bounds(self) -> "AbductionOutput":
        """Ensure best_index is within candidates bounds."""
        if self.best_index >= len(self.candidates):
            raise ValueError(
                f"best_index ({self.best_index}) must be < candidates length ({len(self.candidates)})"
            )
        return self

    @property
    def best_hypothesis(self) -> HypothesisCandidate:
        """Get the selected best hypothesis."""
        return self.candidates[self.best_index]

    @property
    def alternative_hypotheses(self) -> list[HypothesisCandidate]:
        """Get the non-selected hypotheses."""
        return [h for i, h in enumerate(self.candidates) if i != self.best_index]

    def to_hypothesis_dict(self) -> dict:
        """
        Convert to format expected by ADICycleController.

        Maps SGR schema output to the simpler Hypothesis dataclass.
        """
        best = self.best_hypothesis
        return {
            "hypothesis_statement": best.statement,
            "anomaly": self.anomaly.what_needs_explaining,
            "plausibility_score": self.plausibility_score,
            "plausibility_rationale": self.selection_rationale,
            "testable_predictions": best.testable_predictions,
            "competing_hypotheses": [h.statement for h in self.alternative_hypotheses],
        }


# =============================================================================
# SYSTEM PROMPT FOR ABDUCTION
# =============================================================================

ABDUCTION_SYSTEM_PROMPT = """You are a research reasoning engine performing ABDUCTION (FPF B.5.2).

Your task: Generate plausible hypotheses that could EXPLAIN the given problem.

## Rules

1. **Analyze the anomaly first**
   - What exactly needs explaining?
   - Why is existing knowledge insufficient?
   - What constraints must any explanation satisfy?

2. **Generate 2-5 COMPETING hypotheses**
   - These must be genuinely different explanations
   - Not minor variations of the same idea
   - Each must offer a distinct mechanism

3. **Each hypothesis MUST be falsifiable**
   - Can be tested empirically or logically
   - Has at least one prediction that could be wrong
   - Avoid unfalsifiable claims ("it just is", "by definition")

4. **Rank by PLAUSIBILITY, not preference**
   - Consider: simplicity, explanatory power, testability
   - Consider: consistency with known facts
   - Document your ranking criteria explicitly

5. **Document rejected alternatives**
   - They may become relevant if the best hypothesis fails
   - Record WHY they were less preferred

## Output Format

You must output a structured JSON object matching the AbductionOutput schema.
Do not include any text outside the JSON object.
"""

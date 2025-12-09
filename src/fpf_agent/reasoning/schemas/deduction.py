"""
SGR Schemas for Deduction Phase (FPF B.5, CC-B5.2)

Deduction: Derive LOGICAL CONSEQUENCES from the hypothesis.

The schema forces the LLM through:
1. Premise extraction - list all premises explicitly
2. Derivation chain - step-by-step logical reasoning
3. Prediction identification - what can be tested
4. Consistency check - no internal contradictions
5. Formality assessment - how rigorous is this derivation

FPF CC-B5.2: Deduction MUST complete before any testing (induction).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class PremiseSource(str, Enum):
    """Source type for a logical premise."""

    HYPOTHESIS = "hypothesis"
    """The hypothesis being tested (from abduction)."""

    BACKGROUND_KNOWLEDGE = "background_knowledge"
    """Generally accepted facts or domain knowledge."""

    DEFINITION = "definition"
    """Definitional truth (by meaning of terms)."""

    AXIOM = "axiom"
    """Assumed without proof for this context."""

    DERIVED = "derived"
    """Derived from earlier steps in this derivation."""

    OBSERVATION = "observation"
    """Direct observation or measurement."""


class InferenceType(str, Enum):
    """Type of logical inference used in a derivation step."""

    MODUS_PONENS = "modus_ponens"
    """If P then Q; P; therefore Q."""

    MODUS_TOLLENS = "modus_tollens"
    """If P then Q; not Q; therefore not P."""

    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"
    """If P then Q; if Q then R; therefore if P then R."""

    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"
    """P or Q; not P; therefore Q."""

    INSTANTIATION = "instantiation"
    """For all X, P(X); therefore P(a) for specific a."""

    GENERALIZATION = "generalization"
    """P(a) for arbitrary a; therefore for all X, P(X)."""

    ANALOGY = "analogy"
    """A is like B in relevant ways; B has property P; therefore A likely has P."""

    DEFINITION_EXPANSION = "definition_expansion"
    """Replace term with its definition."""

    CAUSAL_REASONING = "causal_reasoning"
    """X causes Y; X occurred; therefore Y (probably) occurred."""

    MATHEMATICAL = "mathematical"
    """Mathematical derivation or calculation."""

    CONJUNCTION = "conjunction"
    """P; Q; therefore P and Q."""

    SIMPLIFICATION = "simplification"
    """P and Q; therefore P."""


class PredictionCriticality(str, Enum):
    """How critical a prediction is to the hypothesis."""

    CRITICAL = "critical"
    """Refutation kills the hypothesis."""

    SUPPORTING = "supporting"
    """Refutation weakens but doesn't kill."""

    AUXILIARY = "auxiliary"
    """Nice to test but not essential."""


class LogicalPremise(BaseModel):
    """A premise used in deduction."""

    premise_id: str = Field(
        description="Unique identifier for this premise (e.g., 'P1', 'P2')."
    )

    statement: str = Field(
        min_length=5,
        description="The premise statement in clear language.",
    )

    source: PremiseSource = Field(
        description="Where this premise comes from."
    )

    confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Confidence in this premise [0-1]. "
            "Hypothesis premises inherit their plausibility score.",
        ),
    ]

    justification: str = Field(
        default="",
        description="Why this premise is accepted. "
        "Required for non-obvious premises.",
    )


class DerivationStep(BaseModel):
    """One step in logical derivation."""

    step_number: int = Field(
        ge=1,
        description="Step number in the derivation (1-indexed)."
    )

    from_premises: list[str] = Field(
        min_length=1,
        description="IDs of premises/steps used (e.g., ['P1', 'P2'] or ['S1', 'P3'])."
    )

    inference_type: InferenceType = Field(
        description="Type of logical inference applied."
    )

    conclusion: str = Field(
        min_length=5,
        description="What is derived in this step."
    )

    explanation: str = Field(
        min_length=10,
        description="Natural language explanation of why this follows."
    )

    confidence_propagation: str = Field(
        default="",
        description="How confidence is affected by this step. "
        "E.g., 'min of inputs' for conjunction, 'degraded' for analogy.",
    )

    @property
    def step_id(self) -> str:
        """Get the ID for this step (for reference in later steps)."""
        return f"S{self.step_number}"


class TestablePrediction(BaseModel):
    """A prediction derived from the hypothesis that can be tested."""

    prediction_id: str = Field(
        description="Unique identifier for this prediction (e.g., 'T1', 'T2')."
    )

    prediction: str = Field(
        min_length=10,
        description="What we predict will happen or be true."
    )

    conditions: list[str] = Field(
        default_factory=list,
        description="Under what conditions this prediction applies."
    )

    observable: str = Field(
        min_length=5,
        description="What to observe or measure to test this prediction."
    )

    expected_outcome: str = Field(
        min_length=5,
        description="Expected value, state, or observation if hypothesis is correct."
    )

    falsification_criterion: str = Field(
        min_length=10,
        description="What would clearly REFUTE this prediction. "
        "Must be specific enough to be unambiguous.",
    )

    derived_from_steps: list[str] = Field(
        default_factory=list,
        description="Which derivation steps lead to this prediction (e.g., ['S3', 'S5']).",
    )

    criticality: PredictionCriticality = Field(
        default=PredictionCriticality.SUPPORTING,
        description="How critical is this test? "
        "'critical' = refutation kills hypothesis, "
        "'supporting' = refutation weakens but doesn't kill, "
        "'auxiliary' = nice to test but not essential.",
    )


class DeductionOutput(BaseModel):
    """
    Complete output of deduction phase.

    FPF CC-B5.2: MUST happen before induction.

    This schema ensures:
    - All premises are explicit (no hidden assumptions)
    - Derivation is traceable step-by-step
    - Predictions are falsifiable
    - Internal consistency is checked
    """

    # Reference to input
    hypothesis_statement: str = Field(
        description="The hypothesis being analyzed (from abduction)."
    )

    # Step 1: Extract premises
    premises: Annotated[
        list[LogicalPremise],
        Field(
            min_length=1,
            description="All premises including the hypothesis itself. "
            "Make hidden assumptions explicit.",
        ),
    ]

    # Step 2: Derive consequences
    derivation_chain: list[DerivationStep] = Field(
        default_factory=list,
        description="Step-by-step logical derivation. "
        "Each step references earlier premises/steps.",
    )

    # Step 3: Identify predictions
    predictions: Annotated[
        list[TestablePrediction],
        Field(
            min_length=1,
            description="Testable predictions derived from the hypothesis. "
            "At least one must be critical.",
        ),
    ]

    # Step 4: Check consistency
    internal_consistency: bool = Field(
        description="Are there any internal contradictions in the derivation?"
    )

    consistency_notes: str = Field(
        default="",
        description="Notes on any consistency concerns or caveats. "
        "Required if internal_consistency is False.",
    )

    # Step 5: Formality assessment
    formality_achieved: Annotated[
        int,
        Field(
            ge=0,
            le=9,
            description="Formality level of this derivation (F0-F9). "
            "F0=informal, F3=structured, F6=calculable, F9=certified.",
        ),
    ]

    formality_rationale: str = Field(
        default="",
        description="Why this formality level was assessed."
    )

    # Confidence propagation
    overall_confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Overall confidence in the derivation [0-1]. "
            "Bounded by min premise confidence and degraded by weak inferences.",
        ),
    ]

    @model_validator(mode="after")
    def validate_consistency_notes(self) -> "DeductionOutput":
        """Ensure consistency notes are provided if inconsistent."""
        if not self.internal_consistency and not self.consistency_notes:
            raise ValueError(
                "consistency_notes required when internal_consistency is False"
            )
        return self

    @property
    def critical_predictions(self) -> list[TestablePrediction]:
        """Get predictions marked as critical."""
        return [p for p in self.predictions if p.criticality == PredictionCriticality.CRITICAL]

    @property
    def has_critical_prediction(self) -> bool:
        """Check if there's at least one critical prediction."""
        return len(self.critical_predictions) > 0

    def to_deduction_dict(self) -> dict:
        """
        Convert to format expected by ADICycleController.

        Maps SGR schema output to the simpler DeductionResult dataclass.
        """
        return {
            "hypothesis_id": "",  # Filled by caller
            "derived_consequences": [s.conclusion for s in self.derivation_chain],
            "predictions_to_test": [p.prediction for p in self.predictions],
            "logical_consistency": self.internal_consistency,
            "consistency_rationale": self.consistency_notes,
        }


# =============================================================================
# SYSTEM PROMPT FOR DEDUCTION
# =============================================================================

DEDUCTION_SYSTEM_PROMPT = """You are a research reasoning engine performing DEDUCTION (FPF B.5).

Your task: Derive LOGICAL CONSEQUENCES from the hypothesis.

## Rules

1. **List all premises explicitly**
   - Include the hypothesis itself as a premise
   - Make hidden assumptions explicit
   - Assign confidence to each premise

2. **Show step-by-step derivation**
   - Each step must reference earlier premises/steps
   - Name the inference type used
   - Explain WHY the conclusion follows

3. **Every prediction must be TESTABLE**
   - Specify what to observe
   - Define expected outcome
   - Define what would FALSIFY the prediction

4. **Check for internal contradictions**
   - Do any derived consequences conflict?
   - Are premises mutually consistent?
   - Flag any concerns

5. **Assess formality level**
   - F0-F2: Informal/narrative reasoning
   - F3-F5: Structured/typed reasoning
   - F6-F9: Calculable/verified reasoning

## Critical Requirement

This phase MUST complete before any testing (CC-B5.2).
Do NOT evaluate whether predictions are true yet - that's Induction's job.

## Output Format

You must output a structured JSON object matching the DeductionOutput schema.
Do not include any text outside the JSON object.
"""

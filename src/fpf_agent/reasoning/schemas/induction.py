"""
SGR Schemas for Induction Phase (FPF B.5, CC-B5.3)

Induction: TEST predictions against evidence and determine verdict.

The schema forces the LLM through:
1. Evidence gathering - for each prediction
2. Comparison - observed vs expected
3. Assessment - match/mismatch analysis
4. Verdict - overall hypothesis status
5. Next steps - refinement or new cycle

FPF CC-B5.3: Assurance level > L0 requires linked evidence.
FPF CC-B5.4: Results feed back into next cycle.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class EvidenceSourceType(str, Enum):
    """Type of evidence source."""

    EXPERIMENTAL = "experimental"
    """Controlled experiment or test."""

    OBSERVATIONAL = "observational"
    """Direct observation without intervention."""

    SIMULATION = "simulation"
    """Computational model or simulation."""

    LITERATURE = "literature"
    """Published research or documentation."""

    EXPERT_OPINION = "expert_opinion"
    """Expert judgment or consensus."""

    HISTORICAL = "historical"
    """Historical records or precedent."""

    LOGICAL = "logical"
    """Logical argument or proof."""

    STATISTICAL = "statistical"
    """Statistical analysis of data."""


class PredictionMatch(str, Enum):
    """Result of comparing prediction to evidence."""

    MATCH = "match"
    """Evidence clearly supports the prediction."""

    PARTIAL = "partial"
    """Evidence partially supports, with caveats."""

    MISMATCH = "mismatch"
    """Evidence clearly contradicts the prediction."""

    INCONCLUSIVE = "inconclusive"
    """Evidence is insufficient to determine match."""


class InductionVerdict(str, Enum):
    """Overall verdict on the hypothesis after testing."""

    CORROBORATED = "corroborated"
    """Hypothesis passed tests - confidence increased."""

    REFUTED = "refuted"
    """Critical prediction failed - hypothesis rejected."""

    WEAKENED = "weakened"
    """Some predictions failed - hypothesis needs refinement."""

    INCONCLUSIVE = "inconclusive"
    """Insufficient evidence to decide."""


class EvidenceItem(BaseModel):
    """A piece of evidence for/against a prediction."""

    evidence_id: str = Field(
        description="Unique identifier for this evidence item."
    )

    description: str = Field(
        min_length=10,
        description="Description of the evidence."
    )

    source: str = Field(
        min_length=3,
        description="Where this evidence comes from (citation, experiment, etc.)."
    )

    source_type: EvidenceSourceType = Field(
        description="Type of evidence source."
    )

    reliability: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Reliability of this evidence [0-1]. "
            "Consider source quality, methodology, recency.",
        ),
    ]

    relevance: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="How directly relevant is this evidence to the prediction [0-1].",
        ),
    ]

    timestamp: str = Field(
        default="",
        description="When this evidence was gathered/published (for temporal validity)."
    )


class PredictionTest(BaseModel):
    """Test of one prediction against evidence."""

    prediction_id: str = Field(
        description="ID of the prediction being tested (from deduction)."
    )

    prediction_text: str = Field(
        description="The prediction statement being tested."
    )

    # Evidence gathered
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items gathered for this prediction."
    )

    # Comparison
    observed_outcome: str = Field(
        min_length=5,
        description="What was actually observed or found."
    )

    expected_outcome: str = Field(
        description="What was expected (from deduction)."
    )

    # Assessment
    match_result: PredictionMatch = Field(
        description="Does evidence match the prediction?"
    )

    assessment: str = Field(
        min_length=10,
        description="Analysis of match/mismatch. Why does evidence support or refute?"
    )

    confidence_in_assessment: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Confidence in this assessment [0-1]. "
            "Low if evidence is sparse or ambiguous.",
        ),
    ]

    # For critical predictions
    is_critical: bool = Field(
        default=False,
        description="Was this a critical prediction? Mismatch = hypothesis failure."
    )

    @property
    def weighted_support(self) -> float:
        """
        Compute weighted support from evidence.

        Positive for match, negative for mismatch.
        """
        if not self.evidence:
            return 0.0

        total_weight = sum(e.reliability * e.relevance for e in self.evidence)
        if total_weight == 0:
            return 0.0

        if self.match_result == PredictionMatch.MATCH:
            return self.confidence_in_assessment * total_weight
        elif self.match_result == PredictionMatch.MISMATCH:
            return -self.confidence_in_assessment * total_weight
        elif self.match_result == PredictionMatch.PARTIAL:
            return 0.5 * self.confidence_in_assessment * total_weight
        else:
            return 0.0


class InductionOutput(BaseModel):
    """
    Complete output of induction phase.

    FPF CC-B5.3: L1+ assurance only after linked test.
    FPF CC-B5.4: Results feed into next cycle.

    This schema ensures:
    - Each prediction from deduction is addressed
    - Evidence is cited specifically
    - Assessment is transparent
    - Next steps are clear
    """

    # Reference to input
    hypothesis_id: str = Field(
        description="ID of the hypothesis being tested."
    )

    hypothesis_statement: str = Field(
        description="The hypothesis statement (for reference)."
    )

    # Step 1: Test each prediction
    prediction_tests: Annotated[
        list[PredictionTest],
        Field(
            min_length=1,
            description="Test results for each prediction from deduction.",
        ),
    ]

    # Step 2: Overall verdict
    overall_verdict: InductionVerdict = Field(
        description="Overall verdict on the hypothesis."
    )

    verdict_rationale: str = Field(
        min_length=10,
        description="Why this verdict was reached. Reference specific test results."
    )

    # Step 3: Reliability contribution
    reliability_delta: Annotated[
        float,
        Field(
            ge=-1.0,
            le=1.0,
            description="Change in reliability. "
            "Positive = evidence supports, negative = evidence refutes.",
        ),
    ]

    # Step 4: Confidence and assurance
    overall_confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Confidence in the verdict [0-1].",
        ),
    ]

    recommended_assurance: Annotated[
        str,
        Field(
            description="Recommended assurance level after this test. "
            "'L0' = untested, 'L1' = tested, 'L2' = verified.",
        ),
    ] = "L0"

    # Step 5: Lessons for next cycle (CC-B5.4)
    if_refuted_new_anomaly: str = Field(
        default="",
        description="If refuted: new anomaly to explain in next abduction cycle. "
        "What did we learn from the failure?",
    )

    refinement_suggestions: list[str] = Field(
        default_factory=list,
        description="How could the hypothesis be refined based on test results?"
    )

    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions that remain unanswered."
    )

    @model_validator(mode="after")
    def validate_refutation_anomaly(self) -> "InductionOutput":
        """Ensure new anomaly is provided if refuted."""
        if self.overall_verdict == InductionVerdict.REFUTED:
            if not self.if_refuted_new_anomaly:
                raise ValueError(
                    "if_refuted_new_anomaly required when verdict is REFUTED"
                )
        return self

    @property
    def should_loop_back(self) -> bool:
        """Check if we need to loop back to abduction."""
        return self.overall_verdict in (
            InductionVerdict.REFUTED,
            InductionVerdict.INCONCLUSIVE,
        )

    @property
    def critical_failures(self) -> list[PredictionTest]:
        """Get critical predictions that failed."""
        return [
            t for t in self.prediction_tests
            if t.is_critical and t.match_result == PredictionMatch.MISMATCH
        ]

    @property
    def confirmation_rate(self) -> float:
        """Rate of confirmed predictions."""
        if not self.prediction_tests:
            return 0.0
        matched = sum(
            1 for t in self.prediction_tests
            if t.match_result in (PredictionMatch.MATCH, PredictionMatch.PARTIAL)
        )
        return matched / len(self.prediction_tests)

    def to_induction_dict(self) -> dict:
        """
        Convert to format expected by ADICycleController.

        Maps SGR schema output to the simpler InductionResult dataclass.
        """
        predictions_confirmed = [
            t.prediction_text for t in self.prediction_tests
            if t.match_result == PredictionMatch.MATCH
        ]
        predictions_refuted = [
            t.prediction_text for t in self.prediction_tests
            if t.match_result == PredictionMatch.MISMATCH
        ]

        # Map verdict
        verdict_map = {
            InductionVerdict.CORROBORATED: "confirmed",
            InductionVerdict.REFUTED: "refuted",
            InductionVerdict.WEAKENED: "partial",
            InductionVerdict.INCONCLUSIVE: "inconclusive",
        }

        return {
            "hypothesis_id": self.hypothesis_id,
            "predictions_tested": [t.prediction_text for t in self.prediction_tests],
            "predictions_confirmed": predictions_confirmed,
            "predictions_refuted": predictions_refuted,
            "overall_verdict": verdict_map.get(self.overall_verdict, "inconclusive"),
            "confidence": self.overall_confidence,
            "evidence_summary": self.verdict_rationale,
        }


# =============================================================================
# SYSTEM PROMPT FOR INDUCTION
# =============================================================================

INDUCTION_SYSTEM_PROMPT = """You are a research reasoning engine performing INDUCTION (FPF B.5).

Your task: TEST predictions against evidence and determine verdict.

## Rules

1. **Address each prediction from deduction**
   - Don't skip any predictions
   - Gather specific evidence for each
   - Be explicit about what you observed

2. **Cite specific evidence**
   - Reference sources by name/citation
   - Rate reliability of each source
   - Note relevance to the specific prediction

3. **Be honest about match/mismatch**
   - Match: Evidence clearly supports
   - Partial: Evidence partially supports with caveats
   - Mismatch: Evidence clearly contradicts
   - Inconclusive: Insufficient evidence

4. **If refuted: Identify new anomaly for next cycle**
   - What did we learn from the failure?
   - What should the next hypothesis explain?
   - This feeds back into abduction (CC-B5.4)

5. **Recommend appropriate assurance level**
   - L0: No testing yet
   - L1: Tested with linked evidence
   - L2: Verified/validated

## FPF Compliance

- CC-B5.3: Assurance level > L0 requires linked evidence
- CC-B5.4: Refutation results feed into next cycle

## Output Format

You must output a structured JSON object matching the InductionOutput schema.
Do not include any text outside the JSON object.
"""

"""
ADI Reasoning Cycle Controller (B.5)

Implements the canonical Abduction-Deduction-Induction reasoning cycle
with gates between phases.

FPF says (B.5):
    ABDUCTION → (gate) → DEDUCTION → (gate) → INDUCTION
        ↑                                          |
        └──────────── (refinement loop) ───────────┘

Key mechanics:
- Abduction MUST produce Hypothesis with testable_predictions
- Deduction MUST derive consequences that can be falsified
- Induction MUST check predictions and decide: confirm/refute/refine
- Artifact lifecycle: Exploration → Shaping → Evidence → Operation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable


# =============================================================================
# REASONING PHASES
# =============================================================================

class ReasoningPhase(Enum):
    """B.5: Phases in the ADI reasoning cycle."""
    ABDUCTION = auto()   # Generate hypotheses to explain anomaly
    DEDUCTION = auto()   # Derive testable consequences
    INDUCTION = auto()   # Test predictions against evidence


class ArtifactLifecycle(Enum):
    """B.5.1: Artifact maturity states."""
    EXPLORATION = auto()  # Initial exploration, high uncertainty
    SHAPING = auto()      # Refining, narrowing options
    EVIDENCE = auto()     # Gathering evidence, validating
    OPERATION = auto()    # Validated, ready for use


class GateDecision(Enum):
    """Result of checking a gate between phases."""
    PASS = auto()         # Gate requirements met, can proceed
    BLOCKED = auto()      # Requirements not met, cannot proceed
    LOOP_BACK = auto()    # Need to return to earlier phase


# =============================================================================
# HYPOTHESIS TRACKING
# =============================================================================

@dataclass
class Hypothesis:
    """
    B.5.2: A hypothesis generated during abduction.

    Must have testable_predictions to pass to deduction.
    """
    hypothesis_id: str
    statement: str
    anomaly_addressed: str
    plausibility_score: float  # [0, 1]
    plausibility_rationale: str
    testable_predictions: list[str]
    competing_hypotheses: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def is_valid_for_deduction(self) -> tuple[bool, str]:
        """Check if hypothesis meets gate requirements for deduction."""
        if not self.statement:
            return False, "Hypothesis statement is empty"
        if not self.testable_predictions:
            return False, "No testable predictions - cannot proceed to deduction"
        if self.plausibility_score < 0.1:
            return False, f"Plausibility too low ({self.plausibility_score}) - generate better hypothesis"
        return True, "OK"


@dataclass
class DeductionResult:
    """Results from the deduction phase."""
    hypothesis_id: str
    derived_consequences: list[str]
    predictions_to_test: list[str] = field(default_factory=list)
    logical_consistency: bool = True
    consistency_rationale: str = ""

    def __post_init__(self):
        """Default predictions_to_test from derived_consequences if empty."""
        if not self.predictions_to_test and self.derived_consequences:
            self.predictions_to_test = list(self.derived_consequences)

    def is_valid_for_induction(self) -> tuple[bool, str]:
        """Check if deduction results meet gate requirements for induction."""
        if not self.predictions_to_test:
            return False, "No predictions to test - cannot proceed to induction"
        if not self.logical_consistency:
            return False, f"Logical inconsistency: {self.consistency_rationale}"
        return True, "OK"


@dataclass
class InductionResult:
    """Results from the induction phase."""
    hypothesis_id: str
    predictions_tested: list[str] = field(default_factory=list)
    predictions_confirmed: list[str] = field(default_factory=list)
    predictions_refuted: list[str] = field(default_factory=list)
    overall_verdict: str = ""  # "confirmed", "refuted", "partial", "inconclusive"
    confidence: float = 0.0
    evidence_summary: str = ""

    def __post_init__(self):
        """Auto-compute verdict if not provided."""
        if not self.overall_verdict:
            if self.predictions_refuted and not self.predictions_confirmed:
                self.overall_verdict = "refuted"
            elif self.predictions_confirmed and not self.predictions_refuted:
                self.overall_verdict = "confirmed"
            elif self.predictions_confirmed and self.predictions_refuted:
                self.overall_verdict = "partial"
            else:
                self.overall_verdict = "inconclusive"

    @property
    def should_loop_back(self) -> bool:
        """Check if we need to loop back to abduction."""
        return self.overall_verdict in ("refuted", "inconclusive")

    @property
    def confirmation_rate(self) -> float:
        """Rate of confirmed predictions."""
        total = len(self.predictions_confirmed) + len(self.predictions_refuted)
        if total == 0:
            return 0.0
        return len(self.predictions_confirmed) / total


# =============================================================================
# PHASE GATES
# =============================================================================

@dataclass
class GateCheckResult:
    """Result of a gate check between phases."""
    decision: GateDecision
    current_phase: ReasoningPhase
    target_phase: ReasoningPhase | None
    reason: str
    requirements_met: list[str] = field(default_factory=list)
    requirements_missing: list[str] = field(default_factory=list)


def check_abduction_to_deduction_gate(hypothesis: Hypothesis | None) -> GateCheckResult:
    """
    Gate from Abduction → Deduction.

    Requirements:
    - Valid hypothesis with statement
    - At least one testable prediction
    - Minimum plausibility threshold
    """
    requirements_met = []
    requirements_missing = []

    if hypothesis is None:
        return GateCheckResult(
            decision=GateDecision.BLOCKED,
            current_phase=ReasoningPhase.ABDUCTION,
            target_phase=ReasoningPhase.DEDUCTION,
            reason="No hypothesis generated",
            requirements_missing=["hypothesis"],
        )

    is_valid, reason = hypothesis.is_valid_for_deduction()

    if hypothesis.statement:
        requirements_met.append("hypothesis_statement")
    else:
        requirements_missing.append("hypothesis_statement")

    if hypothesis.testable_predictions:
        requirements_met.append(f"testable_predictions ({len(hypothesis.testable_predictions)})")
    else:
        requirements_missing.append("testable_predictions")

    if hypothesis.plausibility_score >= 0.1:
        requirements_met.append(f"plausibility ({hypothesis.plausibility_score:.2f})")
    else:
        requirements_missing.append(f"plausibility >= 0.1 (got {hypothesis.plausibility_score:.2f})")

    if is_valid:
        return GateCheckResult(
            decision=GateDecision.PASS,
            current_phase=ReasoningPhase.ABDUCTION,
            target_phase=ReasoningPhase.DEDUCTION,
            reason="Hypothesis meets deduction requirements",
            requirements_met=requirements_met,
        )
    else:
        return GateCheckResult(
            decision=GateDecision.BLOCKED,
            current_phase=ReasoningPhase.ABDUCTION,
            target_phase=ReasoningPhase.DEDUCTION,
            reason=reason,
            requirements_met=requirements_met,
            requirements_missing=requirements_missing,
        )


def check_deduction_to_induction_gate(deduction: DeductionResult | None) -> GateCheckResult:
    """
    Gate from Deduction → Induction.

    Requirements:
    - Predictions derived from hypothesis
    - Logical consistency maintained
    """
    requirements_met = []
    requirements_missing = []

    if deduction is None:
        return GateCheckResult(
            decision=GateDecision.BLOCKED,
            current_phase=ReasoningPhase.DEDUCTION,
            target_phase=ReasoningPhase.INDUCTION,
            reason="No deduction results",
            requirements_missing=["deduction_result"],
        )

    is_valid, reason = deduction.is_valid_for_induction()

    if deduction.predictions_to_test:
        requirements_met.append(f"predictions ({len(deduction.predictions_to_test)})")
    else:
        requirements_missing.append("predictions_to_test")

    if deduction.logical_consistency:
        requirements_met.append("logical_consistency")
    else:
        requirements_missing.append("logical_consistency")

    if is_valid:
        return GateCheckResult(
            decision=GateDecision.PASS,
            current_phase=ReasoningPhase.DEDUCTION,
            target_phase=ReasoningPhase.INDUCTION,
            reason="Deduction complete, ready for testing",
            requirements_met=requirements_met,
        )
    else:
        return GateCheckResult(
            decision=GateDecision.BLOCKED,
            current_phase=ReasoningPhase.DEDUCTION,
            target_phase=ReasoningPhase.INDUCTION,
            reason=reason,
            requirements_met=requirements_met,
            requirements_missing=requirements_missing,
        )


def check_induction_completion_gate(induction: InductionResult | None) -> GateCheckResult:
    """
    Gate after Induction: complete or loop back to Abduction.

    Decides whether to:
    - Complete (hypothesis confirmed)
    - Loop back (hypothesis refuted, need new hypothesis)
    """
    if induction is None:
        return GateCheckResult(
            decision=GateDecision.BLOCKED,
            current_phase=ReasoningPhase.INDUCTION,
            target_phase=None,
            reason="No induction results",
            requirements_missing=["induction_result"],
        )

    if induction.should_loop_back:
        return GateCheckResult(
            decision=GateDecision.LOOP_BACK,
            current_phase=ReasoningPhase.INDUCTION,
            target_phase=ReasoningPhase.ABDUCTION,
            reason=f"Hypothesis {induction.overall_verdict} - generating new hypothesis",
            requirements_met=[f"verdict: {induction.overall_verdict}"],
        )
    else:
        return GateCheckResult(
            decision=GateDecision.PASS,
            current_phase=ReasoningPhase.INDUCTION,
            target_phase=None,  # Complete
            reason=f"Hypothesis {induction.overall_verdict} with confidence {induction.confidence:.2f}",
            requirements_met=[
                f"verdict: {induction.overall_verdict}",
                f"confidence: {induction.confidence:.2f}",
                f"confirmation_rate: {induction.confirmation_rate:.2f}",
            ],
        )


# =============================================================================
# ADI CYCLE CONTROLLER
# =============================================================================

@dataclass
class CycleState:
    """Current state of the ADI cycle."""
    current_phase: ReasoningPhase
    artifact_state: ArtifactLifecycle
    cycle_count: int
    hypotheses: list[Hypothesis] = field(default_factory=list)
    current_hypothesis: Hypothesis | None = None
    current_deduction: DeductionResult | None = None
    current_induction: InductionResult | None = None
    phase_history: list[tuple[ReasoningPhase, str]] = field(default_factory=list)


class ADICycleController:
    """
    Controller for the ADI reasoning cycle.

    Manages phase transitions, enforces gates, prevents infinite loops.
    """

    def __init__(self, max_cycles: int = 5):
        self.max_cycles = max_cycles
        self.state = CycleState(
            current_phase=ReasoningPhase.ABDUCTION,
            artifact_state=ArtifactLifecycle.EXPLORATION,
            cycle_count=0,
        )

    def start_cycle(self) -> None:
        """Start a new reasoning cycle from abduction."""
        self.state.cycle_count += 1
        self.state.current_phase = ReasoningPhase.ABDUCTION
        self.state.current_hypothesis = None
        self.state.current_deduction = None
        self.state.current_induction = None
        self._log_phase("Starting new ADI cycle")

    def submit_hypothesis(self, hypothesis: Hypothesis) -> GateCheckResult:
        """
        Submit hypothesis from abduction phase.

        Returns gate check result for transition to deduction.
        """
        if self.state.current_phase != ReasoningPhase.ABDUCTION:
            return GateCheckResult(
                decision=GateDecision.BLOCKED,
                current_phase=self.state.current_phase,
                target_phase=ReasoningPhase.DEDUCTION,
                reason=f"Cannot submit hypothesis in {self.state.current_phase.name} phase",
            )

        self.state.hypotheses.append(hypothesis)
        self.state.current_hypothesis = hypothesis

        gate_result = check_abduction_to_deduction_gate(hypothesis)

        if gate_result.decision == GateDecision.PASS:
            self.state.current_phase = ReasoningPhase.DEDUCTION
            self.state.artifact_state = ArtifactLifecycle.SHAPING
            self._log_phase(f"Hypothesis accepted, moving to Deduction")

        return gate_result

    def submit_deduction(self, deduction: DeductionResult) -> GateCheckResult:
        """
        Submit deduction results.

        Returns gate check result for transition to induction.
        """
        if self.state.current_phase != ReasoningPhase.DEDUCTION:
            return GateCheckResult(
                decision=GateDecision.BLOCKED,
                current_phase=self.state.current_phase,
                target_phase=ReasoningPhase.INDUCTION,
                reason=f"Cannot submit deduction in {self.state.current_phase.name} phase",
            )

        self.state.current_deduction = deduction

        gate_result = check_deduction_to_induction_gate(deduction)

        if gate_result.decision == GateDecision.PASS:
            self.state.current_phase = ReasoningPhase.INDUCTION
            self.state.artifact_state = ArtifactLifecycle.EVIDENCE
            self._log_phase(f"Deduction complete, moving to Induction")

        return gate_result

    def submit_induction(self, induction: InductionResult) -> GateCheckResult:
        """
        Submit induction results.

        Returns gate check indicating completion or loop back.
        """
        if self.state.current_phase != ReasoningPhase.INDUCTION:
            return GateCheckResult(
                decision=GateDecision.BLOCKED,
                current_phase=self.state.current_phase,
                target_phase=None,
                reason=f"Cannot submit induction in {self.state.current_phase.name} phase",
            )

        self.state.current_induction = induction

        gate_result = check_induction_completion_gate(induction)

        if gate_result.decision == GateDecision.LOOP_BACK:
            if self.state.cycle_count >= self.max_cycles:
                return GateCheckResult(
                    decision=GateDecision.BLOCKED,
                    current_phase=ReasoningPhase.INDUCTION,
                    target_phase=None,
                    reason=f"Max cycles ({self.max_cycles}) reached - stopping with best hypothesis",
                    requirements_met=[f"cycles: {self.state.cycle_count}"],
                )

            # Loop back to abduction
            self.state.current_phase = ReasoningPhase.ABDUCTION
            self.state.artifact_state = ArtifactLifecycle.EXPLORATION
            self._log_phase(f"Looping back to Abduction (cycle {self.state.cycle_count})")

        elif gate_result.decision == GateDecision.PASS:
            self.state.artifact_state = ArtifactLifecycle.OPERATION
            self._log_phase(f"Cycle complete - hypothesis {induction.overall_verdict}")

        return gate_result

    def get_current_phase(self) -> ReasoningPhase:
        """Get current reasoning phase."""
        return self.state.current_phase

    def get_artifact_state(self) -> ArtifactLifecycle:
        """Get current artifact lifecycle state."""
        return self.state.artifact_state

    def get_cycle_count(self) -> int:
        """Get number of ADI cycles executed."""
        return self.state.cycle_count

    def is_complete(self) -> bool:
        """Check if reasoning cycle is complete."""
        return self.state.artifact_state == ArtifactLifecycle.OPERATION

    def get_phase_history(self) -> list[tuple[ReasoningPhase, str]]:
        """Get history of phase transitions."""
        return self.state.phase_history

    def get_best_hypothesis(self) -> Hypothesis | None:
        """Get the best (most recent confirmed or highest plausibility) hypothesis."""
        if self.state.current_induction and self.state.current_induction.overall_verdict in ("confirmed", "partial"):
            return self.state.current_hypothesis

        if self.state.hypotheses:
            return max(self.state.hypotheses, key=lambda h: h.plausibility_score)

        return None

    def _log_phase(self, note: str) -> None:
        """Log a phase transition."""
        self.state.phase_history.append((self.state.current_phase, note))


# =============================================================================
# ARTIFACT LIFECYCLE GATES
# =============================================================================

def can_transition_artifact_state(
    current: ArtifactLifecycle,
    target: ArtifactLifecycle,
) -> tuple[bool, str]:
    """
    B.5.1: Check if artifact state transition is allowed.

    You cannot skip states (e.g., Exploration → Operation is blocked).
    """
    # Define allowed transitions
    allowed = {
        ArtifactLifecycle.EXPLORATION: [ArtifactLifecycle.SHAPING],
        ArtifactLifecycle.SHAPING: [ArtifactLifecycle.EVIDENCE, ArtifactLifecycle.EXPLORATION],
        ArtifactLifecycle.EVIDENCE: [ArtifactLifecycle.OPERATION, ArtifactLifecycle.SHAPING],
        ArtifactLifecycle.OPERATION: [ArtifactLifecycle.EVIDENCE],  # Can degrade if evidence becomes stale
    }

    if target in allowed.get(current, []):
        return True, f"Transition {current.name} → {target.name} allowed"
    else:
        return False, f"Cannot transition directly from {current.name} to {target.name}"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_hypothesis_from_dict(data: dict) -> Hypothesis:
    """Create Hypothesis from dictionary (e.g., from LLM output)."""
    return Hypothesis(
        hypothesis_id=data.get("hypothesis_id", f"h_{datetime.now().isoformat()}"),
        statement=data.get("hypothesis_statement", data.get("hypothesis", "")),
        anomaly_addressed=data.get("anomaly", ""),
        plausibility_score=data.get("plausibility_score", 0.5),
        plausibility_rationale=data.get("plausibility_rationale", ""),
        testable_predictions=data.get("testable_predictions", []),
        competing_hypotheses=data.get("competing_hypotheses", []),
    )


def get_phase_guidance(phase: ReasoningPhase) -> str:
    """Get guidance text for what to do in each phase."""
    guidance = {
        ReasoningPhase.ABDUCTION: """
ABDUCTION PHASE: Generate a hypothesis to explain the anomaly/question.

Requirements to pass to Deduction:
1. Clear hypothesis statement
2. At least one testable prediction
3. Plausibility score >= 0.1 with rationale
4. Consider competing hypotheses

Focus on: What could explain this? What's the simplest explanation?
""",
        ReasoningPhase.DEDUCTION: """
DEDUCTION PHASE: Derive logical consequences from the hypothesis.

Requirements to pass to Induction:
1. Derive consequences from hypothesis
2. Maintain logical consistency
3. Identify predictions that can be tested

Focus on: If this hypothesis is true, what follows? What can we test?
""",
        ReasoningPhase.INDUCTION: """
INDUCTION PHASE: Test predictions against evidence.

After testing:
- If confirmed: Cycle complete, hypothesis validated
- If refuted: Loop back to Abduction with new information
- If partial: Consider refinement or new hypothesis

Focus on: Does the evidence support or refute our predictions?
""",
    }
    return guidance.get(phase, "Unknown phase")

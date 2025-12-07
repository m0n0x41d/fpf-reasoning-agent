"""
Tests for ADI Reasoning Cycle (B.5)
"""

import pytest

from fpf_agent.core.adi_cycle import (
    ReasoningPhase,
    ArtifactLifecycle,
    ADICycleController,
    Hypothesis,
    DeductionResult,
    InductionResult,
    GateDecision,
    check_abduction_to_deduction_gate,
    check_deduction_to_induction_gate,
    check_induction_completion_gate,
)


class TestReasoningPhases:
    """Test reasoning phase enum."""

    def test_phases_exist(self):
        """All three phases should exist."""
        assert ReasoningPhase.ABDUCTION
        assert ReasoningPhase.DEDUCTION
        assert ReasoningPhase.INDUCTION


class TestArtifactLifecycle:
    """Test artifact lifecycle states."""

    def test_lifecycle_states_exist(self):
        """All lifecycle states should exist."""
        assert ArtifactLifecycle.EXPLORATION
        assert ArtifactLifecycle.SHAPING
        assert ArtifactLifecycle.EVIDENCE
        assert ArtifactLifecycle.OPERATION


class TestHypothesis:
    """Test hypothesis dataclass."""

    def test_valid_hypothesis(self):
        """Valid hypothesis passes validation."""
        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="System fails due to memory leak",
            anomaly_addressed="High memory usage",
            plausibility_score=0.7,
            plausibility_rationale="Matches observed pattern",
            testable_predictions=["Memory grows linearly over time"],
        )

        is_valid, msg = hyp.is_valid_for_deduction()
        assert is_valid
        assert msg == "OK"

    def test_hypothesis_without_predictions(self):
        """Hypothesis without predictions fails gate."""
        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="Something is wrong",
            anomaly_addressed="System issues",
            plausibility_score=0.5,
            plausibility_rationale="Vague",
            testable_predictions=[],  # Empty!
        )

        is_valid, msg = hyp.is_valid_for_deduction()
        assert not is_valid
        assert "testable prediction" in msg.lower()

    def test_hypothesis_without_statement(self):
        """Hypothesis without statement fails gate."""
        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="",  # Empty!
            anomaly_addressed="Problem",
            plausibility_score=0.5,
            plausibility_rationale="",
            testable_predictions=["Prediction"],
        )

        is_valid, msg = hyp.is_valid_for_deduction()
        assert not is_valid
        assert "empty" in msg.lower()


class TestADICycleController:
    """Test ADI cycle controller."""

    def test_initial_state(self):
        """Controller starts in ABDUCTION phase."""
        controller = ADICycleController()
        assert controller.get_current_phase() == ReasoningPhase.ABDUCTION
        assert controller.get_artifact_state() == ArtifactLifecycle.EXPLORATION

    def test_submit_valid_hypothesis(self):
        """Valid hypothesis transitions to DEDUCTION."""
        controller = ADICycleController()

        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="Memory leak hypothesis",
            anomaly_addressed="High memory",
            plausibility_score=0.8,
            plausibility_rationale="Based on patterns",
            testable_predictions=["Memory grows over time"],
        )

        result = controller.submit_hypothesis(hyp)
        assert result.decision == GateDecision.PASS
        assert controller.get_current_phase() == ReasoningPhase.DEDUCTION

    def test_submit_invalid_hypothesis(self):
        """Invalid hypothesis stays in ABDUCTION."""
        controller = ADICycleController()

        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="Bad hypothesis",
            anomaly_addressed="Problem",
            plausibility_score=0.5,
            plausibility_rationale="Vague",
            testable_predictions=[],  # Missing predictions
        )

        result = controller.submit_hypothesis(hyp)
        assert result.decision == GateDecision.BLOCKED
        assert controller.get_current_phase() == ReasoningPhase.ABDUCTION

    def test_submit_deduction(self):
        """Valid deduction transitions to INDUCTION."""
        controller = ADICycleController()

        # First submit valid hypothesis
        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="Hypothesis",
            anomaly_addressed="Anomaly",
            plausibility_score=0.8,
            plausibility_rationale="Rationale",
            testable_predictions=["Prediction 1"],
        )
        controller.submit_hypothesis(hyp)

        # Now submit deduction
        deduction = DeductionResult(
            hypothesis_id="h1",
            derived_consequences=["Consequence 1"],
        )

        result = controller.submit_deduction(deduction)
        assert result.decision == GateDecision.PASS
        assert controller.get_current_phase() == ReasoningPhase.INDUCTION


class TestGateChecks:
    """Test individual gate check functions."""

    def test_abduction_to_deduction_gate_pass(self):
        """Valid hypothesis passes A→D gate."""
        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="Valid hypothesis",
            anomaly_addressed="Problem",
            plausibility_score=0.7,
            plausibility_rationale="Good reasoning",
            testable_predictions=["Prediction"],
        )

        result = check_abduction_to_deduction_gate(hyp)
        assert result.decision == GateDecision.PASS

    def test_abduction_to_deduction_gate_fail(self):
        """Invalid hypothesis fails A→D gate."""
        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="",  # Empty statement
            anomaly_addressed="",
            plausibility_score=0.1,
            plausibility_rationale="",
            testable_predictions=[],
        )

        result = check_abduction_to_deduction_gate(hyp)
        assert result.decision == GateDecision.BLOCKED

    def test_deduction_to_induction_gate_pass(self):
        """Valid deduction passes D→I gate."""
        ded = DeductionResult(
            hypothesis_id="h1",
            derived_consequences=["Consequence"],
        )

        result = check_deduction_to_induction_gate(ded)
        assert result.decision == GateDecision.PASS

    def test_deduction_to_induction_gate_fail(self):
        """Deduction without consequences fails D→I gate."""
        ded = DeductionResult(
            hypothesis_id="h1",
            derived_consequences=[],  # Empty!
        )

        result = check_deduction_to_induction_gate(ded)
        assert result.decision == GateDecision.BLOCKED

    def test_induction_completion_confirmed(self):
        """Confirmed induction completes cycle."""
        ind = InductionResult(
            hypothesis_id="h1",
            predictions_tested=["P1", "P2"],
            predictions_confirmed=["P1", "P2"],
            predictions_refuted=[],
        )

        result = check_induction_completion_gate(ind)
        assert result.decision == GateDecision.PASS

    def test_induction_completion_refuted(self):
        """Refuted induction should loop back."""
        ind = InductionResult(
            hypothesis_id="h1",
            predictions_tested=["P1"],
            predictions_confirmed=[],
            predictions_refuted=["P1"],
        )

        result = check_induction_completion_gate(ind)
        # Refuted means we should loop back (not complete)
        assert result.decision == GateDecision.LOOP_BACK

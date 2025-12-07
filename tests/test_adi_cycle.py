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
            testable_predictions=["Memory grows linearly over time"],
            plausibility_score=0.7,
            plausibility_rationale="Matches observed pattern",
        )

        is_valid, msg = hyp.is_valid_for_deduction()
        assert is_valid
        assert msg == ""

    def test_hypothesis_without_predictions(self):
        """Hypothesis without predictions fails gate."""
        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="Something is wrong",
            anomaly_addressed="System issues",
            testable_predictions=[],  # Empty!
            plausibility_score=0.5,
            plausibility_rationale="Vague",
        )

        is_valid, msg = hyp.is_valid_for_deduction()
        assert not is_valid
        assert "testable prediction" in msg.lower()

    def test_hypothesis_without_anomaly(self):
        """Hypothesis without anomaly fails gate."""
        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="Random idea",
            anomaly_addressed="",  # Empty!
            testable_predictions=["Prediction"],
            plausibility_score=0.5,
            plausibility_rationale="",
        )

        is_valid, msg = hyp.is_valid_for_deduction()
        assert not is_valid
        assert "anomaly" in msg.lower()


class TestADICycleController:
    """Test ADI cycle controller."""

    def test_initial_state(self):
        """Controller starts in ABDUCTION phase."""
        controller = ADICycleController()
        assert controller.current_phase == ReasoningPhase.ABDUCTION
        assert controller.artifact_state == ArtifactLifecycle.EXPLORATION

    def test_submit_valid_hypothesis(self):
        """Valid hypothesis transitions to DEDUCTION."""
        controller = ADICycleController()

        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="Memory leak hypothesis",
            anomaly_addressed="High memory",
            testable_predictions=["Memory grows over time"],
            plausibility_score=0.8,
        )

        result = controller.submit_hypothesis(hyp)
        assert result.passed
        assert controller.current_phase == ReasoningPhase.DEDUCTION

    def test_submit_invalid_hypothesis(self):
        """Invalid hypothesis stays in ABDUCTION."""
        controller = ADICycleController()

        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="Bad hypothesis",
            anomaly_addressed="",  # Missing anomaly
            testable_predictions=[],  # Missing predictions
            plausibility_score=0.5,
        )

        result = controller.submit_hypothesis(hyp)
        assert not result.passed
        assert controller.current_phase == ReasoningPhase.ABDUCTION

    def test_submit_deduction(self):
        """Valid deduction transitions to INDUCTION."""
        controller = ADICycleController()

        # First submit valid hypothesis
        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="Hypothesis",
            anomaly_addressed="Anomaly",
            testable_predictions=["Prediction 1"],
            plausibility_score=0.8,
        )
        controller.submit_hypothesis(hyp)

        # Now submit deduction
        deduction = DeductionResult(
            hypothesis_id="h1",
            derived_consequences=["Consequence 1"],
            testable_predictions=["Test prediction"],
        )

        result = controller.submit_deduction(deduction)
        assert result.passed
        assert controller.current_phase == ReasoningPhase.INDUCTION

    def test_max_cycles_limit(self):
        """Controller enforces max cycles."""
        controller = ADICycleController(max_cycles=2)

        for _ in range(3):  # Try 3 cycles
            # Submit hypothesis
            hyp = Hypothesis(
                hypothesis_id=f"h{_}",
                statement="Hypothesis",
                anomaly_addressed="Anomaly",
                testable_predictions=["Prediction"],
                plausibility_score=0.8,
            )
            controller.submit_hypothesis(hyp)

            # Submit deduction
            ded = DeductionResult(
                hypothesis_id=f"h{_}",
                derived_consequences=["Consequence"],
                testable_predictions=["Test"],
            )
            controller.submit_deduction(ded)

            # Submit refuting induction (should loop back)
            ind = InductionResult(
                hypothesis_id=f"h{_}",
                predictions_tested=["Test"],
                predictions_confirmed=[],
                predictions_refuted=["Test"],  # Refuted
                conclusion="refuted",
            )
            controller.submit_induction(ind)

        # After 2 cycles, should stop
        assert controller.cycle_count >= 2


class TestGateChecks:
    """Test individual gate check functions."""

    def test_abduction_to_deduction_gate_pass(self):
        """Valid hypothesis passes A→D gate."""
        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="Valid",
            anomaly_addressed="Problem",
            testable_predictions=["Prediction"],
            plausibility_score=0.7,
        )

        result = check_abduction_to_deduction_gate(hyp)
        assert result.passed

    def test_abduction_to_deduction_gate_fail(self):
        """Invalid hypothesis fails A→D gate."""
        hyp = Hypothesis(
            hypothesis_id="h1",
            statement="",  # Empty statement
            anomaly_addressed="",
            testable_predictions=[],
            plausibility_score=0.1,
        )

        result = check_abduction_to_deduction_gate(hyp)
        assert not result.passed

    def test_deduction_to_induction_gate_pass(self):
        """Valid deduction passes D→I gate."""
        ded = DeductionResult(
            hypothesis_id="h1",
            derived_consequences=["Consequence"],
            testable_predictions=["Prediction"],
        )

        result = check_deduction_to_induction_gate(ded)
        assert result.passed

    def test_deduction_to_induction_gate_fail(self):
        """Deduction without predictions fails D→I gate."""
        ded = DeductionResult(
            hypothesis_id="h1",
            derived_consequences=[],
            testable_predictions=[],  # Empty!
        )

        result = check_deduction_to_induction_gate(ded)
        assert not result.passed

    def test_induction_completion_confirmed(self):
        """Confirmed induction completes cycle."""
        ind = InductionResult(
            hypothesis_id="h1",
            predictions_tested=["P1", "P2"],
            predictions_confirmed=["P1", "P2"],
            predictions_refuted=[],
            conclusion="confirmed",
        )

        result = check_induction_completion_gate(ind)
        assert result.passed
        assert result.should_complete

    def test_induction_completion_refuted(self):
        """Refuted induction should loop back."""
        ind = InductionResult(
            hypothesis_id="h1",
            predictions_tested=["P1"],
            predictions_confirmed=[],
            predictions_refuted=["P1"],
            conclusion="refuted",
        )

        result = check_induction_completion_gate(ind)
        assert result.passed
        assert not result.should_complete  # Should loop back

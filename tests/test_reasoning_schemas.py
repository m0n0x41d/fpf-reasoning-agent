"""
Tests for SGR Reasoning Schemas (Phase 2)

Tests the Pydantic schemas for:
- Abduction (hypothesis generation)
- Deduction (consequence derivation)
- Induction (evidence testing)
- Synthesis (knowledge integration)
- Lifecycle (state transitions)
"""

import pytest
from pydantic import ValidationError

from src.fpf_agent.reasoning.schemas.abduction import (
    AbductionOutput,
    AnomalyAnalysis,
    HypothesisCandidate,
    ProblemType,
)
from src.fpf_agent.reasoning.schemas.deduction import (
    DeductionOutput,
    DerivationStep,
    InferenceType,
    LogicalPremise,
    PremiseSource,
    TestablePrediction,
    PredictionCriticality,
)
from src.fpf_agent.reasoning.schemas.induction import (
    EvidenceItem,
    EvidenceSourceType,
    InductionOutput,
    InductionVerdict,
    PredictionMatch,
    PredictionTest,
)
from src.fpf_agent.reasoning.schemas.synthesis import (
    ConflictResolution,
    DisputedClaim,
    KnowledgeGap,
    SourceContribution,
    SynthesisOutput,
)
from src.fpf_agent.reasoning.lifecycle import (
    LifecycleManager,
    TransitionRule,
    check_can_transition,
)
from src.fpf_agent.kernel.types import LifecycleState


# =============================================================================
# ABDUCTION SCHEMA TESTS
# =============================================================================


class TestAnomalyAnalysis:
    """Tests for AnomalyAnalysis schema."""

    def test_valid_anomaly(self):
        anomaly = AnomalyAnalysis(
            problem_type=ProblemType.EXPLANATORY_GAP,
            what_needs_explaining="Why does caching improve performance?",
            why_existing_insufficient="Current model assumes O(1) access",
            key_observations=["Latency drops 50% with cache"],
        )
        assert anomaly.problem_type == ProblemType.EXPLANATORY_GAP
        assert "caching" in anomaly.what_needs_explaining

    def test_requires_key_observations(self):
        with pytest.raises(ValidationError):
            AnomalyAnalysis(
                problem_type=ProblemType.PREDICTION_FAILURE,
                what_needs_explaining="Why did the test fail?",
                why_existing_insufficient="Unknown",
                key_observations=[],  # Must have at least 1
            )


class TestHypothesisCandidate:
    """Tests for HypothesisCandidate schema."""

    def test_valid_hypothesis(self):
        h = HypothesisCandidate(
            statement="Caching reduces database round-trips",
            mechanism="Cache stores frequently accessed data in memory",
            testable_predictions=["Cached queries should be faster"],
        )
        assert h.statement
        assert len(h.testable_predictions) >= 1

    def test_requires_testable_prediction(self):
        with pytest.raises(ValidationError):
            HypothesisCandidate(
                statement="Something happens",
                mechanism="Through some process",
                testable_predictions=[],  # Must have at least 1
            )


class TestAbductionOutput:
    """Tests for AbductionOutput schema."""

    def test_valid_abduction_output(self):
        output = AbductionOutput(
            anomaly=AnomalyAnalysis(
                problem_type=ProblemType.NOVEL_PHENOMENON,
                what_needs_explaining="New behavior observed",
                why_existing_insufficient="No prior model",
                key_observations=["Obs 1"],
            ),
            candidates=[
                HypothesisCandidate(
                    statement="Hypothesis A",
                    mechanism="Mechanism A",
                    testable_predictions=["Prediction A"],
                ),
                HypothesisCandidate(
                    statement="Hypothesis B",
                    mechanism="Mechanism B",
                    testable_predictions=["Prediction B"],
                ),
            ],
            selection_criteria=["simplicity"],
            best_index=0,
            selection_rationale="A is simpler than B",
            alternatives_summary="B rejected due to complexity",
            plausibility_score=0.7,
            confidence_in_selection=0.6,
        )
        assert output.best_hypothesis.statement == "Hypothesis A"
        assert len(output.alternative_hypotheses) == 1

    def test_requires_at_least_two_candidates(self):
        with pytest.raises(ValidationError):
            AbductionOutput(
                anomaly=AnomalyAnalysis(
                    problem_type=ProblemType.EXPLANATORY_GAP,
                    what_needs_explaining="Test",
                    why_existing_insufficient="Test",
                    key_observations=["Obs"],
                ),
                candidates=[
                    HypothesisCandidate(
                        statement="Only one",
                        mechanism="Single",
                        testable_predictions=["P"],
                    ),
                ],  # Need at least 2
                selection_criteria=["test"],
                best_index=0,
                selection_rationale="Only option",
                alternatives_summary="None",
                plausibility_score=0.5,
                confidence_in_selection=0.5,
            )

    def test_best_index_bounds(self):
        with pytest.raises(ValidationError):
            AbductionOutput(
                anomaly=AnomalyAnalysis(
                    problem_type=ProblemType.EXPLANATORY_GAP,
                    what_needs_explaining="Test",
                    why_existing_insufficient="Test",
                    key_observations=["Obs"],
                ),
                candidates=[
                    HypothesisCandidate(
                        statement="A", mechanism="A", testable_predictions=["A"]
                    ),
                    HypothesisCandidate(
                        statement="B", mechanism="B", testable_predictions=["B"]
                    ),
                ],
                selection_criteria=["test"],
                best_index=-1,  # Invalid index
                selection_rationale="Invalid",
                alternatives_summary="None",
                plausibility_score=0.5,
                confidence_in_selection=0.5,
            )


# =============================================================================
# DEDUCTION SCHEMA TESTS
# =============================================================================


class TestLogicalPremise:
    """Tests for LogicalPremise schema."""

    def test_valid_premise(self):
        p = LogicalPremise(
            premise_id="P1",
            statement="All men are mortal",
            source=PremiseSource.BACKGROUND_KNOWLEDGE,
            confidence=0.99,
        )
        assert p.premise_id == "P1"
        assert p.confidence == 0.99

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            LogicalPremise(
                premise_id="P1",
                statement="Test",
                source=PremiseSource.HYPOTHESIS,
                confidence=1.5,  # Out of bounds
            )


class TestDerivationStep:
    """Tests for DerivationStep schema."""

    def test_valid_step(self):
        step = DerivationStep(
            step_number=1,
            from_premises=["P1", "P2"],
            inference_type=InferenceType.MODUS_PONENS,
            conclusion="Socrates is mortal",
            explanation="From P1 and P2 via modus ponens",
        )
        assert step.step_id == "S1"

    def test_requires_premises(self):
        with pytest.raises(ValidationError):
            DerivationStep(
                step_number=1,
                from_premises=[],  # Need at least 1
                inference_type=InferenceType.MODUS_PONENS,
                conclusion="Test",
                explanation="Test",
            )


class TestDeductionOutput:
    """Tests for DeductionOutput schema."""

    def test_valid_deduction(self):
        output = DeductionOutput(
            hypothesis_statement="X causes Y",
            premises=[
                LogicalPremise(
                    premise_id="P1",
                    statement="X causes Y",
                    source=PremiseSource.HYPOTHESIS,
                    confidence=0.7,
                ),
            ],
            predictions=[
                TestablePrediction(
                    prediction_id="T1",
                    prediction="Y increases when X increases",
                    observable="Y measurement",
                    expected_outcome="Positive correlation",
                    falsification_criterion="No correlation or negative",
                ),
            ],
            internal_consistency=True,
            formality_achieved=3,
            overall_confidence=0.65,
        )
        assert output.has_critical_prediction is False  # Default is 'supporting'

    def test_requires_consistency_notes_when_inconsistent(self):
        with pytest.raises(ValidationError):
            DeductionOutput(
                hypothesis_statement="Test",
                premises=[
                    LogicalPremise(
                        premise_id="P1",
                        statement="Test",
                        source=PremiseSource.HYPOTHESIS,
                        confidence=0.5,
                    ),
                ],
                predictions=[
                    TestablePrediction(
                        prediction_id="T1",
                        prediction="Test",
                        observable="Test",
                        expected_outcome="Test",
                        falsification_criterion="Test falsification",
                    ),
                ],
                internal_consistency=False,  # Inconsistent but no notes
                formality_achieved=2,
                overall_confidence=0.5,
            )


# =============================================================================
# INDUCTION SCHEMA TESTS
# =============================================================================


class TestPredictionTest:
    """Tests for PredictionTest schema."""

    def test_weighted_support_calculation(self):
        test = PredictionTest(
            prediction_id="T1",
            prediction_text="Test prediction",
            evidence=[
                EvidenceItem(
                    evidence_id="E1",
                    description="Supporting evidence",
                    source="Lab",
                    source_type=EvidenceSourceType.EXPERIMENTAL,
                    reliability=0.8,
                    relevance=0.9,
                ),
            ],
            observed_outcome="As predicted",
            expected_outcome="Expected",
            match_result=PredictionMatch.MATCH,
            assessment="Evidence supports",
            confidence_in_assessment=0.85,
        )
        # weighted_support = confidence * sum(reliability * relevance)
        expected = 0.85 * (0.8 * 0.9)
        assert abs(test.weighted_support - expected) < 0.01

    def test_mismatch_gives_negative_support(self):
        test = PredictionTest(
            prediction_id="T1",
            prediction_text="Test prediction",
            evidence=[
                EvidenceItem(
                    evidence_id="E1",
                    description="Refuting evidence from experiment",
                    source="Lab",
                    source_type=EvidenceSourceType.EXPERIMENTAL,
                    reliability=0.9,
                    relevance=0.8,
                ),
            ],
            observed_outcome="Opposite of expected",
            expected_outcome="Something else entirely",
            match_result=PredictionMatch.MISMATCH,
            assessment="Evidence refutes the prediction clearly",
            confidence_in_assessment=0.9,
        )
        assert test.weighted_support < 0


class TestInductionOutput:
    """Tests for InductionOutput schema."""

    def test_valid_induction(self):
        output = InductionOutput(
            hypothesis_id="h1",
            hypothesis_statement="Test hypothesis statement",
            prediction_tests=[
                PredictionTest(
                    prediction_id="T1",
                    prediction_text="Test prediction text",
                    observed_outcome="Observed outcome matched",
                    expected_outcome="Expected outcome",
                    match_result=PredictionMatch.MATCH,
                    assessment="Evidence matched the prediction as expected",
                    confidence_in_assessment=0.8,
                ),
            ],
            overall_verdict=InductionVerdict.CORROBORATED,
            verdict_rationale="All predictions confirmed by evidence",
            reliability_delta=0.2,
            overall_confidence=0.75,
        )
        assert not output.should_loop_back
        assert output.confirmation_rate == 1.0

    def test_refuted_requires_new_anomaly(self):
        with pytest.raises(ValidationError):
            InductionOutput(
                hypothesis_id="h1",
                hypothesis_statement="Failed hypothesis statement",
                prediction_tests=[
                    PredictionTest(
                        prediction_id="T1",
                        prediction_text="Test prediction text",
                        observed_outcome="Wrong outcome observed",
                        expected_outcome="Expected something else",
                        match_result=PredictionMatch.MISMATCH,
                        assessment="The prediction failed to match evidence",
                        confidence_in_assessment=0.9,
                    ),
                ],
                overall_verdict=InductionVerdict.REFUTED,
                verdict_rationale="Critical prediction failed to match evidence",
                reliability_delta=-0.5,
                overall_confidence=0.8,
                # Missing if_refuted_new_anomaly
            )


# =============================================================================
# SYNTHESIS SCHEMA TESTS
# =============================================================================


class TestSynthesisOutput:
    """Tests for SynthesisOutput schema."""

    def test_valid_synthesis(self):
        output = SynthesisOutput(
            topic="Database optimization",
            synthesis_scope="Query performance",
            sources=[
                SourceContribution(
                    source_id="s1",
                    source_entity="Caching paper",
                    assurance_level="L1",
                    key_claims=["Caching helps"],
                    reliability=0.8,
                    weight_in_synthesis=0.6,
                ),
            ],
            consensus_claims=["Caching improves read performance"],
            synthesis_statement="Based on sources, caching is effective for read-heavy workloads",
            key_findings=["Caching helps"],
            overall_confidence=0.7,
            confidence_rationale="Good source reliability",
        )
        assert output.source_count == 1
        assert not output.has_unresolved_conflicts

    def test_unresolved_conflict_detection(self):
        output = SynthesisOutput(
            topic="Test",
            synthesis_scope="Test",
            disputed_claims=[
                DisputedClaim(
                    claim_a="X is true",
                    source_a_ids=["s1"],
                    claim_b="X is false",
                    source_b_ids=["s2"],
                    nature_of_conflict="Direct contradiction",
                    resolution=ConflictResolution.UNRESOLVED,
                ),
            ],
            synthesis_statement="Conflicting evidence on X",
            key_findings=["Conflict exists"],
            overall_confidence=0.3,
            confidence_rationale="Unresolved conflict",
        )
        assert output.has_unresolved_conflicts


# =============================================================================
# LIFECYCLE TESTS
# =============================================================================


class TestLifecycleManager:
    """Tests for LifecycleManager."""

    def test_allowed_transitions(self):
        manager = LifecycleManager()

        # Create a mock episteme-like object
        class MockEpisteme:
            lifecycle_state = LifecycleState.EXPLORATION
            assurance_level = "L0"
            evidence_ids = []
            claim_graph = {
                "statement": "Test hypothesis",
                "testable_predictions": ["P1"],
            }

        episteme = MockEpisteme()
        allowed = manager.get_allowed_transitions(episteme)
        assert LifecycleState.SHAPING in allowed

    def test_blocked_transition_without_hypothesis(self):
        manager = LifecycleManager()

        class MockEpisteme:
            lifecycle_state = LifecycleState.EXPLORATION
            assurance_level = "L0"
            evidence_ids = []
            claim_graph = {}  # No hypothesis

        episteme = MockEpisteme()
        result = manager.check_transition(episteme, LifecycleState.SHAPING)
        assert not result.success
        assert "has_hypothesis" in result.checks_failed

    def test_cannot_skip_states(self):
        manager = LifecycleManager()

        class MockEpisteme:
            lifecycle_state = LifecycleState.EXPLORATION
            assurance_level = "L0"
            evidence_ids = []
            claim_graph = {"statement": "Test"}

        episteme = MockEpisteme()
        # Should not be able to go directly to OPERATE
        result = manager.check_transition(episteme, LifecycleState.OPERATE)
        assert not result.success


# =============================================================================
# CONVERSION TESTS
# =============================================================================


class TestSchemaConversions:
    """Test conversion methods between SGR schemas and ADI controller types."""

    def test_abduction_to_hypothesis_dict(self):
        output = AbductionOutput(
            anomaly=AnomalyAnalysis(
                problem_type=ProblemType.EXPLANATORY_GAP,
                what_needs_explaining="Why does X cause Y in this system?",
                why_existing_insufficient="Current models do not explain this",
                key_observations=["Observation about the phenomenon"],
            ),
            candidates=[
                HypothesisCandidate(
                    statement="Hypothesis 1: X causes Y through mechanism A",
                    mechanism="Mechanism A operates by triggering B",
                    testable_predictions=["Prediction 1", "Prediction 2"],
                ),
                HypothesisCandidate(
                    statement="Hypothesis 2: X causes Y through mechanism B",
                    mechanism="Mechanism B operates differently",
                    testable_predictions=["Prediction 3"],
                ),
            ],
            selection_criteria=["simplicity"],
            best_index=0,
            selection_rationale="H1 is simpler and more directly testable",
            alternatives_summary="H2 rejected due to complexity",
            plausibility_score=0.7,
            confidence_in_selection=0.6,
        )

        d = output.to_hypothesis_dict()
        assert "Hypothesis 1" in d["hypothesis_statement"]
        assert d["plausibility_score"] == 0.7
        assert len(d["testable_predictions"]) == 2
        assert "Hypothesis 2" in d["competing_hypotheses"][0]

    def test_induction_to_induction_dict(self):
        output = InductionOutput(
            hypothesis_id="h1",
            hypothesis_statement="Test hypothesis for conversion",
            prediction_tests=[
                PredictionTest(
                    prediction_id="T1",
                    prediction_text="Prediction 1 text",
                    observed_outcome="Observed outcome 1",
                    expected_outcome="Expected outcome 1",
                    match_result=PredictionMatch.MATCH,
                    assessment="Evidence matched prediction as expected",
                    confidence_in_assessment=0.8,
                ),
                PredictionTest(
                    prediction_id="T2",
                    prediction_text="Prediction 2 text",
                    observed_outcome="Observed outcome 2",
                    expected_outcome="Expected outcome 2",
                    match_result=PredictionMatch.MISMATCH,
                    assessment="Evidence contradicted the prediction",
                    confidence_in_assessment=0.9,
                ),
            ],
            overall_verdict=InductionVerdict.WEAKENED,
            verdict_rationale="Mixed results from testing predictions",
            reliability_delta=-0.1,
            overall_confidence=0.6,
        )

        d = output.to_induction_dict()
        assert d["overall_verdict"] == "partial"
        assert "Prediction 1 text" in d["predictions_confirmed"]
        assert "Prediction 2 text" in d["predictions_refuted"]

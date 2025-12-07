"""
Tests for FPF Integration Layer
"""

import pytest

from fpf_agent.core.integration import (
    FPFReasoningPipeline,
    PipelineState,
    StepResult,
    FPFValidationError,
    create_pipeline,
    validate_step_standalone,
    compute_fgr_standalone,
)
from fpf_agent.core.adi_cycle import ReasoningPhase, ArtifactLifecycle


class TestPipelineState:
    """Test pipeline state tracking."""

    def test_initial_state(self):
        """Pipeline state starts with defaults."""
        state = PipelineState(session_id="test")
        assert state.session_id == "test"
        assert state.step_count == 0
        assert state.current_phase == ReasoningPhase.ABDUCTION
        assert state.artifact_state == ArtifactLifecycle.EXPLORATION


class TestFPFReasoningPipeline:
    """Test main pipeline class."""

    def test_pipeline_creation(self):
        """Can create pipeline."""
        pipeline = FPFReasoningPipeline()
        assert pipeline.session_id is not None
        assert pipeline.state.step_count == 0

    def test_process_basic_step(self):
        """Can process a basic reasoning step."""
        pipeline = FPFReasoningPipeline()

        step = {
            "object_of_talk": {
                "category": "System",
                "description": "Database server",
            },
            "context": {
                "context_id": "engineering",
            },
            "temporal_stance": {
                "scope": "run_time",
            },
            "current_understanding": "Analyzing the system behavior.",
            "next_action": {
                "action_type": "analyze_query",
            },
        }

        result = pipeline.process_step(step)

        assert isinstance(result, StepResult)
        assert result.step_number == 1
        assert pipeline.state.step_count == 1

    def test_process_hypothesis_generation(self):
        """Hypothesis generation triggers ADI cycle."""
        pipeline = FPFReasoningPipeline()

        step = {
            "object_of_talk": {
                "category": "Claim",
                "description": "Hypothesis about cause",
            },
            "context": {
                "context_id": "debugging",
            },
            "temporal_stance": {
                "scope": "run_time",
            },
            "current_understanding": "Generating hypothesis.",
            "next_action": {
                "action_type": "generate_hypothesis",
                "anomaly": "High memory usage",
                "hypothesis_statement": "Memory leak in connection pool",
                "testable_predictions": ["Memory grows over time"],
                "plausibility_score": 0.8,
            },
        }

        result = pipeline.process_step(step)

        assert pipeline.state.hypotheses_generated == 1
        # Should have transitioned to DEDUCTION if hypothesis was valid
        assert pipeline.state.current_phase in [ReasoningPhase.ABDUCTION, ReasoningPhase.DEDUCTION]

    def test_context_transition_tracking(self):
        """Context transitions are tracked."""
        pipeline = FPFReasoningPipeline()

        # First step in context A
        step1 = {
            "object_of_talk": {"category": "System", "description": "Sys"},
            "context": {"context_id": "context_a"},
            "temporal_stance": {"scope": "run_time"},
            "current_understanding": "In context A",
            "next_action": {"action_type": "analyze_query"},
        }
        pipeline.process_step(step1)

        # Second step in context B
        step2 = {
            "object_of_talk": {"category": "System", "description": "Sys"},
            "context": {"context_id": "context_b"},
            "temporal_stance": {"scope": "run_time"},
            "current_understanding": "In context B",
            "next_action": {"action_type": "analyze_query"},
        }
        result = pipeline.process_step(step2)

        # Should have recorded transition
        assert len(pipeline.state.context_transitions) == 1
        assert pipeline.state.context_transitions[0]["from"] == "context_a"
        assert pipeline.state.context_transitions[0]["to"] == "context_b"

    def test_strict_mode_raises_on_critical(self):
        """Strict mode raises on critical violations."""
        pipeline = FPFReasoningPipeline(strict_mode=True)

        # Step with System/Episteme conflation
        step = {
            "object_of_talk": {
                "category": "System",
                "description": "A theory about computation",  # Episteme-like
            },
            "context": {"context_id": "test"},
            "temporal_stance": {
                "scope": "design_time",
            },
            "current_understanding": "The system knows and understands.",  # Conflation
            "trust_assessment": {
                "formality": 8,
                "reliability": 0.9,  # High reliability
                "evidence_references": [],  # No evidence!
            },
            "next_action": {"action_type": "analyze_query"},
        }

        with pytest.raises(FPFValidationError):
            pipeline.process_step(step)

    def test_finalize_summary(self):
        """Finalize returns complete summary."""
        pipeline = FPFReasoningPipeline()

        step = {
            "object_of_talk": {"category": "Question", "description": "User query"},
            "context": {"context_id": "default"},
            "temporal_stance": {"scope": "run_time"},
            "current_understanding": "Processing query.",
            "next_action": {"action_type": "analyze_query"},
        }
        pipeline.process_step(step)

        summary = pipeline.finalize()

        assert "session_id" in summary
        assert "total_steps" in summary
        assert summary["total_steps"] == 1
        assert "validation" in summary
        assert "evidence" in summary
        assert "context" in summary

    def test_validation_summary(self):
        """Can get validation summary."""
        pipeline = FPFReasoningPipeline()

        step = {
            "object_of_talk": {"category": "System", "description": "Server"},
            "context": {"context_id": "engineering"},
            "temporal_stance": {"scope": "run_time"},
            "current_understanding": "Analyzing.",
            "next_action": {"action_type": "analyze_query"},
        }
        pipeline.process_step(step)

        summary = pipeline.get_validation_summary()
        assert "FPF Validation Summary" in summary


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_pipeline(self):
        """create_pipeline returns configured pipeline."""
        pipeline = create_pipeline(strict_mode=False, max_cycles=5)
        assert isinstance(pipeline, FPFReasoningPipeline)
        assert not pipeline.strict_mode

    def test_validate_step_standalone(self):
        """Can validate step without full pipeline."""
        step = {
            "object_of_talk": {
                "category": "System",
                "description": "Database",
            },
            "temporal_stance": {
                "scope": "run_time",
            },
            "current_understanding": "The database is functioning.",
        }

        result = validate_step_standalone(step)
        assert hasattr(result, "is_valid")

    def test_compute_fgr_standalone(self):
        """Can compute F-G-R without pipeline."""
        evidence = [
            {"evidence_id": "e1", "reliability": 0.8},
            {"evidence_id": "e2", "reliability": 0.6},
        ]

        fgr = compute_fgr_standalone(
            claim="The system is reliable.",
            evidence=evidence,
            context="production",
        )

        assert "formality" in fgr
        assert "reliability" in fgr
        # Weakest link: min(0.8, 0.6) = 0.6
        assert fgr["reliability"] == pytest.approx(0.6, rel=0.01)
        assert fgr["scope_context"] == "production"

    def test_compute_fgr_no_evidence(self):
        """F-G-R without evidence is L0."""
        fgr = compute_fgr_standalone(
            claim="Unsubstantiated claim",
            evidence=[],
        )

        assert fgr["reliability"] == 0.0
        assert fgr["assurance_level"] == "L0_unsubstantiated"


class TestPipelineEdgeCases:
    """Test edge cases."""

    def test_empty_step(self):
        """Pipeline handles minimal step."""
        pipeline = FPFReasoningPipeline()

        step = {
            "next_action": {"action_type": "unknown"},
        }

        result = pipeline.process_step(step)
        assert result.step_number == 1

    def test_multiple_steps(self):
        """Pipeline handles multiple steps."""
        pipeline = FPFReasoningPipeline()

        for i in range(5):
            step = {
                "object_of_talk": {"category": "Question", "description": f"Step {i}"},
                "context": {"context_id": "test"},
                "temporal_stance": {"scope": "run_time"},
                "current_understanding": f"Processing step {i}",
                "next_action": {"action_type": "analyze_query"},
            }
            pipeline.process_step(step)

        assert pipeline.state.step_count == 5
        assert len(pipeline.get_step_results()) == 5

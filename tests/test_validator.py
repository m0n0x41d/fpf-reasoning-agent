"""
Tests for Strict Distinction Validator (A.7)
"""

import pytest

from fpf_agent.core.validator import (
    DistinctionCategory,
    ViolationSeverity,
    FPFViolation,
    ValidationResult,
    StrictDistinctionValidator,
    FGRValidator,
    FPFReasoningValidator,
)


class TestDistinctionCategories:
    """Test distinction category definitions."""

    def test_core_categories_exist(self):
        """All 8+ core categories should exist."""
        assert DistinctionCategory.SYSTEM
        assert DistinctionCategory.EPISTEME
        assert DistinctionCategory.ROLE
        assert DistinctionCategory.HOLDER
        assert DistinctionCategory.METHOD
        assert DistinctionCategory.WORK
        assert DistinctionCategory.DESIGN_TIME
        assert DistinctionCategory.RUN_TIME


class TestStrictDistinctionValidator:
    """Test strict distinction validator."""

    def test_clean_text_no_violations(self):
        """Clean text should have no violations."""
        validator = StrictDistinctionValidator()
        violations = validator.validate_text(
            "The database stores user records efficiently.",
            location="test"
        )
        assert len(violations) == 0

    def test_system_episteme_conflation_detected(self):
        """System/Episteme conflation should be detected."""
        validator = StrictDistinctionValidator()
        violations = validator.validate_text(
            "The system knows how to process requests.",
            location="test"
        )
        # Should detect "system knows" as potential conflation
        assert any("conflation" in v.description.lower() for v in violations)

    def test_role_holder_conflation_detected(self):
        """Role/Holder conflation should be detected."""
        validator = StrictDistinctionValidator()
        violations = validator.validate_text(
            "The role is performing the task.",
            location="test"
        )
        assert any("role" in v.description.lower() for v in violations)

    def test_object_of_talk_validation_system_vs_episteme(self):
        """Object category should match description."""
        validator = StrictDistinctionValidator()

        # System category but description suggests Episteme
        violations = validator.validate_object_of_talk(
            category="System",
            description="A theory about computation",
            location="test"
        )
        assert len(violations) > 0
        assert any("episteme" in v.description.lower() for v in violations)

    def test_object_of_talk_validation_correct(self):
        """Correct category/description should pass."""
        validator = StrictDistinctionValidator()

        violations = validator.validate_object_of_talk(
            category="System",
            description="A database server processing queries",
            location="test"
        )
        assert len(violations) == 0

    def test_temporal_stance_chimera_detection(self):
        """Design-time/run-time chimera should be detected."""
        validator = StrictDistinctionValidator()

        violations = validator.validate_temporal_stance(
            declared_scope="design_time",
            reasoning_text="The system is currently running and executing requests",
            location="test"
        )
        # Should detect runtime reference in design-time scope
        assert len(violations) > 0
        assert any(v.severity == ViolationSeverity.CRITICAL for v in violations)


class TestFGRValidator:
    """Test F-G-R validation."""

    def test_reliability_without_evidence_flagged(self):
        """High reliability without evidence should be flagged."""
        validator = FGRValidator()

        violations = validator.validate_fgr_assessment(
            formality=5,
            reliability=0.8,  # High reliability
            evidence_count=0,  # No evidence!
            location="test"
        )

        assert len(violations) > 0
        assert any(v.severity == ViolationSeverity.CRITICAL for v in violations)

    def test_high_formality_without_justification_flagged(self):
        """High formality without sufficient evidence should be flagged."""
        validator = FGRValidator()

        violations = validator.validate_fgr_assessment(
            formality=8,  # F8 = machine-checked
            reliability=0.5,
            evidence_count=1,  # Only 1 evidence
            location="test"
        )

        assert len(violations) > 0

    def test_perfect_reliability_warning(self):
        """Perfect reliability (1.0) should generate warning."""
        validator = FGRValidator()

        violations = validator.validate_fgr_assessment(
            formality=3,
            reliability=1.0,  # Perfect reliability
            evidence_count=5,
            location="test"
        )

        assert len(violations) > 0
        assert any(v.severity == ViolationSeverity.WARNING for v in violations)

    def test_valid_assessment_passes(self):
        """Valid F-G-R assessment should pass."""
        validator = FGRValidator()

        violations = validator.validate_fgr_assessment(
            formality=3,
            reliability=0.6,
            evidence_count=3,
            location="test"
        )

        # Should have no critical violations
        critical = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        assert len(critical) == 0


class TestFPFReasoningValidator:
    """Test comprehensive reasoning validator."""

    def test_validate_reasoning_step(self):
        """Can validate a complete reasoning step."""
        validator = FPFReasoningValidator()

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
            "current_understanding": "The database is processing queries normally.",
            "trust_assessment": {
                "formality": 3,
                "reliability": 0.7,
                "evidence_references": [
                    {"evidence_id": "e1", "reliability": 0.7}
                ],
            },
        }

        result = validator.validate_reasoning_step(step)
        assert isinstance(result, ValidationResult)
        assert "A.7_StrictDistinction" in result.fpf_coverage

    def test_validate_response(self):
        """Can validate a final response."""
        validator = FPFReasoningValidator()

        response = {
            "response": "The system is functioning correctly.",
            "confidence": "high",
            "reasoning_trace": ["Step 1: Analyzed", "Step 2: Confirmed"],
        }

        result = validator.validate_response(response)
        assert isinstance(result, ValidationResult)


class TestValidationResult:
    """Test ValidationResult behavior."""

    def test_is_valid_without_critical(self):
        """Result is valid if no critical violations."""
        result = ValidationResult(
            is_valid=True,
            violations=[
                FPFViolation(
                    violation_id="v1",
                    category="test",
                    severity=ViolationSeverity.WARNING,
                    description="Minor issue",
                    location="test",
                    suggestion="Fix it",
                )
            ],
        )
        assert result.is_valid
        assert not result.has_critical

    def test_is_invalid_with_critical(self):
        """Result is invalid if has critical violations."""
        result = ValidationResult(
            is_valid=False,
            violations=[
                FPFViolation(
                    violation_id="v1",
                    category="test",
                    severity=ViolationSeverity.CRITICAL,
                    description="Critical issue",
                    location="test",
                    suggestion="Fix immediately",
                )
            ],
        )
        assert not result.is_valid
        assert result.has_critical
        assert len(result.critical_violations) == 1

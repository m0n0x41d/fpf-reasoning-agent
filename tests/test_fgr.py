"""
Tests for F-G-R Trust Calculus (B.3)
"""

import pytest
from datetime import datetime, timedelta

from fpf_agent.core.fgr import (
    FormalityLevel,
    EvidenceNode,
    EvidenceGraph,
    ClaimScope,
    compute_fgr_assessment,
    compute_formality,
)


class TestFormalityLevel:
    """Test formality level estimation."""

    def test_vague_prose_estimation(self):
        """Vague text should be F0-F2."""
        text = "I think this might work somehow"
        level = compute_formality(text)
        assert level.value <= 2

    def test_structured_prose_estimation(self):
        """Structured text should be F2-F4."""
        text = "The system shall process requests within 100ms latency requirement"
        level = compute_formality(text)
        assert 2 <= level.value <= 5

    def test_formal_text_estimation(self):
        """Formal text with technical terms should be higher."""
        text = "∀x ∈ S: P(x) → Q(x), where S is the set of valid inputs"
        level = compute_formality(text)
        assert level.value >= 4


class TestEvidenceNode:
    """Test evidence node behavior."""

    def test_basic_evidence_creation(self):
        """Can create basic evidence node."""
        node = EvidenceNode(
            evidence_id="ev1",
            evidence_type="observation",
            content_summary="Test observation",
            source="test_source",
            reliability=0.8,
        )
        assert node.reliability == 0.8
        assert node.current_reliability() == 0.8

    def test_evidence_decay_after_expiry(self):
        """Evidence reliability should decay after valid_until."""
        yesterday = datetime.now() - timedelta(days=1)
        node = EvidenceNode(
            evidence_id="ev1",
            evidence_type="observation",
            content_summary="Old observation",
            source="test_source",
            reliability=0.8,
            valid_until=yesterday,
        )
        # Current reliability should be less than base
        assert node.current_reliability() < 0.8

    def test_evidence_no_decay_before_expiry(self):
        """Evidence reliability should not decay before valid_until."""
        tomorrow = datetime.now() + timedelta(days=1)
        node = EvidenceNode(
            evidence_id="ev1",
            evidence_type="observation",
            content_summary="Fresh observation",
            source="test_source",
            reliability=0.8,
            valid_until=tomorrow,
        )
        assert node.current_reliability() == 0.8


class TestEvidenceGraph:
    """Test evidence graph behavior."""

    def test_empty_graph(self):
        """Empty graph should have zero reliability."""
        graph = EvidenceGraph()
        assert len(graph.nodes) == 0

    def test_add_evidence(self):
        """Can add evidence to graph."""
        graph = EvidenceGraph()
        node = EvidenceNode(
            evidence_id="ev1",
            evidence_type="observation",
            content_summary="Test",
            source="test",
            reliability=0.8,
        )
        graph.add_evidence(node)
        assert len(graph.nodes) == 1

    def test_weakest_link_computation(self):
        """Reliability should use weakest-link rule."""
        graph = EvidenceGraph()
        graph.add_evidence(EvidenceNode(
            evidence_id="ev1",
            evidence_type="observation",
            content_summary="Strong evidence",
            source="test",
            reliability=0.9,
        ))
        graph.add_evidence(EvidenceNode(
            evidence_id="ev2",
            evidence_type="observation",
            content_summary="Weak evidence",
            source="test",
            reliability=0.3,
        ))

        # Link both to a claim
        graph.link_evidence_to_claim("claim1", "ev1")
        graph.link_evidence_to_claim("claim1", "ev2")

        # Weakest link: reliability should be min(0.9, 0.3) = 0.3
        reliability = graph.compute_claim_reliability("claim1")
        assert reliability == pytest.approx(0.3, rel=0.01)

    def test_epistemic_debt_with_stale_evidence(self):
        """Stale evidence should contribute to epistemic debt."""
        graph = EvidenceGraph()
        yesterday = datetime.now() - timedelta(days=1)
        graph.add_evidence(EvidenceNode(
            evidence_id="ev1",
            evidence_type="observation",
            content_summary="Stale evidence",
            source="test",
            reliability=0.8,
            valid_until=yesterday,
        ))

        debt = graph.compute_epistemic_debt()
        assert debt > 0


class TestClaimScope:
    """Test claim scope handling."""

    def test_basic_scope(self):
        """Can create basic scope."""
        scope = ClaimScope(
            bounded_context="engineering",
            applicability_conditions=["when system is running"],
        )
        assert scope.bounded_context == "engineering"

    def test_scope_with_exclusions(self):
        """Scope can have exclusions."""
        scope = ClaimScope(
            bounded_context="engineering",
            applicability_conditions=["normal operation"],
            exclusions=["emergency shutdown mode"],
        )
        assert len(scope.exclusions) == 1


class TestComputeFGRAssessment:
    """Test F-G-R assessment computation."""

    def test_basic_assessment(self):
        """Can compute basic F-G-R assessment."""
        assessment = compute_fgr_assessment(
            claim="The system processes requests",
            evidence_nodes=[
                EvidenceNode(
                    evidence_id="ev1",
                    evidence_type="test_result",
                    content_summary="Load test passed",
                    source="qa_team",
                    reliability=0.7,
                ),
            ],
            context="production",
        )

        assert assessment is not None
        assert 0 <= assessment.reliability <= 1
        assert assessment.scope_context == "production"

    def test_assessment_without_evidence(self):
        """Assessment without evidence should have low assurance."""
        assessment = compute_fgr_assessment(
            claim="Unsubstantiated claim",
            evidence_nodes=[],
            context="default",
        )

        assert assessment.assurance_level == "L0_unsubstantiated"
        assert assessment.reliability == 0.0

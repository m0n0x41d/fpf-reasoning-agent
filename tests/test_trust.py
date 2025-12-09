"""
Tests for FPF Trust Layer (Phase 0).

Tests contracts for:
- FormalityLevel ordinal properties
- ClaimScope set operations
- FGRTuple composition
- TrustCalculus WLNK and composition rules
"""
import pytest
from datetime import datetime, timezone, timedelta

from fpf_agent.trust import (
    ClaimScope,
    CongruenceLevel,
    FGRTuple,
    FormalityLevel,
    TrustCalculus,
    check_trust_consistency,
)


class TestFormalityLevel:
    """Test formality level enumeration."""

    def test_ordinal_ordering(self):
        """Formality levels are ordered F0 < F1 < ... < F9."""
        assert FormalityLevel.F0_INFORMAL < FormalityLevel.F1_NARRATIVE
        assert FormalityLevel.F4_TYPED < FormalityLevel.F9_CERTIFIED

    def test_level_descriptions(self):
        """Each level has a description."""
        for level in FormalityLevel:
            assert level.description is not None
            assert len(level.description) > 0


class TestCongruenceLevel:
    """Test congruence level enumeration."""

    def test_ordering(self):
        """CL1 (worst) < CL5 (best)."""
        assert CongruenceLevel.CL1_INCOMPATIBLE < CongruenceLevel.CL5_EXACT

    def test_penalty_cl5(self):
        """CL5 has 0% penalty."""
        assert CongruenceLevel.CL5_EXACT.reliability_penalty() == 0.0

    def test_penalty_cl1(self):
        """CL1 has 40% penalty."""
        assert CongruenceLevel.CL1_INCOMPATIBLE.reliability_penalty() == 0.4


class TestClaimScope:
    """Test claim scope set operations."""

    def test_empty_scope(self):
        """Scope with no contexts and no domain is empty."""
        scope = ClaimScope()
        assert scope.is_empty()

    def test_non_empty_scope(self):
        """Scope with contexts is not empty."""
        scope = ClaimScope(contexts={"physics"})
        assert not scope.is_empty()

    def test_intersect_contexts(self):
        """Intersection keeps only common contexts."""
        scope_a = ClaimScope(contexts={"physics", "chemistry"})
        scope_b = ClaimScope(contexts={"chemistry", "biology"})

        result = scope_a.intersect(scope_b)
        assert result.contexts == {"chemistry"}

    def test_intersect_empty_result(self):
        """Disjoint contexts produce empty intersection."""
        scope_a = ClaimScope(contexts={"physics"})
        scope_b = ClaimScope(contexts={"biology"})

        result = scope_a.intersect(scope_b)
        assert result.contexts == set()

    def test_intersect_temporal_bounds(self):
        """Intersection narrows temporal bounds."""
        now = datetime.now(timezone.utc)
        scope_a = ClaimScope(
            contexts={"ctx"},
            valid_from=now - timedelta(days=10),
            valid_until=now + timedelta(days=10)
        )
        scope_b = ClaimScope(
            contexts={"ctx"},
            valid_from=now - timedelta(days=5),
            valid_until=now + timedelta(days=5)
        )

        result = scope_a.intersect(scope_b)
        assert result.valid_from == now - timedelta(days=5)
        assert result.valid_until == now + timedelta(days=5)

    def test_union_contexts(self):
        """Union combines all contexts."""
        scope_a = ClaimScope(contexts={"physics"})
        scope_b = ClaimScope(contexts={"biology"})

        result = scope_a.union(scope_b)
        assert result.contexts == {"physics", "biology"}

    def test_contains(self):
        """Broader scope contains narrower scope."""
        broad = ClaimScope(contexts={"physics", "chemistry"})
        narrow = ClaimScope(contexts={"physics"})

        assert broad.contains(narrow)
        assert not narrow.contains(broad)

    def test_scope_frozen(self):
        """ClaimScope is immutable."""
        scope = ClaimScope(contexts={"physics"})
        with pytest.raises(Exception):
            scope.contexts = {"biology"}


class TestFGRTuple:
    """Test F-G-R tuple operations."""

    def test_create_tuple(self):
        """Can create F-G-R tuple."""
        fgr = FGRTuple(
            formality=FormalityLevel.F3_STRUCTURED,
            claim_scope=ClaimScope(contexts={"physics"}),
            reliability=0.8
        )
        assert fgr.formality == FormalityLevel.F3_STRUCTURED
        assert fgr.reliability == 0.8

    def test_apply_cl_penalty(self):
        """CL penalty reduces reliability only."""
        fgr = FGRTuple(
            formality=FormalityLevel.F5_CONSTRAINED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.9
        )

        penalized = fgr.apply_cl_penalty(CongruenceLevel.CL3_PARTIAL)

        assert penalized.formality == fgr.formality
        assert penalized.claim_scope == fgr.claim_scope
        assert penalized.reliability == pytest.approx(0.7, abs=0.01)

    def test_with_reliability(self):
        """Can create tuple with updated reliability."""
        fgr = FGRTuple(
            formality=FormalityLevel.F3_STRUCTURED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.5
        )

        updated = fgr.with_reliability(0.8)
        assert fgr.reliability == 0.5
        assert updated.reliability == 0.8

    def test_reliability_clamped(self):
        """Reliability is clamped to [0, 1]."""
        fgr = FGRTuple(
            formality=FormalityLevel.F3_STRUCTURED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.5
        )

        high = fgr.with_reliability(1.5)
        low = fgr.with_reliability(-0.5)

        assert high.reliability == 1.0
        assert low.reliability == 0.0

    def test_is_trustworthy(self):
        """Trustworthy requires R >= 0.5 and non-empty scope."""
        trusty = FGRTuple(
            formality=FormalityLevel.F3_STRUCTURED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.6
        )
        not_trusty = FGRTuple(
            formality=FormalityLevel.F3_STRUCTURED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.3
        )
        empty_scope = FGRTuple(
            formality=FormalityLevel.F3_STRUCTURED,
            claim_scope=ClaimScope(),
            reliability=0.9
        )

        assert trusty.is_trustworthy
        assert not not_trusty.is_trustworthy
        assert not empty_scope.is_trustworthy


class TestTrustCalculus:
    """Test trust composition rules."""

    def test_weakest_link_single(self):
        """Single item returns itself."""
        fgr = FGRTuple(
            formality=FormalityLevel.F5_CONSTRAINED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.9
        )

        result = TrustCalculus.weakest_link([fgr])
        assert result.reliability == 0.9

    def test_weakest_link_min_formality(self):
        """WLNK uses minimum formality."""
        fgr_high = FGRTuple(
            formality=FormalityLevel.F7_DECIDABLE,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.8
        )
        fgr_low = FGRTuple(
            formality=FormalityLevel.F2_CONTROLLED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.9
        )

        result = TrustCalculus.weakest_link([fgr_high, fgr_low])
        assert result.formality == FormalityLevel.F2_CONTROLLED

    def test_weakest_link_min_reliability(self):
        """WLNK uses minimum reliability."""
        fgr_high_r = FGRTuple(
            formality=FormalityLevel.F5_CONSTRAINED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.95
        )
        fgr_low_r = FGRTuple(
            formality=FormalityLevel.F5_CONSTRAINED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.4
        )

        result = TrustCalculus.weakest_link([fgr_high_r, fgr_low_r])
        assert result.reliability == 0.4

    def test_weakest_link_empty_raises(self):
        """WLNK on empty list raises."""
        with pytest.raises(ValueError):
            TrustCalculus.weakest_link([])

    def test_compose_serial(self):
        """Serial composition applies CL penalties."""
        fgr_a = FGRTuple(
            formality=FormalityLevel.F5_CONSTRAINED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.9
        )
        fgr_b = FGRTuple(
            formality=FormalityLevel.F5_CONSTRAINED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.8
        )

        result = TrustCalculus.compose_serial(
            [fgr_a, fgr_b],
            [CongruenceLevel.CL4_ALIGNED]
        )

        assert result.reliability < 0.8
        assert result.reliability == pytest.approx(0.7, abs=0.01)

    def test_compose_serial_wrong_edge_count_raises(self):
        """Serial composition requires N-1 edges for N items."""
        fgr = FGRTuple(
            formality=FormalityLevel.F3_STRUCTURED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.8
        )

        with pytest.raises(ValueError):
            TrustCalculus.compose_serial([fgr, fgr], [])

    def test_compose_parallel_boost(self):
        """Parallel composition can boost reliability."""
        fgr_a = FGRTuple(
            formality=FormalityLevel.F5_CONSTRAINED,
            claim_scope=ClaimScope(contexts={"ctx_a"}),
            reliability=0.7
        )
        fgr_b = FGRTuple(
            formality=FormalityLevel.F5_CONSTRAINED,
            claim_scope=ClaimScope(contexts={"ctx_b"}),
            reliability=0.7
        )

        result = TrustCalculus.compose_parallel([fgr_a, fgr_b])

        assert result.reliability > 0.7
        assert result.reliability <= 0.8
        assert result.claim_scope.contexts == {"ctx_a", "ctx_b"}

    def test_compose_parallel_capped(self):
        """Parallel composition is capped at max(R_i) + 0.1."""
        fgr_low = FGRTuple(
            formality=FormalityLevel.F5_CONSTRAINED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.3
        )

        result = TrustCalculus.compose_parallel([fgr_low, fgr_low, fgr_low])

        assert result.reliability <= 0.4


class TestTrustConsistency:
    """Test trust consistency checking."""

    def test_consistent_trust(self):
        """Correctly derived trust is consistent."""
        evidence = [
            FGRTuple(
                formality=FormalityLevel.F5_CONSTRAINED,
                claim_scope=ClaimScope(contexts={"ctx"}),
                reliability=0.8
            )
        ]
        claimed = FGRTuple(
            formality=FormalityLevel.F3_STRUCTURED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.75
        )

        is_consistent, violations = check_trust_consistency(claimed, evidence)
        assert is_consistent
        assert len(violations) == 0

    def test_inconsistent_formality(self):
        """Higher formality than evidence is inconsistent."""
        evidence = [
            FGRTuple(
                formality=FormalityLevel.F3_STRUCTURED,
                claim_scope=ClaimScope(contexts={"ctx"}),
                reliability=0.8
            )
        ]
        claimed = FGRTuple(
            formality=FormalityLevel.F7_DECIDABLE,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.7
        )

        is_consistent, violations = check_trust_consistency(claimed, evidence)
        assert not is_consistent
        assert any("exceeds evidence max" in v for v in violations)

    def test_inconsistent_reliability(self):
        """Higher reliability than evidence is inconsistent."""
        evidence = [
            FGRTuple(
                formality=FormalityLevel.F5_CONSTRAINED,
                claim_scope=ClaimScope(contexts={"ctx"}),
                reliability=0.5
            )
        ]
        claimed = FGRTuple(
            formality=FormalityLevel.F3_STRUCTURED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.9
        )

        is_consistent, violations = check_trust_consistency(claimed, evidence)
        assert not is_consistent
        assert any("reliability" in v.lower() or "wlnk" in v.lower() for v in violations)

    def test_no_evidence_with_nonzero_reliability(self):
        """Non-zero reliability without evidence is inconsistent."""
        claimed = FGRTuple(
            formality=FormalityLevel.F3_STRUCTURED,
            claim_scope=ClaimScope(contexts={"ctx"}),
            reliability=0.5
        )

        is_consistent, violations = check_trust_consistency(claimed, [])
        assert not is_consistent

"""
Tests for Γ Aggregation Operators (B.1)
"""

import pytest

from fpf_agent.core.aggregation import (
    AggregationInvariant,
    AggregationPolicy,
    GammaType,
    AggregationResult,
    AggregationGuard,
    GammaSys,
    GammaEpist,
    GammaCtx,
    GammaWork,
    PhysicalMeasurement,
    EpistemicClaim,
    ContextFragment,
    WorkRecord,
    aggregate_reliabilities,
    aggregate_formalities,
    check_invariant_compatibility,
    create_policy,
)


class TestAggregationInvariants:
    """Test invariant definitions."""

    def test_all_invariants_exist(self):
        """All five invariants should exist."""
        assert AggregationInvariant.IDEM
        assert AggregationInvariant.COMM
        assert AggregationInvariant.LOC
        assert AggregationInvariant.WLNK
        assert AggregationInvariant.MONO

    def test_wlnk_mono_incompatible(self):
        """WLNK and MONO should be flagged as potentially incompatible."""
        compatible, msg = check_invariant_compatibility([
            AggregationInvariant.WLNK,
            AggregationInvariant.MONO,
        ])
        assert not compatible


class TestAggregationPolicy:
    """Test aggregation policy behavior."""

    def test_create_policy(self):
        """Can create aggregation policy."""
        policy = AggregationPolicy(
            gamma_type=GammaType.EPIST,
            invariants=[AggregationInvariant.WLNK],
            description="Test policy",
        )
        assert policy.gamma_type == GammaType.EPIST

    def test_policy_auto_adds_idem(self):
        """Policy auto-adds IDEM when duplicates not allowed."""
        policy = AggregationPolicy(
            gamma_type=GammaType.CTX,
            invariants=[],
            description="No duplicates",
            allow_duplicates=False,
        )
        assert AggregationInvariant.IDEM in policy.invariants

    def test_policy_warns_on_contradiction(self):
        """Policy warns when COMM declared but order_sensitive."""
        policy = AggregationPolicy(
            gamma_type=GammaType.TIME,
            invariants=[AggregationInvariant.COMM],
            description="Contradictory",
            order_sensitive=True,
        )
        assert len(policy.warnings) > 0


class TestGammaEpist:
    """Test epistemic aggregation (Γ_epist)."""

    def test_weakest_link_reliability(self):
        """Reliability should use weakest-link rule."""
        claims = [
            EpistemicClaim(
                claim_id="c1",
                statement="Claim 1",
                formality=5,
                scope="test",
                reliability=0.9,
            ),
            EpistemicClaim(
                claim_id="c2",
                statement="Claim 2",
                formality=3,
                scope="test",
                reliability=0.4,
            ),
        ]

        operator = GammaEpist()
        result = operator.aggregate(claims)

        # Weakest link: min(0.9, 0.4) = 0.4
        assert result.value.reliability == pytest.approx(0.4, rel=0.01)

    def test_weakest_link_formality(self):
        """Formality should also use weakest-link."""
        claims = [
            EpistemicClaim(
                claim_id="c1",
                statement="Formal claim",
                formality=7,
                scope="test",
                reliability=0.8,
            ),
            EpistemicClaim(
                claim_id="c2",
                statement="Informal claim",
                formality=2,
                scope="test",
                reliability=0.8,
            ),
        ]

        operator = GammaEpist()
        result = operator.aggregate(claims)

        # min(7, 2) = 2
        assert result.value.formality == 2


class TestGammaSys:
    """Test physical system aggregation (Γ_sys)."""

    def test_extensive_property_sum(self):
        """Extensive properties should sum."""
        measurements = [
            PhysicalMeasurement(name="mass", value=10.0, unit="kg", is_extensive=True),
            PhysicalMeasurement(name="mass", value=20.0, unit="kg", is_extensive=True),
        ]

        operator = GammaSys()
        result = operator.aggregate(measurements)

        assert result.value.value == 30.0

    def test_intensive_property_error(self):
        """Intensive properties should raise error."""
        measurements = [
            PhysicalMeasurement(name="temp", value=20.0, unit="C", is_extensive=False),
            PhysicalMeasurement(name="temp", value=30.0, unit="C", is_extensive=False),
        ]

        operator = GammaSys()
        with pytest.raises(ValueError, match="intensive"):
            operator.aggregate(measurements)

    def test_mixed_units_error(self):
        """Different units should raise error."""
        measurements = [
            PhysicalMeasurement(name="mass", value=10.0, unit="kg", is_extensive=True),
            PhysicalMeasurement(name="mass", value=20.0, unit="lb", is_extensive=True),
        ]

        operator = GammaSys()
        with pytest.raises(ValueError, match="units"):
            operator.aggregate(measurements)


class TestGammaCtx:
    """Test context aggregation (Γ_ctx)."""

    def test_context_composition(self):
        """Contexts should compose with term union."""
        fragments = [
            ContextFragment(
                context_id="ctx1",
                terms=["term_a", "term_b"],
                invariants=["inv1"],
            ),
            ContextFragment(
                context_id="ctx2",
                terms=["term_c", "term_a"],
                invariants=["inv2"],
            ),
        ]

        operator = GammaCtx()
        result = operator.aggregate(fragments)

        # Union of terms
        assert set(result.value.terms) == {"term_a", "term_b", "term_c"}
        # Union of invariants
        assert set(result.value.invariants) == {"inv1", "inv2"}


class TestGammaWork:
    """Test work/resource aggregation (Γ_work)."""

    def test_work_sum(self):
        """Work records should sum."""
        records = [
            WorkRecord(work_id="w1", resource_type="time", amount=3.0, unit="hours"),
            WorkRecord(work_id="w2", resource_type="time", amount=2.0, unit="hours"),
        ]

        operator = GammaWork()
        result = operator.aggregate(records)

        assert result.value.amount == 5.0


class TestAggregationGuard:
    """Test free-hand average prevention."""

    def test_guard_requires_policy(self):
        """Aggregation without policy should be blocked."""
        guard = AggregationGuard()

        with pytest.raises(ValueError, match="without.*policy"):
            guard.require_policy(
                operation_name="test_aggregation",
                items=[1, 2, 3],
                policy=None,
            )

    def test_guard_allows_with_policy(self):
        """Aggregation with policy should pass."""
        guard = AggregationGuard()
        policy = create_policy(GammaType.EPIST)

        # Should not raise
        guard.require_policy(
            operation_name="test_aggregation",
            items=[1, 2, 3],
            policy=policy,
        )


class TestConvenienceFunctions:
    """Test convenience aggregation functions."""

    def test_aggregate_reliabilities(self):
        """Can aggregate reliability values."""
        result = aggregate_reliabilities([0.9, 0.7, 0.5])

        # Weakest link: min = 0.5
        assert result.value == pytest.approx(0.5, rel=0.01)
        assert result.invariants_checked[AggregationInvariant.WLNK]

    def test_aggregate_formalities(self):
        """Can aggregate formality values."""
        result = aggregate_formalities([7, 3, 5])

        # Weakest link: min = 3
        assert result.value == 3

    def test_create_policy_defaults(self):
        """create_policy provides sensible defaults."""
        epist_policy = create_policy(GammaType.EPIST)
        assert AggregationInvariant.WLNK in epist_policy.invariants

        sys_policy = create_policy(GammaType.SYS)
        assert AggregationInvariant.COMM in sys_policy.invariants

"""
Tests for FPF Kernel types (Phase 0).

Tests contracts for:
- HolonId, Edition versioning
- UEpisteme immutability and state transitions
- UBoundedContext term validation
- ContextBridge reliability penalties
"""
import pytest
from datetime import datetime, timezone
from uuid import uuid4

from fpf_agent.kernel import (
    AssuranceLevel,
    ContextBridge,
    Edition,
    HolonId,
    LifecycleState,
    StrictDistinctionSlots,
    TemporalStance,
    UBoundedContext,
    UEpisteme,
)


class TestEdition:
    """Test Edition immutability and versioning."""

    def test_default_edition_is_v1(self):
        """New editions start at version 1."""
        edition = Edition()
        assert edition.number == 1
        assert edition.supersedes is None

    def test_edition_is_frozen(self):
        """Editions are immutable after creation."""
        edition = Edition()
        with pytest.raises(Exception):
            edition.number = 2


class TestHolonId:
    """Test HolonId identity and edition chains."""

    def test_holon_id_creation(self):
        """Can create HolonId with context."""
        hid = HolonId(context_id="test_ctx")
        assert hid.context_id == "test_ctx"
        assert hid.edition.number == 1

    def test_next_edition_creates_new_id(self):
        """next_edition creates new HolonId, doesn't mutate."""
        original = HolonId(context_id="ctx")
        next_hid = original.next_edition()

        assert next_hid.id != original.id
        assert next_hid.edition.number == 2
        assert next_hid.edition.supersedes == original.id
        assert original.edition.number == 1

    def test_edition_chain(self):
        """Can create chain of editions."""
        v1 = HolonId(context_id="ctx")
        v2 = v1.next_edition()
        v3 = v2.next_edition()

        assert v3.edition.number == 3
        assert v3.edition.supersedes == v2.id
        assert v2.edition.supersedes == v1.id

    def test_str_representation(self):
        """String representation is readable."""
        hid = HolonId(context_id="my_ctx")
        s = str(hid)
        assert "my_ctx" in s
        assert "@v1" in s


class TestTemporalStance:
    """Test temporal stance enumeration."""

    def test_design_time_lt_run_time(self):
        """Design-time comes before run-time."""
        assert TemporalStance.DESIGN_TIME < TemporalStance.RUN_TIME


class TestLifecycleState:
    """Test lifecycle state progression."""

    def test_lifecycle_ordering(self):
        """States are ordered: exploration < shaping < evidence < operate."""
        assert LifecycleState.EXPLORATION < LifecycleState.SHAPING
        assert LifecycleState.SHAPING < LifecycleState.EVIDENCE
        assert LifecycleState.EVIDENCE < LifecycleState.OPERATE


class TestAssuranceLevel:
    """Test assurance level enumeration."""

    def test_assurance_ordering(self):
        """L0 < L1 < L2."""
        assert AssuranceLevel.L0 < AssuranceLevel.L1
        assert AssuranceLevel.L1 < AssuranceLevel.L2


class TestUEpisteme:
    """Test UEpisteme knowledge artifact."""

    def test_default_state(self):
        """New epistemes start at L0 in exploration."""
        hid = HolonId(context_id="ctx")
        episteme = UEpisteme(
            holon_id=hid,
            described_entity="Test claim"
        )

        assert episteme.assurance_level == AssuranceLevel.L0
        assert episteme.lifecycle_state == LifecycleState.EXPLORATION
        assert episteme.temporal_stance == TemporalStance.DESIGN_TIME

    def test_with_claim_returns_new_instance(self):
        """with_claim returns new episteme, doesn't mutate."""
        hid = HolonId(context_id="ctx")
        original = UEpisteme(holon_id=hid, described_entity="Test")

        updated = original.with_claim("statement", "The sky is blue")

        assert "statement" not in original.claim_graph
        assert updated.claim_graph["statement"] == "The sky is blue"

    def test_with_evidence_returns_new_instance(self):
        """with_evidence returns new episteme, doesn't mutate."""
        hid = HolonId(context_id="ctx")
        original = UEpisteme(holon_id=hid, described_entity="Test")
        ev_id = uuid4()

        updated = original.with_evidence(ev_id)

        assert ev_id not in original.evidence_ids
        assert ev_id in updated.evidence_ids

    def test_transition_lifecycle(self):
        """Can transition lifecycle state."""
        hid = HolonId(context_id="ctx")
        episteme = UEpisteme(holon_id=hid, described_entity="Test")

        shaped = episteme.transition_lifecycle(LifecycleState.SHAPING)

        assert episteme.lifecycle_state == LifecycleState.EXPLORATION
        assert shaped.lifecycle_state == LifecycleState.SHAPING

    def test_elevate_assurance(self):
        """Can elevate assurance level."""
        hid = HolonId(context_id="ctx")
        episteme = UEpisteme(holon_id=hid, described_entity="Test")

        elevated = episteme.elevate_assurance(AssuranceLevel.L1)

        assert episteme.assurance_level == AssuranceLevel.L0
        assert elevated.assurance_level == AssuranceLevel.L1

    def test_elevate_assurance_noop_if_lower(self):
        """Elevating to lower level returns same instance."""
        hid = HolonId(context_id="ctx")
        episteme = UEpisteme(
            holon_id=hid,
            described_entity="Test",
            assurance_level=AssuranceLevel.L2
        )

        same = episteme.elevate_assurance(AssuranceLevel.L1)
        assert same.assurance_level == AssuranceLevel.L2


class TestUBoundedContext:
    """Test bounded context semantics."""

    def test_create_context(self):
        """Can create bounded context."""
        ctx = UBoundedContext(
            context_id="physics",
            name="Classical Physics",
            description="Newtonian mechanics domain"
        )
        assert ctx.context_id == "physics"

    def test_define_term(self):
        """Can define terms in glossary."""
        ctx = UBoundedContext(
            context_id="test",
            name="Test",
            description="Test context"
        )
        updated = ctx.define_term("velocity", "Rate of change of position")

        assert ctx.has_term("velocity") is False
        assert updated.has_term("velocity") is True
        assert updated.get_definition("velocity") == "Rate of change of position"

    def test_term_lookup_case_insensitive(self):
        """Term lookup is case-insensitive."""
        ctx = UBoundedContext(
            context_id="test",
            name="Test",
            description="Test"
        ).define_term("Velocity", "Speed with direction")

        assert ctx.has_term("velocity")
        assert ctx.has_term("VELOCITY")

    def test_validate_terms(self):
        """Can validate terms against glossary."""
        ctx = UBoundedContext(
            context_id="test",
            name="Test",
            description="Test"
        ).define_term("mass", "Amount of matter")

        invalid = ctx.validate_terms(["mass", "undefined_term"])
        assert "undefined_term" in invalid
        assert "mass" not in invalid

    def test_add_invariant(self):
        """Can add invariants to context."""
        ctx = UBoundedContext(
            context_id="test",
            name="Test",
            description="Test"
        )
        updated = ctx.add_invariant("Energy is conserved")

        assert len(ctx.invariants) == 0
        assert "Energy is conserved" in updated.invariants


class TestContextBridge:
    """Test context bridge reliability penalties."""

    def test_bridge_creation(self):
        """Can create bridge between contexts."""
        bridge = ContextBridge(
            source_context_id="physics",
            target_context_id="engineering",
            term_mappings={"force": "load"},
            congruence_level=4
        )
        assert bridge.translate("force") == "load"

    def test_reliability_penalty_cl5(self):
        """CL5 (exact) has 0% penalty."""
        bridge = ContextBridge(
            source_context_id="a",
            target_context_id="b",
            congruence_level=5
        )
        assert bridge.reliability_penalty() == 0.0

    def test_reliability_penalty_cl1(self):
        """CL1 (incompatible) has 40% penalty."""
        bridge = ContextBridge(
            source_context_id="a",
            target_context_id="b",
            congruence_level=1
        )
        assert bridge.reliability_penalty() == 0.4

    def test_reliability_penalty_cl3(self):
        """CL3 (partial) has 20% penalty."""
        bridge = ContextBridge(
            source_context_id="a",
            target_context_id="b",
            congruence_level=3
        )
        assert bridge.reliability_penalty() == 0.2

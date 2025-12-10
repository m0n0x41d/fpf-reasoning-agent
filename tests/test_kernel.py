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
    AssuranceRecord,
    CarrierType,
    ContextBridge,
    Edition,
    HolonId,
    LifecycleState,
    StrictDistinctionSlots,
    SymbolCarrierRecord,
    TemporalStance,
    TypingAssuranceLevel,
    UBoundedContext,
    UEpisteme,
    ValidationAssuranceLevel,
    VerificationAssuranceLevel,
)
from fpf_agent.trust.fgr import ClaimScope, FGRTuple, FormalityLevel


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


class TestAssuranceSubtypes:
    """Test TA/VA/LA assurance subtypes (FPF B.3.3)."""

    def test_typing_assurance_ordering(self):
        """TA0 < TA1 < TA2 < TA3."""
        assert TypingAssuranceLevel.TA0 < TypingAssuranceLevel.TA1
        assert TypingAssuranceLevel.TA1 < TypingAssuranceLevel.TA2
        assert TypingAssuranceLevel.TA2 < TypingAssuranceLevel.TA3

    def test_verification_assurance_ordering(self):
        """VA0 < VA1 < VA2 < VA3."""
        assert VerificationAssuranceLevel.VA0 < VerificationAssuranceLevel.VA1
        assert VerificationAssuranceLevel.VA1 < VerificationAssuranceLevel.VA2
        assert VerificationAssuranceLevel.VA2 < VerificationAssuranceLevel.VA3

    def test_validation_assurance_ordering(self):
        """LA0 < LA1 < LA2 < LA3."""
        assert ValidationAssuranceLevel.LA0 < ValidationAssuranceLevel.LA1
        assert ValidationAssuranceLevel.LA1 < ValidationAssuranceLevel.LA2
        assert ValidationAssuranceLevel.LA2 < ValidationAssuranceLevel.LA3

    def test_assurance_descriptions(self):
        """Each level has a description."""
        assert "No typing" in TypingAssuranceLevel.TA0.description
        assert "proof" in VerificationAssuranceLevel.VA3.description.lower()
        assert "replication" in ValidationAssuranceLevel.LA3.description.lower()


class TestAssuranceRecord:
    """Test AssuranceRecord three-lane tracking."""

    def test_default_record_is_l0(self):
        """New record computes to L0."""
        record = AssuranceRecord()
        assert record.compute_level() == AssuranceLevel.L0
        assert record.typing_assurance == TypingAssuranceLevel.TA0
        assert record.verification_assurance == VerificationAssuranceLevel.VA0
        assert record.validation_assurance == ValidationAssuranceLevel.LA0

    def test_one_lane_at_level_2_gives_l1(self):
        """One lane at level 2 or higher → L1."""
        record = AssuranceRecord(
            typing_assurance=TypingAssuranceLevel.TA2,
            verification_assurance=VerificationAssuranceLevel.VA0,
            validation_assurance=ValidationAssuranceLevel.LA0,
        )
        assert record.compute_level() == AssuranceLevel.L1

    def test_all_lanes_at_level_2_gives_l2(self):
        """All lanes at level 2 or higher → L2."""
        record = AssuranceRecord(
            typing_assurance=TypingAssuranceLevel.TA2,
            verification_assurance=VerificationAssuranceLevel.VA2,
            validation_assurance=ValidationAssuranceLevel.LA2,
        )
        assert record.compute_level() == AssuranceLevel.L2
        assert record.is_fully_assured

    def test_weakest_lane_identification(self):
        """Can identify the weakest lane."""
        record = AssuranceRecord(
            typing_assurance=TypingAssuranceLevel.TA3,
            verification_assurance=VerificationAssuranceLevel.VA1,
            validation_assurance=ValidationAssuranceLevel.LA2,
        )
        assert record.weakest_lane == "verification"

    def test_with_typing_returns_new_record(self):
        """with_typing returns new record, doesn't mutate."""
        original = AssuranceRecord()
        ev_id = uuid4()
        updated = original.with_typing(
            TypingAssuranceLevel.TA2,
            evidence_id=ev_id,
            rationale="Schema validated"
        )

        assert original.typing_assurance == TypingAssuranceLevel.TA0
        assert updated.typing_assurance == TypingAssuranceLevel.TA2
        assert ev_id in updated.typing_evidence_ids
        assert ev_id not in original.typing_evidence_ids
        assert updated.typing_rationale == "Schema validated"

    def test_with_verification_returns_new_record(self):
        """with_verification returns new record, doesn't mutate."""
        original = AssuranceRecord()
        ev_id = uuid4()
        updated = original.with_verification(
            VerificationAssuranceLevel.VA2,
            evidence_id=ev_id,
        )

        assert original.verification_assurance == VerificationAssuranceLevel.VA0
        assert updated.verification_assurance == VerificationAssuranceLevel.VA2
        assert ev_id in updated.verification_evidence_ids

    def test_with_validation_returns_new_record(self):
        """with_validation returns new record, doesn't mutate."""
        original = AssuranceRecord()
        ev_id = uuid4()
        updated = original.with_validation(
            ValidationAssuranceLevel.LA3,
            evidence_id=ev_id,
        )

        assert original.validation_assurance == ValidationAssuranceLevel.LA0
        assert updated.validation_assurance == ValidationAssuranceLevel.LA3
        assert ev_id in updated.validation_evidence_ids

    def test_all_evidence_ids_aggregates_lanes(self):
        """all_evidence_ids returns evidence from all lanes."""
        ta_ev = uuid4()
        va_ev = uuid4()
        la_ev = uuid4()

        record = AssuranceRecord(
            typing_evidence_ids=[ta_ev],
            verification_evidence_ids=[va_ev],
            validation_evidence_ids=[la_ev],
        )

        all_ids = record.all_evidence_ids
        assert ta_ev in all_ids
        assert va_ev in all_ids
        assert la_ev in all_ids

    def test_summary_format(self):
        """Summary is human-readable."""
        record = AssuranceRecord(
            typing_assurance=TypingAssuranceLevel.TA2,
            verification_assurance=VerificationAssuranceLevel.VA1,
            validation_assurance=ValidationAssuranceLevel.LA3,
        )
        summary = record.summary()
        assert "TA2" in summary
        assert "VA1" in summary
        assert "LA3" in summary
        assert "L1" in summary  # computed level


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
        assert episteme.assurance_record.compute_level() == AssuranceLevel.L0

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
        # Create episteme with full assurance via assurance_record
        record = AssuranceRecord(
            typing_assurance=TypingAssuranceLevel.TA2,
            verification_assurance=VerificationAssuranceLevel.VA2,
            validation_assurance=ValidationAssuranceLevel.LA2,
        )
        episteme = UEpisteme(
            holon_id=hid,
            described_entity="Test",
            assurance_record=record,
        )
        assert episteme.assurance_level == AssuranceLevel.L2

        same = episteme.elevate_assurance(AssuranceLevel.L1)
        assert same.assurance_level == AssuranceLevel.L2

    def test_with_typing_assurance(self):
        """Can update typing assurance lane."""
        hid = HolonId(context_id="ctx")
        episteme = UEpisteme(holon_id=hid, described_entity="Test")
        ev_id = uuid4()

        updated = episteme.with_typing_assurance(
            TypingAssuranceLevel.TA2,
            evidence_id=ev_id,
            rationale="Schema validated"
        )

        assert episteme.assurance_record.typing_assurance == TypingAssuranceLevel.TA0
        assert updated.assurance_record.typing_assurance == TypingAssuranceLevel.TA2
        assert ev_id in updated.assurance_record.typing_evidence_ids
        assert updated.assurance_record.typing_rationale == "Schema validated"
        assert updated.assurance_level == AssuranceLevel.L1  # One lane at 2 → L1

    def test_with_verification_assurance(self):
        """Can update verification assurance lane."""
        hid = HolonId(context_id="ctx")
        episteme = UEpisteme(holon_id=hid, described_entity="Test")
        ev_id = uuid4()

        updated = episteme.with_verification_assurance(
            VerificationAssuranceLevel.VA2,
            evidence_id=ev_id,
        )

        assert updated.assurance_record.verification_assurance == VerificationAssuranceLevel.VA2
        assert ev_id in updated.assurance_record.verification_evidence_ids
        assert updated.assurance_level == AssuranceLevel.L1

    def test_with_validation_assurance(self):
        """Can update validation assurance lane."""
        hid = HolonId(context_id="ctx")
        episteme = UEpisteme(holon_id=hid, described_entity="Test")
        ev_id = uuid4()

        updated = episteme.with_validation_assurance(
            ValidationAssuranceLevel.LA2,
            evidence_id=ev_id,
        )

        assert updated.assurance_record.validation_assurance == ValidationAssuranceLevel.LA2
        assert ev_id in updated.assurance_record.validation_evidence_ids
        assert updated.assurance_level == AssuranceLevel.L1

    def test_full_assurance_progression(self):
        """Episteme reaches L2 when all lanes are at level 2+."""
        hid = HolonId(context_id="ctx")
        episteme = UEpisteme(holon_id=hid, described_entity="Test")

        # Start at L0
        assert episteme.assurance_level == AssuranceLevel.L0

        # Add typing → still L1 (only one lane)
        with_ta = episteme.with_typing_assurance(TypingAssuranceLevel.TA2)
        assert with_ta.assurance_level == AssuranceLevel.L1

        # Add verification → still L1 (two lanes, missing validation)
        with_va = with_ta.with_verification_assurance(VerificationAssuranceLevel.VA2)
        assert with_va.assurance_level == AssuranceLevel.L1

        # Add validation → now L2 (all three lanes)
        with_la = with_va.with_validation_assurance(ValidationAssuranceLevel.LA2)
        assert with_la.assurance_level == AssuranceLevel.L2
        assert with_la.assurance_record.is_fully_assured

    def test_assurance_level_synced_with_record(self):
        """assurance_level field stays synced with assurance_record."""
        hid = HolonId(context_id="ctx")

        # Create with pre-filled assurance_record
        record = AssuranceRecord(
            typing_assurance=TypingAssuranceLevel.TA2,
            verification_assurance=VerificationAssuranceLevel.VA2,
            validation_assurance=ValidationAssuranceLevel.LA2,
        )
        episteme = UEpisteme(
            holon_id=hid,
            described_entity="Test",
            assurance_record=record,
        )

        # assurance_level should be computed from record
        assert episteme.assurance_level == AssuranceLevel.L2


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


class TestCarrierType:
    """Test CarrierType enumeration and defaults."""

    def test_carrier_type_values(self):
        """CarrierType has expected values."""
        assert CarrierType.UNKNOWN == 0
        assert CarrierType.PEER_REVIEWED_JOURNAL == 1
        assert CarrierType.PREPRINT == 2

    def test_default_formality(self):
        """Carrier types have sensible default formality."""
        assert CarrierType.PEER_REVIEWED_JOURNAL.default_formality == 3
        assert CarrierType.BLOG_POST.default_formality == 1
        assert CarrierType.DATASET.default_formality == 5
        assert CarrierType.UNKNOWN.default_formality == 0

    def test_default_reliability(self):
        """Carrier types have sensible default reliability."""
        assert CarrierType.PEER_REVIEWED_JOURNAL.default_reliability == 0.7
        assert CarrierType.PREPRINT.default_reliability == 0.4
        assert CarrierType.BLOG_POST.default_reliability == 0.2
        assert CarrierType.UNKNOWN.default_reliability == 0.1


class TestSymbolCarrierRecord:
    """Test SymbolCarrierRecord provenance tracking."""

    def test_create_scr(self):
        """Can create SCR with required fields."""
        scr = SymbolCarrierRecord(
            uri="file:///papers/test.pdf",
            carrier_type=CarrierType.PEER_REVIEWED_JOURNAL,
            content_hash="abc123def456",
        )
        assert scr.uri == "file:///papers/test.pdf"
        assert scr.carrier_type == CarrierType.PEER_REVIEWED_JOURNAL
        assert scr.content_hash == "abc123def456"

    def test_scr_is_frozen(self):
        """SCR is immutable."""
        scr = SymbolCarrierRecord(
            uri="file:///test.pdf",
            carrier_type=CarrierType.PREPRINT,
            content_hash="hash",
        )
        with pytest.raises(Exception):
            scr.uri = "file:///other.pdf"

    def test_scr_initial_formality_from_carrier(self):
        """initial_formality delegates to carrier type."""
        scr = SymbolCarrierRecord(
            uri="https://arxiv.org/abs/2401.12345",
            carrier_type=CarrierType.PREPRINT,
            content_hash="hash",
        )
        assert scr.initial_formality() == 2  # F2

    def test_scr_initial_reliability_from_carrier(self):
        """initial_reliability delegates to carrier type."""
        scr = SymbolCarrierRecord(
            uri="https://example.com/blog/post",
            carrier_type=CarrierType.BLOG_POST,
            content_hash="hash",
        )
        assert scr.initial_reliability() == 0.2

    def test_scr_with_optional_fields(self):
        """Can include title and authors."""
        scr = SymbolCarrierRecord(
            uri="doi:10.1000/xyz123",
            carrier_type=CarrierType.PEER_REVIEWED_JOURNAL,
            content_hash="hash",
            title="A Study of Things",
            authors=["Alice", "Bob"],
        )
        assert scr.title == "A Study of Things"
        assert scr.authors == ["Alice", "Bob"]


class TestUEpistemeFGR:
    """Test F-G-R tuple integration with UEpisteme."""

    def test_default_fgr(self):
        """New epistemes have default F-G-R (F0, universal scope, R=0)."""
        hid = HolonId(context_id="ctx")
        episteme = UEpisteme(holon_id=hid, described_entity="Test")

        assert episteme.fgr.formality == FormalityLevel.F0_INFORMAL
        assert episteme.fgr.reliability == 0.0
        assert not episteme.fgr.claim_scope.contexts  # universal/empty

    def test_with_fgr_returns_new_instance(self):
        """with_fgr returns new episteme, doesn't mutate."""
        hid = HolonId(context_id="ctx")
        original = UEpisteme(holon_id=hid, described_entity="Test")

        new_fgr = FGRTuple(
            formality=FormalityLevel.F3_STRUCTURED,
            claim_scope=ClaimScope(contexts={"ml", "research"}),
            reliability=0.7,
        )
        updated = original.with_fgr(new_fgr)

        assert original.fgr.formality == FormalityLevel.F0_INFORMAL
        assert updated.fgr.formality == FormalityLevel.F3_STRUCTURED
        assert updated.fgr.reliability == 0.7
        assert "ml" in updated.fgr.claim_scope.contexts

    def test_with_reliability_convenience(self):
        """with_reliability updates R only, keeps F and G."""
        hid = HolonId(context_id="ctx")
        fgr = FGRTuple(
            formality=FormalityLevel.F4_TYPED,
            claim_scope=ClaimScope(contexts={"domain"}),
            reliability=0.5,
        )
        original = UEpisteme(holon_id=hid, described_entity="Test", fgr=fgr)

        updated = original.with_reliability(0.8)

        assert updated.fgr.formality == FormalityLevel.F4_TYPED  # unchanged
        assert updated.fgr.claim_scope.contexts == {"domain"}  # unchanged
        assert updated.fgr.reliability == 0.8  # updated

    def test_with_claim_scope_convenience(self):
        """with_claim_scope updates G only, keeps F and R."""
        hid = HolonId(context_id="ctx")
        fgr = FGRTuple(
            formality=FormalityLevel.F5_CONSTRAINED,
            claim_scope=ClaimScope(contexts={"old"}),
            reliability=0.6,
        )
        original = UEpisteme(holon_id=hid, described_entity="Test", fgr=fgr)

        new_scope = ClaimScope(contexts={"new", "scope"})
        updated = original.with_claim_scope(new_scope)

        assert updated.fgr.formality == FormalityLevel.F5_CONSTRAINED  # unchanged
        assert updated.fgr.claim_scope.contexts == {"new", "scope"}  # updated
        assert updated.fgr.reliability == 0.6  # unchanged


class TestUEpistemeSCR:
    """Test Symbol Carrier Register integration with UEpisteme."""

    def test_default_scr_empty(self):
        """New epistemes have no SCR refs."""
        hid = HolonId(context_id="ctx")
        episteme = UEpisteme(holon_id=hid, described_entity="Test")

        assert episteme.scr_refs == []

    def test_with_scr_returns_new_instance(self):
        """with_scr returns new episteme, doesn't mutate."""
        hid = HolonId(context_id="ctx")
        original = UEpisteme(holon_id=hid, described_entity="Test")

        scr = SymbolCarrierRecord(
            uri="file:///paper.pdf",
            carrier_type=CarrierType.PEER_REVIEWED_JOURNAL,
            content_hash="hash123",
        )
        updated = original.with_scr(scr)

        assert len(original.scr_refs) == 0
        assert len(updated.scr_refs) == 1
        assert updated.scr_refs[0].uri == "file:///paper.pdf"

    def test_with_scr_appends(self):
        """with_scr appends to existing refs."""
        hid = HolonId(context_id="ctx")
        scr1 = SymbolCarrierRecord(
            uri="file:///paper1.pdf",
            carrier_type=CarrierType.PREPRINT,
            content_hash="hash1",
        )
        original = UEpisteme(holon_id=hid, described_entity="Test", scr_refs=[scr1])

        scr2 = SymbolCarrierRecord(
            uri="file:///paper2.pdf",
            carrier_type=CarrierType.TECHNICAL_REPORT,
            content_hash="hash2",
        )
        updated = original.with_scr(scr2)

        assert len(updated.scr_refs) == 2
        assert updated.scr_refs[0].uri == "file:///paper1.pdf"
        assert updated.scr_refs[1].uri == "file:///paper2.pdf"

    def test_episteme_with_fgr_and_scr(self):
        """Can create episteme with both F-G-R and SCR initialized."""
        hid = HolonId(context_id="research")
        scr = SymbolCarrierRecord(
            uri="https://arxiv.org/abs/2401.12345",
            carrier_type=CarrierType.PREPRINT,
            content_hash="arxivhash",
            title="Neural Networks",
            authors=["Smith", "Jones"],
        )
        fgr = FGRTuple(
            formality=FormalityLevel(scr.initial_formality()),
            claim_scope=ClaimScope(contexts={"ml", "neural-nets"}),
            reliability=scr.initial_reliability(),
        )
        episteme = UEpisteme(
            holon_id=hid,
            described_entity="Neural network convergence claim",
            fgr=fgr,
            scr_refs=[scr],
        )

        assert episteme.fgr.formality == FormalityLevel.F2_CONTROLLED
        assert episteme.fgr.reliability == 0.4
        assert len(episteme.scr_refs) == 1
        assert episteme.scr_refs[0].title == "Neural Networks"

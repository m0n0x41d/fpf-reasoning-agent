"""
FPF Kernel: Holon types (A.1, A.7, A.15).

Holons are the fundamental unit of FPF — everything is a holon.
UEpisteme is the knowledge artifact type used for research.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .types import (
    AssuranceLevel,
    AssuranceRecord,
    CarrierType,
    HolonId,
    LifecycleState,
    SymbolCarrierRecord,
    TemporalStance,
    TypingAssuranceLevel,
    ValidationAssuranceLevel,
    VerificationAssuranceLevel,
)
from ..trust.fgr import ClaimScope, FGRTuple, FormalityLevel


class StrictDistinctionSlots(BaseModel):
    """
    FPF A.7/A.15: Strict Distinction.

    Every holon separates these orthogonal concerns:
    - structure: what it comprises (composition, parts)
    - order: dependencies, sequences, causation
    - time: versioning, phases, temporal aspects
    - work: effort expended, resources consumed
    - values: objectives, criteria, measures of success

    Conflating slots is a category error. Each slot is optional
    but if present, must contain only that category of information.
    """
    structure: Optional[dict[str, Any]] = None
    order: Optional[dict[str, Any]] = None
    time: Optional[dict[str, Any]] = None
    work: Optional[dict[str, Any]] = None
    values: Optional[dict[str, Any]] = None


HolonType = Literal["episteme", "system", "method", "work"]


class UHolon(BaseModel):
    """
    Base holon type (FPF A.1).

    All FPF entities inherit from this. A holon is a whole that is
    also part of larger wholes — recursive composition.

    Attributes:
        holon_id: Unique identity within bounded context
        holon_type: Discriminator for subtype routing
        slots: Strict Distinction slots (A.7/A.15)
        temporal_stance: Design-time vs run-time (A.4)
        created_at: Creation timestamp
        updated_at: Last modification (edition creation)
    """
    holon_id: HolonId
    holon_type: HolonType

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    slots: StrictDistinctionSlots = Field(default_factory=StrictDistinctionSlots)
    temporal_stance: TemporalStance = TemporalStance.DESIGN_TIME


def _default_fgr() -> FGRTuple:
    """Default F-G-R tuple for new epistemes."""
    return FGRTuple(
        formality=FormalityLevel.F0_INFORMAL,
        claim_scope=ClaimScope(),
        reliability=0.0,
    )


class UEpisteme(UHolon):
    """
    Episteme — unit of knowledge (FPF C.2.1).

    An episteme is a knowledge artifact with:
    - described_entity: What is being described (the subject)
    - claim_graph: The claims and their relationships
    - grounding_holon_id: Connection to physical reality (optional)
    - viewpoint: Perspective from which claims are made

    FPF Compliance:
    - B.3: fgr tuple tracks Formality, claim Scope (G), and Reliability
    - B.5.1: lifecycle_state tracks Explore→Shape→Evidence→Operate
    - B.3.3: assurance_record tracks TA/VA/LA lanes → computed L0/L1/L2
    - A.10: scr_refs tracks symbol carriers (provenance)
    - A.10: evidence_ids link to supporting evidence

    New epistemes start at L0 (untested) in exploration state.
    Progression requires passing gates, not just assertion.
    """
    holon_type: Literal["episteme"] = "episteme"

    described_entity: str
    claim_graph: dict[str, Any] = Field(default_factory=dict)
    grounding_holon_id: Optional[UUID] = None
    viewpoint: Optional[str] = None

    lifecycle_state: LifecycleState = LifecycleState.EXPLORATION

    # FPF B.3: Trust tuple (Formality, claim Scope, Reliability)
    fgr: FGRTuple = Field(default_factory=_default_fgr)

    # FPF A.10: Symbol Carrier Register — provenance tracking
    scr_refs: list[SymbolCarrierRecord] = Field(default_factory=list)

    # FPF B.3.3: Three-lane assurance tracking
    assurance_record: AssuranceRecord = Field(default_factory=AssuranceRecord)

    # Legacy field for backward compatibility — computed from assurance_record
    assurance_level: AssuranceLevel = AssuranceLevel.L0

    evidence_ids: list[UUID] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Sync assurance_level with assurance_record after init."""
        object.__setattr__(
            self, "assurance_level", self.assurance_record.compute_level()
        )

    def with_claim(self, key: str, value: Any) -> UEpisteme:
        """
        Return new episteme with additional claim.

        Does not mutate — returns copy with updated claim_graph.
        """
        new_claims = {**self.claim_graph, key: value}
        return self.model_copy(update={"claim_graph": new_claims})

    def with_evidence(self, evidence_id: UUID) -> UEpisteme:
        """
        Return new episteme with additional evidence reference.

        Does not mutate — returns copy with updated evidence_ids.
        """
        new_evidence = [*self.evidence_ids, evidence_id]
        return self.model_copy(update={"evidence_ids": new_evidence})

    def with_fgr(self, fgr: FGRTuple) -> UEpisteme:
        """
        Return new episteme with updated F-G-R tuple.

        Use this for trust updates. Does not mutate.
        """
        return self.model_copy(
            update={
                "fgr": fgr,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def with_reliability(self, reliability: float) -> UEpisteme:
        """
        Return new episteme with updated reliability only.

        Convenience method — keeps F and G unchanged.
        """
        new_fgr = self.fgr.with_reliability(reliability)
        return self.with_fgr(new_fgr)

    def with_claim_scope(self, scope: ClaimScope) -> UEpisteme:
        """
        Return new episteme with updated claim scope (G).

        Convenience method — keeps F and R unchanged.
        """
        new_fgr = FGRTuple(
            formality=self.fgr.formality,
            claim_scope=scope,
            reliability=self.fgr.reliability,
        )
        return self.with_fgr(new_fgr)

    def with_scr(self, scr: SymbolCarrierRecord) -> UEpisteme:
        """
        Return new episteme with additional symbol carrier reference.

        Tracks provenance per FPF A.10. Does not mutate.
        """
        new_scr_refs = [*self.scr_refs, scr]
        return self.model_copy(
            update={
                "scr_refs": new_scr_refs,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def transition_lifecycle(self, new_state: LifecycleState) -> UEpisteme:
        """
        Return new episteme with updated lifecycle state.

        NOTE: This does not check gates — use LifecycleManager for gated transitions.
        """
        return self.model_copy(
            update={
                "lifecycle_state": new_state,
                "updated_at": datetime.now(timezone.utc)
            }
        )

    def with_typing_assurance(
        self,
        level: TypingAssuranceLevel,
        evidence_id: Optional[UUID] = None,
        rationale: str = "",
    ) -> UEpisteme:
        """Return new episteme with updated typing assurance."""
        new_record = self.assurance_record.with_typing(level, evidence_id, rationale)
        return self.model_copy(
            update={
                "assurance_record": new_record,
                "assurance_level": new_record.compute_level(),
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def with_verification_assurance(
        self,
        level: VerificationAssuranceLevel,
        evidence_id: Optional[UUID] = None,
        rationale: str = "",
    ) -> UEpisteme:
        """Return new episteme with updated verification assurance."""
        new_record = self.assurance_record.with_verification(level, evidence_id, rationale)
        return self.model_copy(
            update={
                "assurance_record": new_record,
                "assurance_level": new_record.compute_level(),
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def with_validation_assurance(
        self,
        level: ValidationAssuranceLevel,
        evidence_id: Optional[UUID] = None,
        rationale: str = "",
    ) -> UEpisteme:
        """Return new episteme with updated validation assurance."""
        new_record = self.assurance_record.with_validation(level, evidence_id, rationale)
        return self.model_copy(
            update={
                "assurance_record": new_record,
                "assurance_level": new_record.compute_level(),
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def elevate_assurance(self, new_level: AssuranceLevel) -> UEpisteme:
        """
        Return new episteme with elevated assurance level.

        DEPRECATED: Use with_typing_assurance, with_verification_assurance,
        or with_validation_assurance instead. This method exists for
        backward compatibility only.
        """
        if new_level.value <= self.assurance_level.value:
            return self
        return self.model_copy(
            update={
                "assurance_level": new_level,
                "updated_at": datetime.now(timezone.utc)
            }
        )


class USystem(UHolon):
    """
    System holon — represents a physical or logical system.

    Used for grounding epistemes to physical reality.
    """
    holon_type: Literal["system"] = "system"

    system_name: str
    system_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    subsystem_ids: list[UUID] = Field(default_factory=list)

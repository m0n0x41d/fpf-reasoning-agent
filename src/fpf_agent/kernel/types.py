"""
FPF Kernel: Core types used across the system.

FPF Compliance:
- A.1: Identity via HolonId
- A.4: Temporal Duality via TemporalStance
- A.7/A.15: Immutable editions (supersedes chain)
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Edition(BaseModel):
    """
    Immutable version identifier for any holon.

    FPF A.4: Temporal Duality requires explicit versioning.
    Editions form a chain via `supersedes` — no mutation, only new editions.
    """
    number: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    supersedes: Optional[UUID] = None

    model_config = {"frozen": True}


class HolonId(BaseModel):
    """
    Unique identifier for a holon within a bounded context.

    FPF A.1: Every holon has identity within its context.
    FPF A.1.1: Meaning is local to context — same UUID in different contexts
               represents different holons.
    """
    id: UUID = Field(default_factory=uuid4)
    context_id: str
    edition: Edition = Field(default_factory=Edition)

    def next_edition(self) -> HolonId:
        """
        Create identifier for next edition.

        Returns new HolonId — old one remains immutable.
        The new edition's `supersedes` points to current `id`.
        """
        return HolonId(
            id=uuid4(),
            context_id=self.context_id,
            edition=Edition(
                number=self.edition.number + 1,
                supersedes=self.id
            )
        )

    def __str__(self) -> str:
        return f"{self.context_id}:{self.id.hex[:8]}@v{self.edition.number}"


class TemporalStance(IntEnum):
    """
    FPF A.4: Temporal Duality.

    Distinguishes design-time (planning, modeling) from run-time (executing, observing).
    Conflating these is a category error per FPF Strict Distinction.

    Ordinal scale — values indicate sequence, not arithmetic.
    """
    DESIGN_TIME = 0
    RUN_TIME = 1


class LifecycleState(IntEnum):
    """
    FPF B.5.1: Artifact Lifecycle States.

    Epistemes progress through these states as evidence accumulates.
    Transitions are gated — see reasoning/lifecycle.py.
    """
    EXPLORATION = 0
    SHAPING = 1
    EVIDENCE = 2
    OPERATE = 3


class AssuranceLevel(IntEnum):
    """
    FPF B.3.3: Assurance Levels.

    L0: Untested claim (default for new epistemes)
    L1: Tested with linked evidence
    L2: Independently verified

    Higher levels require passing gates, not just assertions.
    """
    L0 = 0
    L1 = 1
    L2 = 2


# =============================================================================
# FPF B.3.3: Assurance Subtypes (TA/VA/LA)
# =============================================================================


class TypingAssuranceLevel(IntEnum):
    """
    TA: Typing Assurance — does artifact represent intended concept?

    Measures semantic alignment between artifact and what it claims to describe.
    Based on Congruence Level (CL) between artifact and its referent.

    TA0: No typing evidence (untyped/informal)
    TA1: Informal typing (naming conventions, comments)
    TA2: Schema-validated (passes structural validation)
    TA3: Semantically verified (meaning checked against ontology)
    """
    TA0 = 0
    TA1 = 1
    TA2 = 2
    TA3 = 3

    @property
    def description(self) -> str:
        return {
            0: "No typing evidence",
            1: "Informal typing (naming, comments)",
            2: "Schema-validated structure",
            3: "Semantically verified against ontology",
        }[self.value]


class VerificationAssuranceLevel(IntEnum):
    """
    VA: Verification Assurance — logically correct under assumptions?

    Measures logical soundness of derivations and claims.
    Does NOT check if assumptions hold in reality — that's LA's job.

    VA0: No verification (unverified claim)
    VA1: Peer reviewed / informal check
    VA2: Systematic testing against spec
    VA3: Formal proof or model-checked
    """
    VA0 = 0
    VA1 = 1
    VA2 = 2
    VA3 = 3

    @property
    def description(self) -> str:
        return {
            0: "No verification",
            1: "Peer reviewed / informal check",
            2: "Systematic testing against spec",
            3: "Formal proof or model-checked",
        }[self.value]


class ValidationAssuranceLevel(IntEnum):
    """
    LA: Validation Assurance — works in reality?

    Measures empirical evidence that claims hold in the real world.
    This is the grounding check — does theory match practice?

    LA0: No validation (untested in reality)
    LA1: Anecdotal evidence / case study
    LA2: Systematic empirical testing
    LA3: Independent replication / field-proven
    """
    LA0 = 0
    LA1 = 1
    LA2 = 2
    LA3 = 3

    @property
    def description(self) -> str:
        return {
            0: "No validation",
            1: "Anecdotal evidence / case study",
            2: "Systematic empirical testing",
            3: "Independent replication / field-proven",
        }[self.value]


class AssuranceRecord(BaseModel):
    """
    FPF B.3.3: Complete assurance record with TA/VA/LA lanes.

    Each lane tracks a different kind of evidence:
    - TA: Does it mean what we think? (semantic)
    - VA: Is the logic sound? (formal)
    - LA: Does it work in practice? (empirical)

    The computed_level (L0/L1/L2) is derived from the lane values:
    - L0: No substantive evidence in any lane
    - L1: At least one lane has level >= 2
    - L2: All three lanes have level >= 2
    """
    typing_assurance: TypingAssuranceLevel = TypingAssuranceLevel.TA0
    verification_assurance: VerificationAssuranceLevel = VerificationAssuranceLevel.VA0
    validation_assurance: ValidationAssuranceLevel = ValidationAssuranceLevel.LA0

    # Evidence references for each lane
    typing_evidence_ids: list[UUID] = Field(default_factory=list)
    verification_evidence_ids: list[UUID] = Field(default_factory=list)
    validation_evidence_ids: list[UUID] = Field(default_factory=list)

    # Optional notes explaining the assurance state
    typing_rationale: str = ""
    verification_rationale: str = ""
    validation_rationale: str = ""

    def compute_level(self) -> AssuranceLevel:
        """
        Compute aggregate assurance level from TA/VA/LA.

        L0: max(TA, VA, LA) < 2 — no substantive evidence
        L1: max(TA, VA, LA) >= 2 — at least one lane substantiated
        L2: min(TA, VA, LA) >= 2 — all lanes substantiated
        """
        levels = [
            self.typing_assurance.value,
            self.verification_assurance.value,
            self.validation_assurance.value,
        ]

        if min(levels) >= 2:
            return AssuranceLevel.L2
        if max(levels) >= 2:
            return AssuranceLevel.L1
        return AssuranceLevel.L0

    @property
    def is_fully_assured(self) -> bool:
        """Check if all lanes have substantive evidence (level >= 2)."""
        return self.compute_level() == AssuranceLevel.L2

    @property
    def weakest_lane(self) -> str:
        """Identify the weakest assurance lane."""
        levels = {
            "typing": self.typing_assurance.value,
            "verification": self.verification_assurance.value,
            "validation": self.validation_assurance.value,
        }
        return min(levels, key=levels.get)  # type: ignore

    @property
    def all_evidence_ids(self) -> list[UUID]:
        """Get all evidence IDs across all lanes."""
        return (
            self.typing_evidence_ids
            + self.verification_evidence_ids
            + self.validation_evidence_ids
        )

    def with_typing(
        self,
        level: TypingAssuranceLevel,
        evidence_id: Optional[UUID] = None,
        rationale: str = "",
    ) -> "AssuranceRecord":
        """Return new record with updated typing assurance."""
        new_evidence = list(self.typing_evidence_ids)
        if evidence_id:
            new_evidence.append(evidence_id)
        return self.model_copy(
            update={
                "typing_assurance": level,
                "typing_evidence_ids": new_evidence,
                "typing_rationale": rationale or self.typing_rationale,
            }
        )

    def with_verification(
        self,
        level: VerificationAssuranceLevel,
        evidence_id: Optional[UUID] = None,
        rationale: str = "",
    ) -> "AssuranceRecord":
        """Return new record with updated verification assurance."""
        new_evidence = list(self.verification_evidence_ids)
        if evidence_id:
            new_evidence.append(evidence_id)
        return self.model_copy(
            update={
                "verification_assurance": level,
                "verification_evidence_ids": new_evidence,
                "verification_rationale": rationale or self.verification_rationale,
            }
        )

    def with_validation(
        self,
        level: ValidationAssuranceLevel,
        evidence_id: Optional[UUID] = None,
        rationale: str = "",
    ) -> "AssuranceRecord":
        """Return new record with updated validation assurance."""
        new_evidence = list(self.validation_evidence_ids)
        if evidence_id:
            new_evidence.append(evidence_id)
        return self.model_copy(
            update={
                "validation_assurance": level,
                "validation_evidence_ids": new_evidence,
                "validation_rationale": rationale or self.validation_rationale,
            }
        )

    def summary(self) -> str:
        """Human-readable summary of assurance state."""
        return (
            f"TA{self.typing_assurance.value}/"
            f"VA{self.verification_assurance.value}/"
            f"LA{self.validation_assurance.value} "
            f"→ L{self.compute_level().value}"
        )

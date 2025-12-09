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

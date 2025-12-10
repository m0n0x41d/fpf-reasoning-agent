"""
FPF Kernel: Core type system foundation.

Provides:
- HolonId, Edition: Identity and versioning
- TemporalStance: Design-time vs run-time
- LifecycleState, AssuranceLevel: State tracking
- TA/VA/LA assurance subtypes: Three-lane assurance tracking
- UHolon, UEpisteme: Base and knowledge holons
- UBoundedContext, ContextBridge: Semantic boundaries
"""
from .types import (
    AssuranceLevel,
    AssuranceRecord,
    Edition,
    HolonId,
    LifecycleState,
    TemporalStance,
    TypingAssuranceLevel,
    ValidationAssuranceLevel,
    VerificationAssuranceLevel,
)
from .holons import (
    HolonType,
    StrictDistinctionSlots,
    UEpisteme,
    UHolon,
    USystem,
)
from .bounded_context import (
    ContextBridge,
    UBoundedContext,
)

__all__ = [
    "AssuranceLevel",
    "AssuranceRecord",
    "ContextBridge",
    "Edition",
    "HolonId",
    "HolonType",
    "LifecycleState",
    "StrictDistinctionSlots",
    "TemporalStance",
    "TypingAssuranceLevel",
    "UBoundedContext",
    "UEpisteme",
    "UHolon",
    "USystem",
    "ValidationAssuranceLevel",
    "VerificationAssuranceLevel",
]

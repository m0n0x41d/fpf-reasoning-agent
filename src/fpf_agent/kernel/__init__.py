"""
FPF Kernel: Core type system foundation.

Provides:
- HolonId, Edition: Identity and versioning
- TemporalStance: Design-time vs run-time
- LifecycleState, AssuranceLevel: State tracking
- UHolon, UEpisteme: Base and knowledge holons
- UBoundedContext, ContextBridge: Semantic boundaries
"""
from .types import (
    AssuranceLevel,
    Edition,
    HolonId,
    LifecycleState,
    TemporalStance,
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
    "ContextBridge",
    "Edition",
    "HolonId",
    "HolonType",
    "LifecycleState",
    "StrictDistinctionSlots",
    "TemporalStance",
    "UBoundedContext",
    "UEpisteme",
    "UHolon",
    "USystem",
]

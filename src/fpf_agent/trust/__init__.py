"""
FPF Trust Layer: F-G-R calculus and composition.

Provides:
- FormalityLevel: F0-F9 formality scale
- ClaimScope: Set-valued scope (G)
- CongruenceLevel: CL1-CL5 congruence
- FGRTuple: Complete trust tuple
- TrustCalculus: Composition rules (serial, parallel)
"""
from .fgr import (
    ClaimScope,
    CongruenceLevel,
    FGRTuple,
    FormalityLevel,
)
from .calculus import (
    TrustCalculus,
    check_trust_consistency,
)

__all__ = [
    "ClaimScope",
    "CongruenceLevel",
    "FGRTuple",
    "FormalityLevel",
    "TrustCalculus",
    "check_trust_consistency",
]

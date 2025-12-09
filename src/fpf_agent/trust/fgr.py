"""
FPF B.3: Trust & Assurance Calculus (F-G-R).

Enhanced F-G-R model with Pydantic types and set-theoretic scope operations.

F = Formality (how rigorous the expression)
G = Scope (where the claim is valid)
R = Reliability (how trustworthy, computed from evidence)

Key invariants:
- F and G are invariant under composition
- Only R is affected by CL penalties
- WLNK: aggregate never exceeds weakest component
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field


class FormalityLevel(IntEnum):
    """
    Formality scale (FPF C.2.3).

    How constrained the reasoning is by explicit, proof-grade structure.
    Ordinal scale — levels do NOT admit arithmetic (no averaging!).
    """
    F0_INFORMAL = 0
    F1_NARRATIVE = 1
    F2_CONTROLLED = 2
    F3_STRUCTURED = 3
    F4_TYPED = 4
    F5_CONSTRAINED = 5
    F6_CALCULABLE = 6
    F7_DECIDABLE = 7
    F8_VERIFIED = 8
    F9_CERTIFIED = 9

    @property
    def description(self) -> str:
        """Human-readable description of this formality level."""
        descriptions = {
            0: "Unstructured prose, opinions",
            1: "Structured narrative with sections",
            2: "Controlled vocabulary, defined terms",
            3: "Structured data (JSON, tables)",
            4: "Typed schema with validation",
            5: "Schema + invariants/constraints",
            6: "Executable/calculable rules",
            7: "Decidable formalism",
            8: "Machine-verified proofs",
            9: "Certified/audited formalism",
        }
        return descriptions.get(self.value, "Unknown")


class CongruenceLevel(IntEnum):
    """
    Congruence Level for integration (FPF B.3).

    Measures how well pieces fit together when composed.
    Lower CL = higher penalty on Reliability.
    """
    CL1_INCOMPATIBLE = 1
    CL2_LOOSE = 2
    CL3_PARTIAL = 3
    CL4_ALIGNED = 4
    CL5_EXACT = 5

    def reliability_penalty(self) -> float:
        """
        Compute reliability penalty for this congruence level.

        CL5 = 0% penalty (exact match)
        CL1 = 40% penalty (incompatible)
        """
        return (5 - self.value) * 0.1


class ClaimScope(BaseModel):
    """
    Scope of a claim (G in F-G-R).

    NOT "level of abstraction" but "WHERE the claim is valid".
    FPF A.2.6: Unified Scope Mechanism.

    Scope is set-valued: a claim may apply to multiple contexts
    under specific conditions within temporal bounds.
    """
    contexts: set[str] = Field(default_factory=set)

    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    domain: Optional[str] = None
    subdomain: Optional[str] = None

    preconditions: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}

    def intersect(self, other: ClaimScope) -> ClaimScope:
        """
        Intersection of scopes (for serial dependencies).

        Result is valid only where BOTH are valid.
        Used when composing claims in sequence (A depends on B).
        """
        new_valid_from = None
        if self.valid_from and other.valid_from:
            new_valid_from = max(self.valid_from, other.valid_from)
        elif self.valid_from:
            new_valid_from = self.valid_from
        elif other.valid_from:
            new_valid_from = other.valid_from

        new_valid_until = None
        if self.valid_until and other.valid_until:
            new_valid_until = min(self.valid_until, other.valid_until)
        elif self.valid_until:
            new_valid_until = self.valid_until
        elif other.valid_until:
            new_valid_until = other.valid_until

        new_domain = self.domain if self.domain == other.domain else None

        return ClaimScope(
            contexts=self.contexts & other.contexts,
            valid_from=new_valid_from,
            valid_until=new_valid_until,
            domain=new_domain,
            preconditions=list(set(self.preconditions + other.preconditions)),
        )

    def union(self, other: ClaimScope) -> ClaimScope:
        """
        Union of scopes (for parallel/independent sources).

        Result is valid where EITHER is valid.
        Used when combining independent evidence.
        """
        new_valid_from = None
        if self.valid_from and other.valid_from:
            new_valid_from = min(self.valid_from, other.valid_from)

        new_valid_until = None
        if self.valid_until and other.valid_until:
            new_valid_until = max(self.valid_until, other.valid_until)

        new_domain = self.domain if self.domain == other.domain else None

        return ClaimScope(
            contexts=self.contexts | other.contexts,
            valid_from=new_valid_from,
            valid_until=new_valid_until,
            domain=new_domain,
            preconditions=[p for p in self.preconditions if p in other.preconditions],
        )

    def is_empty(self) -> bool:
        """Check if scope is empty (no valid region)."""
        if self.valid_from and self.valid_until and self.valid_from > self.valid_until:
            return True
        return len(self.contexts) == 0 and self.domain is None

    def contains(self, other: ClaimScope) -> bool:
        """Check if this scope fully contains another."""
        if not other.contexts.issubset(self.contexts):
            return False
        if self.valid_from and other.valid_from and self.valid_from > other.valid_from:
            return False
        if self.valid_until and other.valid_until and self.valid_until < other.valid_until:
            return False
        return True

    def __str__(self) -> str:
        parts = []
        if self.contexts:
            parts.append(f"ctx={{{','.join(sorted(self.contexts))}}}")
        if self.domain:
            d = f"{self.domain}/{self.subdomain}" if self.subdomain else self.domain
            parts.append(f"domain={d}")
        if self.valid_from or self.valid_until:
            f = self.valid_from.isoformat() if self.valid_from else "∞"
            t = self.valid_until.isoformat() if self.valid_until else "∞"
            parts.append(f"time=[{f},{t}]")
        if self.preconditions:
            parts.append(f"pre={len(self.preconditions)}")
        return f"G({'; '.join(parts) or 'universal'})"


class FGRTuple(BaseModel):
    """
    Trust tuple for an episteme (FPF B.3).

    F = Formality (how rigorous)
    G = Claim Scope (where valid)
    R = Reliability (how trustworthy)

    Invariants:
    - F is ordinal, not arithmetic (no averaging)
    - G is set-valued with intersection/union ops
    - R is [0,1], affected by CL penalties
    """
    formality: FormalityLevel
    claim_scope: ClaimScope
    reliability: float = Field(ge=0.0, le=1.0)

    def apply_cl_penalty(self, cl: CongruenceLevel) -> FGRTuple:
        """
        Apply Congruence Level penalty.

        FPF B.3: CL penalties affect R only; F and G are invariant.
        """
        penalty = cl.reliability_penalty()
        new_reliability = max(0.0, self.reliability - penalty)

        return FGRTuple(
            formality=self.formality,
            claim_scope=self.claim_scope,
            reliability=new_reliability
        )

    def with_reliability(self, new_r: float) -> FGRTuple:
        """Return new tuple with updated reliability."""
        return FGRTuple(
            formality=self.formality,
            claim_scope=self.claim_scope,
            reliability=max(0.0, min(1.0, new_r))
        )

    @property
    def is_trustworthy(self) -> bool:
        """Quick check: R >= 0.5 and scope is non-empty."""
        return self.reliability >= 0.5 and not self.claim_scope.is_empty()

    def __str__(self) -> str:
        return f"⟨F{self.formality.value}, {self.claim_scope}, R={self.reliability:.2f}⟩"

    def short_str(self) -> str:
        """Compact representation."""
        return f"F{self.formality.value}/R{self.reliability:.2f}"

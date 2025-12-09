"""
FPF A.1.1: Bounded Context.

All meaning is local to a context. Terms defined in one context
may have different meanings in another. Cross-context translation
requires explicit bridges with Congruence-Loss tracking.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class UBoundedContext(BaseModel):
    """
    Bounded Context — semantic boundary for meaning.

    FPF A.1.1: All meaning is local to a context.

    A bounded context defines:
    - Vocabulary (glossary) with precise definitions
    - Invariants that hold within this context
    - Optional parent for nested contexts

    Terms used in epistemes must be defined in the context's glossary,
    or explicitly bridged from another context with CL penalty.
    """
    context_id: str
    name: str
    description: str

    glossary: dict[str, str] = Field(
        default_factory=dict,
        description="Term definitions local to this context"
    )

    invariants: list[str] = Field(
        default_factory=list,
        description="Statements that hold true within this context"
    )

    parent_context_id: Optional[str] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"

    def define_term(self, term: str, definition: str) -> UBoundedContext:
        """
        Return new context with term added to glossary.

        Does not mutate — returns copy with updated glossary.
        """
        new_glossary = {**self.glossary, term: definition}
        return self.model_copy(update={"glossary": new_glossary})

    def add_invariant(self, invariant: str) -> UBoundedContext:
        """
        Return new context with invariant added.

        Does not mutate — returns copy with updated invariants.
        """
        new_invariants = [*self.invariants, invariant]
        return self.model_copy(update={"invariants": new_invariants})

    def has_term(self, term: str) -> bool:
        """Check if term is defined in this context."""
        return term.lower() in {t.lower() for t in self.glossary}

    def get_definition(self, term: str) -> Optional[str]:
        """Get definition for term, case-insensitive."""
        for t, d in self.glossary.items():
            if t.lower() == term.lower():
                return d
        return None

    def validate_terms(self, terms: list[str]) -> list[str]:
        """
        Return list of terms not defined in this context.

        Empty list means all terms are valid.
        """
        return [t for t in terms if not self.has_term(t)]


class ContextBridge(BaseModel):
    """
    Bridge between two bounded contexts (FPF F.9).

    When translating terms across contexts, congruence loss (CL)
    accumulates. CL affects reliability in F-G-R calculations.

    CL scale (FPF B.3):
    - CL1: Incompatible (explicit conflicts)
    - CL2: Loose correspondence
    - CL3: Partial correspondence
    - CL4: Aligned
    - CL5: Exact correspondence
    """
    source_context_id: str
    target_context_id: str

    term_mappings: dict[str, str] = Field(
        default_factory=dict,
        description="source_term -> target_term mappings"
    )

    congruence_level: int = Field(
        default=3,
        ge=1,
        le=5,
        description="CL1-CL5: how well contexts align"
    )

    notes: Optional[str] = None

    def translate(self, source_term: str) -> Optional[str]:
        """Translate term from source to target context."""
        return self.term_mappings.get(source_term)

    def reliability_penalty(self) -> float:
        """
        Compute reliability penalty for crossing this bridge.

        CL5 = 0% penalty
        CL1 = 40% penalty

        Applied multiplicatively to R in F-G-R.
        """
        return (5 - self.congruence_level) * 0.1

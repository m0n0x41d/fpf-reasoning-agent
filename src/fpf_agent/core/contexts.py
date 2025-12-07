"""
Bounded Context & Bridges (A.1.1, F.9)

Implements FPF's semantic frame model:
- BoundedContext: Local meaning space with invariants and glossary
- BridgeMapping: Cross-context term translation
- Congruence-Loss (CL): Penalty for semantic drift across bridges

FPF principle: "There is no absolute meaning - all terms are context-local."

Key mechanics:
- Every term must be defined in a local glossary
- Cross-context references require explicit bridges
- Bridging incurs CL penalty to reliability
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable


# =============================================================================
# BOUNDED CONTEXT (A.1.1)
# =============================================================================

@dataclass
class TermDefinition:
    """A term with its local definition in a bounded context."""
    term: str
    definition: str
    examples: list[str] = field(default_factory=list)
    related_terms: list[str] = field(default_factory=list)
    formality_level: int = 1  # F-scale of the definition itself
    source: str | None = None  # Where this definition comes from


@dataclass
class Invariant:
    """
    A.1.1: An invariant that holds within this bounded context.

    Invariants are rules that must always be true within the context.
    They constrain what can be validly stated.
    """
    invariant_id: str
    statement: str
    formality_level: int  # How formally is this invariant stated?
    enforcement: str  # "strict" | "advisory" | "informational"
    violation_message: str

    def check(self, claim: str) -> tuple[bool, str]:
        """
        Check if a claim respects this invariant.

        This is a simplified heuristic check. Full FPF would use
        formal logic to verify invariant compliance.
        """
        # Default implementation - subclass for specific invariant types
        return True, "OK"


@dataclass
class BoundedContext:
    """
    A.1.1: A semantic frame where meaning is local.

    The BoundedContext is the fundamental unit of semantic isolation in FPF.
    Terms defined here have LOCAL meaning that may differ from other contexts.

    DDD inspiration: Similar to Domain-Driven Design's Bounded Context.
    """
    context_id: str
    name: str
    description: str

    # Local glossary - terms with their meanings in THIS context
    glossary: dict[str, TermDefinition] = field(default_factory=dict)

    # Invariants that hold within this context
    invariants: list[Invariant] = field(default_factory=list)

    # Parent context (for hierarchical contexts)
    parent_context_id: str | None = None

    # Metadata
    owner: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def add_term(self, term: str, definition: str, **kwargs) -> None:
        """Add a term to the local glossary."""
        self.glossary[term.lower()] = TermDefinition(
            term=term,
            definition=definition,
            **kwargs
        )

    def get_term(self, term: str) -> TermDefinition | None:
        """Look up a term in the local glossary."""
        return self.glossary.get(term.lower())

    def has_term(self, term: str) -> bool:
        """Check if a term is defined in this context."""
        return term.lower() in self.glossary

    def add_invariant(self, invariant: Invariant) -> None:
        """Add an invariant to this context."""
        self.invariants.append(invariant)

    def check_invariants(self, claim: str) -> list[tuple[Invariant, bool, str]]:
        """Check all invariants against a claim."""
        results = []
        for inv in self.invariants:
            passed, message = inv.check(claim)
            results.append((inv, passed, message))
        return results

    def validate_terms_used(self, text: str) -> list[str]:
        """
        Check which terms in text are not defined in glossary.

        Returns list of undefined terms (potential semantic drift).
        """
        # Simple word extraction - in practice would use NLP
        words = set(text.lower().split())

        # Filter to likely domain terms (capitalized in original, etc.)
        # This is a heuristic
        undefined = []
        for word in words:
            # Skip common words
            if len(word) < 4:
                continue
            if word not in self.glossary:
                # Check if it might be a domain term
                if word[0].isupper() or '_' in word:
                    undefined.append(word)

        return undefined


# =============================================================================
# CONTEXT REGISTRY
# =============================================================================

class ContextRegistry:
    """
    Registry of all bounded contexts in the system.

    Provides lookup and validation services.
    """

    def __init__(self):
        self.contexts: dict[str, BoundedContext] = {}
        self._add_default_contexts()

    def _add_default_contexts(self) -> None:
        """Add standard FPF contexts."""
        # FPF Core context
        fpf_core = BoundedContext(
            context_id="fpf_core",
            name="FPF Core",
            description="First Principles Framework core concepts",
        )
        fpf_core.add_term("Holon", "A part-whole: simultaneously a whole in itself and a part of a larger whole")
        fpf_core.add_term("System", "A physical or operational holon that exists in spacetime")
        fpf_core.add_term("Episteme", "A knowledge artifact - theory, model, description")
        fpf_core.add_term("Role", "A contextual responsibility that can be enacted by a holder")
        fpf_core.add_term("Method", "An abstract way of doing something")
        fpf_core.add_term("Work", "An actual execution record of applying a method")
        self.register(fpf_core)

        # General reasoning context
        general = BoundedContext(
            context_id="general",
            name="General Reasoning",
            description="Default context for general-purpose reasoning",
        )
        self.register(general)

    def register(self, context: BoundedContext) -> None:
        """Register a bounded context."""
        self.contexts[context.context_id] = context

    def get(self, context_id: str) -> BoundedContext | None:
        """Get a context by ID."""
        return self.contexts.get(context_id)

    def exists(self, context_id: str) -> bool:
        """Check if context exists."""
        return context_id in self.contexts

    def list_contexts(self) -> list[str]:
        """List all registered context IDs."""
        return list(self.contexts.keys())


# =============================================================================
# BRIDGE MAPPING (F.9)
# =============================================================================

class BridgeType(Enum):
    """Types of cross-context bridges."""
    EQUIVALENT = auto()      # Terms are semantically equivalent
    SPECIALIZATION = auto()  # Target is more specific than source
    GENERALIZATION = auto()  # Target is more general than source
    PARTIAL = auto()         # Partial overlap in meaning
    ANALOGICAL = auto()      # Similar but not same concept


@dataclass
class TermMapping:
    """Mapping of a single term across contexts."""
    source_term: str
    target_term: str
    bridge_type: BridgeType
    congruence_loss: float  # CL: 0 = perfect match, 1 = complete loss
    rationale: str
    examples: list[str] = field(default_factory=list)

    def apply_reliability_penalty(self, source_reliability: float) -> float:
        """Apply CL penalty to reliability when crossing bridge."""
        return source_reliability * (1.0 - self.congruence_loss)


@dataclass
class BridgeMapping:
    """
    F.9: A bridge between two bounded contexts.

    Bridges enable cross-context communication while making
    semantic drift explicit through Congruence-Loss (CL).

    FPF principle: "Crossing context boundaries always costs reliability."
    """
    bridge_id: str
    source_context_id: str
    target_context_id: str
    name: str
    description: str

    # Term mappings
    term_mappings: dict[str, TermMapping] = field(default_factory=dict)

    # Aggregate CL for the whole bridge
    aggregate_cl: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    validated: bool = False
    validator: str | None = None

    def add_mapping(
        self,
        source_term: str,
        target_term: str,
        bridge_type: BridgeType,
        congruence_loss: float,
        rationale: str,
    ) -> None:
        """Add a term mapping to this bridge."""
        self.term_mappings[source_term.lower()] = TermMapping(
            source_term=source_term,
            target_term=target_term,
            bridge_type=bridge_type,
            congruence_loss=congruence_loss,
            rationale=rationale,
        )
        self._update_aggregate_cl()

    def _update_aggregate_cl(self) -> None:
        """Update aggregate CL from all mappings."""
        if not self.term_mappings:
            self.aggregate_cl = 0.0
            return

        # Aggregate CL is max of individual CLs (weakest link)
        self.aggregate_cl = max(m.congruence_loss for m in self.term_mappings.values())

    def translate_term(self, source_term: str) -> tuple[str, float] | None:
        """
        Translate a term from source to target context.

        Returns (translated_term, congruence_loss) or None if not mapped.
        """
        mapping = self.term_mappings.get(source_term.lower())
        if mapping:
            return mapping.target_term, mapping.congruence_loss
        return None

    def get_reliability_penalty(self) -> float:
        """Get reliability penalty factor for crossing this bridge."""
        return 1.0 - self.aggregate_cl

    def apply_to_reliability(self, source_reliability: float) -> float:
        """Apply bridge penalty to a reliability value."""
        return source_reliability * self.get_reliability_penalty()


# =============================================================================
# BRIDGE REGISTRY
# =============================================================================

class BridgeRegistry:
    """Registry of all bridges between contexts."""

    def __init__(self, context_registry: ContextRegistry):
        self.context_registry = context_registry
        self.bridges: dict[str, BridgeMapping] = {}
        self._bridge_index: dict[tuple[str, str], str] = {}  # (src, tgt) -> bridge_id

    def register(self, bridge: BridgeMapping) -> None:
        """Register a bridge."""
        # Validate contexts exist
        if not self.context_registry.exists(bridge.source_context_id):
            raise ValueError(f"Source context '{bridge.source_context_id}' not found")
        if not self.context_registry.exists(bridge.target_context_id):
            raise ValueError(f"Target context '{bridge.target_context_id}' not found")

        self.bridges[bridge.bridge_id] = bridge
        self._bridge_index[(bridge.source_context_id, bridge.target_context_id)] = bridge.bridge_id

    def get(self, bridge_id: str) -> BridgeMapping | None:
        """Get bridge by ID."""
        return self.bridges.get(bridge_id)

    def find_bridge(self, source_id: str, target_id: str) -> BridgeMapping | None:
        """Find bridge between two contexts."""
        bridge_id = self._bridge_index.get((source_id, target_id))
        if bridge_id:
            return self.bridges[bridge_id]
        return None

    def get_bridges_from(self, source_id: str) -> list[BridgeMapping]:
        """Get all bridges from a source context."""
        return [
            b for b in self.bridges.values()
            if b.source_context_id == source_id
        ]

    def get_bridges_to(self, target_id: str) -> list[BridgeMapping]:
        """Get all bridges to a target context."""
        return [
            b for b in self.bridges.values()
            if b.target_context_id == target_id
        ]


# =============================================================================
# CONTEXT TRANSITION TRACKER
# =============================================================================

@dataclass
class ContextTransition:
    """Record of a context transition during reasoning."""
    from_context: str
    to_context: str
    bridge_id: str | None
    congruence_loss: float
    reliability_before: float
    reliability_after: float
    terms_translated: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


class ContextTransitionTracker:
    """
    Tracks context transitions during reasoning.

    Records all context boundary crossings and their cumulative
    impact on reliability.
    """

    def __init__(self, bridge_registry: BridgeRegistry):
        self.bridge_registry = bridge_registry
        self.transitions: list[ContextTransition] = []
        self.current_context_id: str = "general"
        self.cumulative_cl: float = 0.0

    def transition_to(
        self,
        target_context_id: str,
        current_reliability: float,
    ) -> tuple[float, ContextTransition]:
        """
        Transition to a new context.

        Returns (new_reliability, transition_record).
        """
        source_id = self.current_context_id

        # Find bridge
        bridge = self.bridge_registry.find_bridge(source_id, target_context_id)

        if bridge:
            # Apply bridge penalty
            cl = bridge.aggregate_cl
            new_reliability = bridge.apply_to_reliability(current_reliability)
            bridge_id = bridge.bridge_id
            terms = list(bridge.term_mappings.keys())
        else:
            # No explicit bridge - assume high CL (0.3 default)
            cl = 0.3
            new_reliability = current_reliability * (1 - cl)
            bridge_id = None
            terms = []

        # Record transition
        transition = ContextTransition(
            from_context=source_id,
            to_context=target_context_id,
            bridge_id=bridge_id,
            congruence_loss=cl,
            reliability_before=current_reliability,
            reliability_after=new_reliability,
            terms_translated=terms,
        )

        self.transitions.append(transition)
        self.cumulative_cl = min(1.0, self.cumulative_cl + cl)
        self.current_context_id = target_context_id

        return new_reliability, transition

    def get_cumulative_reliability_factor(self) -> float:
        """Get cumulative reliability factor from all transitions."""
        return 1.0 - self.cumulative_cl

    def get_transition_count(self) -> int:
        """Get number of context transitions."""
        return len(self.transitions)

    def get_summary(self) -> str:
        """Get summary of all transitions."""
        if not self.transitions:
            return "No context transitions"

        lines = [f"Context transitions: {len(self.transitions)}"]
        for t in self.transitions:
            bridge_note = f" via {t.bridge_id}" if t.bridge_id else " (no bridge)"
            lines.append(
                f"  {t.from_context} → {t.to_context}{bridge_note} "
                f"(CL={t.congruence_loss:.2f}, R: {t.reliability_before:.2f} → {t.reliability_after:.2f})"
            )
        lines.append(f"Cumulative CL: {self.cumulative_cl:.2f}")
        return "\n".join(lines)


# =============================================================================
# CONTEXT VALIDATION
# =============================================================================

@dataclass
class ContextViolation:
    """A violation of context rules."""
    violation_type: str  # "undefined_term", "invariant_violated", "no_bridge"
    context_id: str
    description: str
    severity: str  # "error", "warning", "info"
    suggestion: str


def validate_text_in_context(
    text: str,
    context: BoundedContext,
) -> list[ContextViolation]:
    """
    Validate that text respects context constraints.

    Checks:
    - Terms used are defined in glossary (or common words)
    - Invariants are not violated
    """
    violations = []

    # Check for undefined terms
    undefined = context.validate_terms_used(text)
    for term in undefined:
        violations.append(ContextViolation(
            violation_type="undefined_term",
            context_id=context.context_id,
            description=f"Term '{term}' not defined in context glossary",
            severity="warning",
            suggestion=f"Define '{term}' in the glossary or use a defined synonym",
        ))

    # Check invariants
    for inv, passed, message in context.check_invariants(text):
        if not passed:
            violations.append(ContextViolation(
                violation_type="invariant_violated",
                context_id=context.context_id,
                description=f"Invariant '{inv.invariant_id}' violated: {message}",
                severity="error" if inv.enforcement == "strict" else "warning",
                suggestion=inv.violation_message,
            ))

    return violations


def validate_cross_context_reference(
    source_context_id: str,
    target_context_id: str,
    term: str,
    bridge_registry: BridgeRegistry,
) -> tuple[bool, str, float]:
    """
    Validate a cross-context term reference.

    Returns (is_valid, message, congruence_loss).
    """
    bridge = bridge_registry.find_bridge(source_context_id, target_context_id)

    if not bridge:
        return (
            False,
            f"No bridge from '{source_context_id}' to '{target_context_id}'",
            1.0  # Maximum CL when no bridge
        )

    result = bridge.translate_term(term)
    if not result:
        return (
            False,
            f"Term '{term}' has no mapping in bridge '{bridge.bridge_id}'",
            bridge.aggregate_cl
        )

    translated, cl = result
    return (
        True,
        f"'{term}' → '{translated}' (CL={cl:.2f})",
        cl
    )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_context_system() -> tuple[ContextRegistry, BridgeRegistry]:
    """Create a complete context management system."""
    context_registry = ContextRegistry()
    bridge_registry = BridgeRegistry(context_registry)
    return context_registry, bridge_registry


def create_engineering_context() -> BoundedContext:
    """Create a sample engineering domain context."""
    ctx = BoundedContext(
        context_id="engineering",
        name="Engineering Domain",
        description="Context for engineering systems and processes",
    )
    ctx.add_term("Component", "A replaceable part of a system with defined interfaces")
    ctx.add_term("Interface", "The boundary through which a component interacts with others")
    ctx.add_term("Requirement", "A documented need that a system must satisfy")
    ctx.add_term("Specification", "A formal description of system properties")
    ctx.add_term("Verification", "Checking that a system meets its specification")
    ctx.add_term("Validation", "Checking that a system meets user needs")

    # Add invariant
    ctx.add_invariant(Invariant(
        invariant_id="eng_001",
        statement="All requirements must be verifiable",
        formality_level=2,
        enforcement="strict",
        violation_message="Requirements must have clear verification criteria",
    ))

    return ctx


def create_fpf_to_engineering_bridge() -> BridgeMapping:
    """Create bridge from FPF core to engineering context."""
    bridge = BridgeMapping(
        bridge_id="fpf_to_engineering",
        source_context_id="fpf_core",
        target_context_id="engineering",
        name="FPF-Engineering Bridge",
        description="Maps FPF concepts to engineering domain terminology",
    )

    bridge.add_mapping(
        source_term="System",
        target_term="Component",
        bridge_type=BridgeType.SPECIALIZATION,
        congruence_loss=0.1,
        rationale="FPF System is broader; engineering Component is more specific",
    )

    bridge.add_mapping(
        source_term="Episteme",
        target_term="Specification",
        bridge_type=BridgeType.PARTIAL,
        congruence_loss=0.2,
        rationale="Episteme includes all knowledge; Specification is formal subset",
    )

    bridge.add_mapping(
        source_term="Method",
        target_term="Process",
        bridge_type=BridgeType.EQUIVALENT,
        congruence_loss=0.05,
        rationale="Nearly equivalent in engineering context",
    )

    return bridge

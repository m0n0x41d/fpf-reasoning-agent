"""
F-G-R Trust Calculus Implementation (B.3)

Core computation functions for FPF trust assessment:
- F (Formality): F0-F9 scale based on rigor of expression
- G (Scope): Set-valued U.ClaimScope — where the claim holds
- R (Reliability): Computed from evidence anchors, with decay

Key FPF principles implemented:
- "Trust is computed, not intuited"
- Weakest-link propagation: trust_aggregate ≤ min(component_trusts)
- Evidence decay (B.3.4): Evidence is perishable
"""

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Annotated

from pydantic import BaseModel, Field


# =============================================================================
# FORMALITY SCALE (F)
# =============================================================================

class FormalityLevel(IntEnum):
    """
    B.3 / C.2.3: Unified Formality Characteristic F.

    The F-scale measures the rigor and precision of a claim's expression.
    Higher F means more formal, verifiable, but also more costly to produce.
    """
    F0_VAGUE_PROSE = 0       # Unstructured, ambiguous natural language
    F1_STRUCTURED_PROSE = 1  # Organized prose with clear sections
    F2_SEMI_FORMAL = 2       # Defined terms, explicit scope
    F3_TYPED_SCHEMA = 3      # Typed data structures, JSON Schema level
    F4_CONSTRAINED_MODEL = 4 # OCL constraints, invariants stated
    F5_FORMAL_SPEC = 5       # Z, VDM, Alloy - formal but not executable
    F6_EXECUTABLE_SPEC = 6   # TLA+, P - executable formal spec
    F7_VERIFIED_MODEL = 7    # Model checking applied
    F8_MACHINE_CHECKED = 8   # Proof assistant (Lean, Coq) - tactics
    F9_MACHINE_VERIFIED = 9  # Fully machine-verified proof


# Heuristics for detecting formality level from text
FORMALITY_INDICATORS = {
    FormalityLevel.F0_VAGUE_PROSE: [],
    FormalityLevel.F1_STRUCTURED_PROSE: [
        "defined as", "means that", "specifically",
    ],
    FormalityLevel.F2_SEMI_FORMAL: [
        "invariant:", "precondition:", "postcondition:",
        "constraint:", "domain:", "range:",
    ],
    FormalityLevel.F3_TYPED_SCHEMA: [
        "type:", "schema:", "interface", "struct",
        "class ", "def ", "function",
    ],
    FormalityLevel.F4_CONSTRAINED_MODEL: [
        "forall", "exists", "implies", "iff",
        "∀", "∃", "→", "↔",
    ],
    FormalityLevel.F5_FORMAL_SPEC: [
        "specification", "formal model", "axiom",
    ],
    FormalityLevel.F6_EXECUTABLE_SPEC: [
        "module", "process", "action", "next",
        "eventually", "always", "temporal",
    ],
    FormalityLevel.F7_VERIFIED_MODEL: [
        "model check", "verified", "counterexample",
    ],
    FormalityLevel.F8_MACHINE_CHECKED: [
        "theorem", "proof", "qed", "lemma",
        "tactic", "induction", "rewrite",
    ],
    FormalityLevel.F9_MACHINE_VERIFIED: [
        "certified", "formally verified", "proof complete",
    ],
}


def compute_formality(text: str, explicit_level: FormalityLevel | None = None) -> FormalityLevel:
    """
    Compute formality level of a claim or statement.

    If explicit_level is provided, use it (author declaration).
    Otherwise, heuristically detect from text.

    In practice, F should be author-declared and reviewer-validated.
    """
    if explicit_level is not None:
        return explicit_level

    text_lower = text.lower()
    detected_level = FormalityLevel.F0_VAGUE_PROSE

    # Check for indicators from low to high, highest match wins
    for level in FormalityLevel:
        indicators = FORMALITY_INDICATORS.get(level, [])
        for indicator in indicators:
            if indicator.lower() in text_lower:
                if level > detected_level:
                    detected_level = level

    return detected_level


# =============================================================================
# SCOPE (G)
# =============================================================================

@dataclass
class ClaimScope:
    """
    A.2.6 / B.3: Set-valued scope describing where a claim holds.

    G is not a single string but a set of applicability conditions.
    A claim's truth is always relative to its declared scope.
    """
    bounded_context: str  # The U.BoundedContext ID
    applicability_conditions: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    temporal_bounds: tuple[str | None, str | None] = field(default=(None, None))

    def contains(self, other: "ClaimScope") -> bool:
        """Check if this scope contains (is broader than) another scope."""
        # Same context required
        if self.bounded_context != other.bounded_context:
            return False

        # Our applicability must be subset (fewer restrictions = broader)
        our_conditions = set(self.applicability_conditions)
        their_conditions = set(other.applicability_conditions)
        if not our_conditions.issubset(their_conditions):
            return False

        return True

    def overlaps(self, other: "ClaimScope") -> bool:
        """Check if scopes overlap (could both apply to same situation)."""
        if self.bounded_context != other.bounded_context:
            return False

        # Check if exclusions contradict
        our_exclusions = set(self.exclusions)
        their_conditions = set(other.applicability_conditions)
        if our_exclusions & their_conditions:
            return False

        return True

    def to_string(self) -> str:
        """Human-readable representation of scope."""
        parts = [f"Context: {self.bounded_context}"]
        if self.applicability_conditions:
            parts.append(f"When: {', '.join(self.applicability_conditions)}")
        if self.exclusions:
            parts.append(f"Except: {', '.join(self.exclusions)}")
        if self.temporal_bounds[0] or self.temporal_bounds[1]:
            start = self.temporal_bounds[0] or "unbounded"
            end = self.temporal_bounds[1] or "unbounded"
            parts.append(f"Time: {start} to {end}")
        return "; ".join(parts)


def compute_scope(
    context_id: str,
    conditions: list[str] | None = None,
    exclusions: list[str] | None = None,
) -> ClaimScope:
    """
    Construct a ClaimScope from components.

    This is a simple constructor; in full FPF, scope would be
    parsed from formal declarations.
    """
    return ClaimScope(
        bounded_context=context_id,
        applicability_conditions=conditions or [],
        exclusions=exclusions or [],
    )


# =============================================================================
# RELIABILITY (R)
# =============================================================================

@dataclass
class EvidenceNode:
    """
    A.10 / B.3: A node in the evidence graph supporting a claim.

    Every claim should be anchored to evidence. R is computed from
    the reliability of supporting evidence.
    """
    evidence_id: str
    evidence_type: str  # "observation", "test_result", "document", "derivation"
    content_summary: str
    source: str
    reliability: float  # Base reliability [0, 1]
    valid_until: datetime | None = None  # B.3.4: Evidence expiry
    timestamp: datetime = field(default_factory=datetime.now)

    def current_reliability(self, at_time: datetime | None = None) -> float:
        """
        Compute current reliability accounting for decay.

        B.3.4: Evidence is perishable. After valid_until, reliability
        decays linearly (simplified model).
        """
        at_time = at_time or datetime.now()

        if self.valid_until is None:
            # Perpetual evidence (rare - mathematical axioms only)
            return self.reliability

        if at_time <= self.valid_until:
            # Within validity period
            return self.reliability

        # After expiry: decay
        days_past = (at_time - self.valid_until).days
        decay_rate = 0.01  # 1% per day past expiry (configurable)
        decayed = self.reliability * (1 - decay_rate * days_past)
        return max(0.0, decayed)


@dataclass
class EvidenceGraph:
    """
    A.10: Collection of evidence nodes supporting claims.

    Tracks evidence, computes aggregate reliability, monitors debt.
    """
    nodes: dict[str, EvidenceNode] = field(default_factory=dict)
    claim_anchors: dict[str, list[str]] = field(default_factory=dict)

    def add_evidence(self, node: EvidenceNode) -> None:
        """Add an evidence node to the graph."""
        self.nodes[node.evidence_id] = node

    def anchor_claim(self, claim_id: str, evidence_ids: list[str]) -> None:
        """Link a claim to its supporting evidence."""
        self.claim_anchors[claim_id] = evidence_ids

    def get_evidence_for_claim(self, claim_id: str) -> list[EvidenceNode]:
        """Get all evidence nodes supporting a claim."""
        evidence_ids = self.claim_anchors.get(claim_id, [])
        return [self.nodes[eid] for eid in evidence_ids if eid in self.nodes]

    def compute_claim_reliability(
        self,
        claim_id: str,
        at_time: datetime | None = None,
    ) -> float:
        """
        Compute reliability for a claim using weakest-link rule.

        B.3: trust_aggregate ≤ min(component_trusts)
        """
        evidence = self.get_evidence_for_claim(claim_id)

        if not evidence:
            return 0.0  # No evidence = no reliability

        reliabilities = [e.current_reliability(at_time) for e in evidence]
        return min(reliabilities)  # Weakest-link

    def compute_epistemic_debt(self, at_time: datetime | None = None) -> float:
        """
        B.3.4: Compute total epistemic debt in the graph.

        ED = sum of (time_past_expiry * decay_rate) for all stale evidence.
        """
        at_time = at_time or datetime.now()
        total_debt = 0.0
        decay_rate_per_day = 1.0  # Configurable

        for node in self.nodes.values():
            if node.valid_until and at_time > node.valid_until:
                days_past = (at_time - node.valid_until).days
                total_debt += decay_rate_per_day * days_past

        return total_debt

    def get_stale_evidence(self, at_time: datetime | None = None) -> list[EvidenceNode]:
        """Get all evidence nodes past their valid_until date."""
        at_time = at_time or datetime.now()
        return [
            node for node in self.nodes.values()
            if node.valid_until and at_time > node.valid_until
        ]


def compute_reliability(
    evidence_nodes: list[EvidenceNode],
    at_time: datetime | None = None,
) -> float:
    """
    Compute reliability from evidence using weakest-link.

    This is a pure function version for use outside EvidenceGraph.
    """
    if not evidence_nodes:
        return 0.0

    reliabilities = [e.current_reliability(at_time) for e in evidence_nodes]
    return min(reliabilities)


# =============================================================================
# ASSURANCE LEVELS (B.3.3)
# =============================================================================

class AssuranceLevel(IntEnum):
    """
    B.3.3: Assurance levels for artifacts.

    L0: Unsubstantiated - claim exists but has no backing
    L1: Partial - some evidence, not complete
    L2: Assured - complete evidence chain, validated
    """
    L0_UNSUBSTANTIATED = 0
    L1_PARTIAL = 1
    L2_ASSURED = 2


class AssuranceSubtype(str):
    """
    B.3.3: What kind of assurance we have.

    TA: Type Assurance - we're talking about the same thing
    VA: Verification Assurance - logically correct
    LA: Logical Assurance - experimentally validated
    """
    TA = "TA"  # Type Assurance
    VA = "VA"  # Verification Assurance
    LA = "LA"  # Logical Assurance


def compute_assurance_level(
    formality: FormalityLevel,
    reliability: float,
    evidence_count: int,
) -> AssuranceLevel:
    """
    Determine assurance level from F-G-R components.

    This is a simplified heuristic. Full FPF would track
    assurance subtypes (TA, VA, LA) separately.
    """
    # No evidence = unsubstantiated
    if evidence_count == 0 or reliability < 0.1:
        return AssuranceLevel.L0_UNSUBSTANTIATED

    # High formality + high reliability + multiple evidence = assured
    if formality >= FormalityLevel.F5_FORMAL_SPEC and reliability >= 0.8:
        return AssuranceLevel.L2_ASSURED

    if formality >= FormalityLevel.F3_TYPED_SCHEMA and reliability >= 0.6:
        return AssuranceLevel.L2_ASSURED

    # Otherwise partial
    return AssuranceLevel.L1_PARTIAL


# =============================================================================
# F-G-R ASSESSMENT (COMBINED)
# =============================================================================

@dataclass
class FGRAssessment:
    """
    Complete F-G-R trust assessment for a claim.

    This is what gets attached to reasoning steps.
    """
    formality: FormalityLevel
    scope: ClaimScope
    reliability: float
    assurance_level: AssuranceLevel

    # Metadata
    evidence_count: int = 0
    epistemic_debt: float = 0.0
    stale_evidence_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "formality": self.formality.value,
            "formality_name": self.formality.name,
            "scope": self.scope.to_string(),
            "reliability": round(self.reliability, 3),
            "assurance_level": self.assurance_level.name,
            "evidence_count": self.evidence_count,
            "epistemic_debt": round(self.epistemic_debt, 2),
            "stale_evidence_count": self.stale_evidence_count,
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"F{self.formality.value} | G:{self.scope.bounded_context} | "
            f"R={self.reliability:.2f} | {self.assurance_level.name}"
        )

    @property
    def is_trustworthy(self) -> bool:
        """Quick check if this assessment indicates trustworthiness."""
        return (
            self.assurance_level >= AssuranceLevel.L1_PARTIAL
            and self.reliability >= 0.5
            and self.stale_evidence_count == 0
        )


def compute_fgr_assessment(
    claim_text: str,
    context_id: str,
    evidence_graph: EvidenceGraph,
    claim_id: str,
    explicit_formality: FormalityLevel | None = None,
    scope_conditions: list[str] | None = None,
) -> FGRAssessment:
    """
    Compute complete F-G-R assessment for a claim.

    This is the main entry point for trust computation.

    Args:
        claim_text: The claim statement
        context_id: The bounded context ID
        evidence_graph: Evidence graph with anchored evidence
        claim_id: ID of the claim in the evidence graph
        explicit_formality: Author-declared formality level
        scope_conditions: Applicability conditions for scope

    Returns:
        Complete FGRAssessment
    """
    # Compute F
    formality = compute_formality(claim_text, explicit_formality)

    # Compute G
    scope = compute_scope(context_id, scope_conditions)

    # Get evidence
    evidence = evidence_graph.get_evidence_for_claim(claim_id)

    # Compute R (weakest-link)
    now = datetime.now()
    reliability = compute_reliability(evidence, now)

    # Check for stale evidence
    stale = [e for e in evidence if e.valid_until and now > e.valid_until]

    # Compute epistemic debt for this claim's evidence
    claim_debt = sum(
        max(0, (now - e.valid_until).days)
        for e in evidence
        if e.valid_until and now > e.valid_until
    )

    # Determine assurance level
    assurance = compute_assurance_level(formality, reliability, len(evidence))

    return FGRAssessment(
        formality=formality,
        scope=scope,
        reliability=reliability,
        assurance_level=assurance,
        evidence_count=len(evidence),
        epistemic_debt=claim_debt,
        stale_evidence_count=len(stale),
    )


# =============================================================================
# CONTEXT BRIDGE RELIABILITY (F.9)
# =============================================================================

def compute_bridge_reliability_penalty(
    source_context: str,
    target_context: str,
    congruence_loss: float,
) -> float:
    """
    F.9: Compute reliability penalty when crossing context boundaries.

    When bridging between bounded contexts, reliability degrades
    proportionally to congruence loss (CL).

    CL=0: Perfect mapping, no penalty
    CL=1: Complete loss of meaning, R→0
    """
    return 1.0 - congruence_loss


def apply_bridge_penalty(
    source_reliability: float,
    congruence_loss: float,
) -> float:
    """
    Apply reliability penalty after crossing a context bridge.

    R_target = R_source * (1 - CL)
    """
    penalty = compute_bridge_reliability_penalty("", "", congruence_loss)
    return source_reliability * penalty

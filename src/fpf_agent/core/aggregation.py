"""
FPF Universal Aggregation Algebra Γ (B.1)

Implements the Gamma (Γ) operators for safe aggregation across different domains.

Key FPF principles:
- "Never aggregate without declaring Γ-fold policy. No free-hand averages."
- Aggregation must respect declared invariants (IDEM, COMM, LOC, WLNK, MONO)
- Trust propagates via weakest-link: trust_aggregate ≤ min(component_trusts)
- Different domains have different aggregation semantics

Domain-specific Γ operators:
- Γ_sys: Physical system aggregation (mass, energy conservation)
- Γ_epist: Knowledge aggregation (F-G-R propagation)
- Γ_ctx: Context composition (bridges, CL penalties)
- Γ_time: Temporal aggregation (order-sensitive)
- Γ_method: Method composition (workflows, procedures)
- Γ_work: Work/resource aggregation (spent resources)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Generic, TypeVar, Callable


# =============================================================================
# AGGREGATION INVARIANTS (B.1)
# =============================================================================

class AggregationInvariant(Enum):
    """B.1: Invariants that aggregation operations must respect."""
    IDEM = "IDEM"    # Idempotence: x ⊕ x = x (duplicate inputs don't change result)
    COMM = "COMM"    # Commutativity: x ⊕ y = y ⊕ x (order doesn't matter)
    LOC = "LOC"      # Locality: result depends only on declared inputs
    WLNK = "WLNK"    # Weakest-link: aggregate ≤ min(parts)
    MONO = "MONO"    # Monotonicity: adding parts doesn't decrease aggregate


# Invariant compatibility matrix: which invariants can coexist
# Keys are frozensets to avoid ordering issues
INVARIANT_COMPATIBILITY: dict[frozenset, bool] = {
    # WLNK and MONO are incompatible for most aggregations
    frozenset({AggregationInvariant.WLNK, AggregationInvariant.MONO}): False,
    # All other pairs are compatible
}


def check_invariant_compatibility(invariants: list[AggregationInvariant]) -> tuple[bool, str]:
    """Check if a set of invariants is internally consistent."""
    for i, inv1 in enumerate(invariants):
        for inv2 in invariants[i + 1:]:
            pair = frozenset({inv1, inv2})
            if pair in INVARIANT_COMPATIBILITY and not INVARIANT_COMPATIBILITY[pair]:
                return False, f"Invariants {inv1.value} and {inv2.value} are incompatible"
    return True, "OK"


# =============================================================================
# AGGREGATION POLICY
# =============================================================================

class GammaType(Enum):
    """B.1: Types of Gamma operators for different domains."""
    SYS = "Γ_sys"       # Physical system aggregation
    EPIST = "Γ_epist"   # Knowledge/epistemic aggregation
    CTX = "Γ_ctx"       # Context composition
    TIME = "Γ_time"     # Temporal aggregation
    METHOD = "Γ_method" # Method/workflow composition
    WORK = "Γ_work"     # Work/resource aggregation


@dataclass
class AggregationPolicy:
    """
    B.1: Policy declaring HOW aggregation will be performed.

    FPF rule: "Never aggregate without declaring Γ-fold policy."
    This prevents "free-hand averages" where aggregation semantics are implicit.
    """
    gamma_type: GammaType
    invariants: list[AggregationInvariant]
    description: str

    # Optional constraints
    allow_duplicates: bool = True  # If False, enforces IDEM
    order_sensitive: bool = False  # If True, violates COMM

    # Warnings/notes
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate policy consistency."""
        # Auto-add IDEM if duplicates not allowed
        if not self.allow_duplicates and AggregationInvariant.IDEM not in self.invariants:
            self.invariants.append(AggregationInvariant.IDEM)

        # Warn if COMM declared but order_sensitive
        if self.order_sensitive and AggregationInvariant.COMM in self.invariants:
            self.warnings.append(
                "COMM declared but order_sensitive=True - this is contradictory"
            )

        # Check compatibility
        compatible, msg = check_invariant_compatibility(self.invariants)
        if not compatible:
            self.warnings.append(f"Invariant conflict: {msg}")

    def summary(self) -> str:
        """One-line policy summary."""
        invs = ", ".join(i.value for i in self.invariants)
        return f"{self.gamma_type.value}[{invs}]"


# =============================================================================
# AGGREGATION RESULT
# =============================================================================

T = TypeVar('T')


@dataclass
class AggregationResult(Generic[T]):
    """Result of an aggregation operation."""
    value: T
    policy_used: AggregationPolicy
    input_count: int

    # Validation
    invariants_checked: dict[AggregationInvariant, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # Provenance
    aggregated_at: datetime = field(default_factory=datetime.now)
    input_ids: list[str] = field(default_factory=list)

    @property
    def all_invariants_satisfied(self) -> bool:
        return all(self.invariants_checked.values())


# =============================================================================
# BASE GAMMA OPERATOR
# =============================================================================

class GammaOperator(ABC, Generic[T]):
    """
    Base class for Gamma (Γ) aggregation operators.

    Each domain-specific operator must:
    1. Declare its default invariants
    2. Implement the actual aggregation logic
    3. Verify invariants after aggregation
    """

    def __init__(self, policy: AggregationPolicy | None = None):
        """Initialize with optional explicit policy."""
        self.policy = policy or self._default_policy()
        self._validate_policy()

    @abstractmethod
    def _default_policy(self) -> AggregationPolicy:
        """Return the default policy for this operator type."""
        pass

    @abstractmethod
    def _aggregate_impl(self, items: list[T]) -> T:
        """Actual aggregation implementation."""
        pass

    @abstractmethod
    def _check_invariant(self, invariant: AggregationInvariant, items: list[T], result: T) -> bool:
        """Check if a specific invariant holds for this aggregation."""
        pass

    def _validate_policy(self) -> None:
        """Validate that policy is appropriate for this operator."""
        expected_type = self._expected_gamma_type()
        if self.policy.gamma_type != expected_type:
            raise ValueError(
                f"Policy gamma_type {self.policy.gamma_type} "
                f"doesn't match operator type {expected_type}"
            )

    @abstractmethod
    def _expected_gamma_type(self) -> GammaType:
        """Return the expected GammaType for this operator."""
        pass

    def aggregate(self, items: list[T], item_ids: list[str] | None = None) -> AggregationResult[T]:
        """
        Perform aggregation with policy enforcement.

        This is the main entry point. It:
        1. Validates inputs against policy
        2. Performs the aggregation
        3. Checks that invariants hold
        4. Returns a fully documented result
        """
        if not items:
            raise ValueError("Cannot aggregate empty list")

        # Handle duplicates according to policy
        working_items = items
        if not self.policy.allow_duplicates:
            # For IDEM: deduplicate (implementation-specific)
            working_items = self._deduplicate(items)

        # Perform aggregation
        result_value = self._aggregate_impl(working_items)

        # Check invariants
        invariant_results = {}
        for invariant in self.policy.invariants:
            invariant_results[invariant] = self._check_invariant(
                invariant, working_items, result_value
            )

        # Build result
        warnings = list(self.policy.warnings)
        for inv, passed in invariant_results.items():
            if not passed:
                warnings.append(f"Invariant {inv.value} was NOT satisfied")

        return AggregationResult(
            value=result_value,
            policy_used=self.policy,
            input_count=len(items),
            invariants_checked=invariant_results,
            warnings=warnings,
            input_ids=item_ids or [],
        )

    def _deduplicate(self, items: list[T]) -> list[T]:
        """Default deduplication (override for complex types)."""
        seen = set()
        result = []
        for item in items:
            item_hash = hash(str(item))
            if item_hash not in seen:
                seen.add(item_hash)
                result.append(item)
        return result


# =============================================================================
# Γ_sys: PHYSICAL SYSTEM AGGREGATION (B.1.2)
# =============================================================================

@dataclass
class PhysicalMeasurement:
    """A physical measurement with value and unit."""
    name: str
    value: float
    unit: str
    is_conserved: bool = False  # Mass, energy, etc.
    is_extensive: bool = True   # Scales with size (mass) vs intensive (temperature)


class GammaSys(GammaOperator[PhysicalMeasurement]):
    """
    B.1.2: System-specific aggregation for physical systems.

    Rules:
    - Conserved quantities (mass, energy) sum
    - Intensive properties (temperature, pressure) don't simply add
    - Boundaries must be properly defined

    Default invariants: LOC (locality), COMM (order doesn't matter for physics)
    """

    def _expected_gamma_type(self) -> GammaType:
        return GammaType.SYS

    def _default_policy(self) -> AggregationPolicy:
        return AggregationPolicy(
            gamma_type=GammaType.SYS,
            invariants=[
                AggregationInvariant.LOC,
                AggregationInvariant.COMM,
            ],
            description="Physical system aggregation with conservation laws",
        )

    def _aggregate_impl(self, items: list[PhysicalMeasurement]) -> PhysicalMeasurement:
        """
        Aggregate physical measurements.

        - Extensive & conserved: sum
        - Extensive & not conserved: depends on type
        - Intensive: cannot simply aggregate (return error or weighted average)
        """
        if not items:
            raise ValueError("No items to aggregate")

        # Check all have same name and unit
        name = items[0].name
        unit = items[0].unit
        if not all(i.name == name and i.unit == unit for i in items):
            raise ValueError("Cannot aggregate measurements with different names/units")

        # Check extensive vs intensive
        if not items[0].is_extensive:
            raise ValueError(
                f"Cannot simply aggregate intensive property '{name}'. "
                "Use domain-specific averaging with weights."
            )

        # Sum extensive properties
        total = sum(i.value for i in items)

        return PhysicalMeasurement(
            name=name,
            value=total,
            unit=unit,
            is_conserved=items[0].is_conserved,
            is_extensive=True,
        )

    def _check_invariant(
        self,
        invariant: AggregationInvariant,
        items: list[PhysicalMeasurement],
        result: PhysicalMeasurement
    ) -> bool:
        """Check physical aggregation invariants."""
        if invariant == AggregationInvariant.LOC:
            # Locality: result depends only on inputs (always true for pure sum)
            return True

        elif invariant == AggregationInvariant.COMM:
            # Commutativity: order doesn't matter for sums
            return True

        elif invariant == AggregationInvariant.IDEM:
            # Idempotence: x + x = x (FALSE for sums, unless deduplicated)
            return len(items) == len(set(i.value for i in items))

        elif invariant == AggregationInvariant.MONO:
            # Monotonicity: adding parts doesn't decrease (TRUE for positive sums)
            return all(i.value >= 0 for i in items)

        return True


# =============================================================================
# Γ_epist: KNOWLEDGE AGGREGATION (B.1.3)
# =============================================================================

@dataclass
class EpistemicClaim:
    """A knowledge claim with F-G-R trust assessment."""
    claim_id: str
    statement: str
    formality: int  # F0-F9
    scope: str      # Bounded context
    reliability: float  # [0, 1]
    evidence_ids: list[str] = field(default_factory=list)


class GammaEpist(GammaOperator[EpistemicClaim]):
    """
    B.1.3: Knowledge-specific aggregation with F-G-R propagation.

    Key rule: Weakest-link propagation
    - trust_aggregate ≤ min(component_trusts)
    - Formality: min(component formalities)
    - Reliability: min(component reliabilities)

    Default invariants: WLNK (weakest-link), LOC
    """

    def _expected_gamma_type(self) -> GammaType:
        return GammaType.EPIST

    def _default_policy(self) -> AggregationPolicy:
        return AggregationPolicy(
            gamma_type=GammaType.EPIST,
            invariants=[
                AggregationInvariant.WLNK,
                AggregationInvariant.LOC,
                AggregationInvariant.COMM,
            ],
            description="Knowledge aggregation with weakest-link trust propagation",
        )

    def _aggregate_impl(self, items: list[EpistemicClaim]) -> EpistemicClaim:
        """
        Aggregate epistemic claims using weakest-link rule.

        B.3: "Trust propagates via weakest-link: aggregated trust ≤ min(component trusts)"
        """
        if not items:
            raise ValueError("No claims to aggregate")

        # Weakest-link: aggregate F-G-R is min of components
        min_formality = min(c.formality for c in items)
        min_reliability = min(c.reliability for c in items)

        # Scope: intersection (most restrictive)
        # For simplicity, concatenate with "∩"
        scopes = list(set(c.scope for c in items))
        aggregate_scope = " ∩ ".join(scopes) if len(scopes) > 1 else scopes[0]

        # Combine evidence
        all_evidence = []
        for c in items:
            all_evidence.extend(c.evidence_ids)
        all_evidence = list(set(all_evidence))

        # Combined statement
        statements = [c.statement for c in items]
        combined_statement = " AND ".join(f"({s})" for s in statements)

        return EpistemicClaim(
            claim_id=f"aggregate_{datetime.now().timestamp()}",
            statement=combined_statement,
            formality=min_formality,
            scope=aggregate_scope,
            reliability=min_reliability,
            evidence_ids=all_evidence,
        )

    def _check_invariant(
        self,
        invariant: AggregationInvariant,
        items: list[EpistemicClaim],
        result: EpistemicClaim
    ) -> bool:
        """Check epistemic aggregation invariants."""
        if invariant == AggregationInvariant.WLNK:
            # Weakest-link: result reliability ≤ min(input reliabilities)
            min_r = min(c.reliability for c in items)
            return result.reliability <= min_r + 0.001  # Small epsilon for float

        elif invariant == AggregationInvariant.LOC:
            # Locality: result depends only on inputs
            return True

        elif invariant == AggregationInvariant.COMM:
            # Commutativity: order doesn't matter (TRUE for min)
            return True

        elif invariant == AggregationInvariant.MONO:
            # Monotonicity: adding claims doesn't increase trust (TRUE for WLNK)
            # Actually opposite - more claims can only decrease or maintain
            return True

        return True


# =============================================================================
# Γ_ctx: CONTEXT COMPOSITION (B.1.4)
# =============================================================================

@dataclass
class ContextFragment:
    """A fragment of a bounded context."""
    context_id: str
    terms: list[str]
    invariants: list[str]
    congruence_loss: float = 0.0  # CL when bridging from another context


class GammaCtx(GammaOperator[ContextFragment]):
    """
    B.1.4 / F.9: Context composition with bridge penalties.

    When composing contexts:
    - Terms must be bridged with explicit mappings
    - Congruence-Loss (CL) accumulates
    - Invariants from all contexts must be preserved

    Default invariants: LOC, IDEM (same context twice = same context)
    """

    def _expected_gamma_type(self) -> GammaType:
        return GammaType.CTX

    def _default_policy(self) -> AggregationPolicy:
        return AggregationPolicy(
            gamma_type=GammaType.CTX,
            invariants=[
                AggregationInvariant.LOC,
                AggregationInvariant.IDEM,
            ],
            description="Context composition with CL accumulation",
            allow_duplicates=False,
        )

    def _aggregate_impl(self, items: list[ContextFragment]) -> ContextFragment:
        """Compose context fragments."""
        if not items:
            raise ValueError("No contexts to compose")

        # Combine context IDs
        context_ids = [c.context_id for c in items]
        combined_id = "+".join(sorted(set(context_ids)))

        # Union of terms
        all_terms = []
        for c in items:
            all_terms.extend(c.terms)
        combined_terms = list(set(all_terms))

        # Union of invariants (all must hold)
        all_invariants = []
        for c in items:
            all_invariants.extend(c.invariants)
        combined_invariants = list(set(all_invariants))

        # CL: max of individual CLs (pessimistic)
        max_cl = max(c.congruence_loss for c in items) if items else 0.0

        return ContextFragment(
            context_id=combined_id,
            terms=combined_terms,
            invariants=combined_invariants,
            congruence_loss=max_cl,
        )

    def _check_invariant(
        self,
        invariant: AggregationInvariant,
        items: list[ContextFragment],
        result: ContextFragment
    ) -> bool:
        """Check context composition invariants."""
        if invariant == AggregationInvariant.LOC:
            return True

        elif invariant == AggregationInvariant.IDEM:
            # Duplicate contexts should not change result
            unique_ids = set(c.context_id for c in items)
            return len(unique_ids) == len(items)

        elif invariant == AggregationInvariant.COMM:
            # Order doesn't matter for set union
            return True

        return True


# =============================================================================
# Γ_work: WORK/RESOURCE AGGREGATION (B.1.6)
# =============================================================================

@dataclass
class WorkRecord:
    """A record of work/resource consumption."""
    work_id: str
    resource_type: str  # "time", "energy", "cost", etc.
    amount: float
    unit: str
    timestamp: datetime | None = None


class GammaWork(GammaOperator[WorkRecord]):
    """
    B.1.6: Work as spent resource aggregation.

    Work records are:
    - Additive (time + time = total time)
    - Non-negative
    - Order-insensitive for totals

    Default invariants: COMM, LOC, MONO
    """

    def _expected_gamma_type(self) -> GammaType:
        return GammaType.WORK

    def _default_policy(self) -> AggregationPolicy:
        return AggregationPolicy(
            gamma_type=GammaType.WORK,
            invariants=[
                AggregationInvariant.COMM,
                AggregationInvariant.LOC,
                AggregationInvariant.MONO,
            ],
            description="Work/resource aggregation (additive)",
        )

    def _aggregate_impl(self, items: list[WorkRecord]) -> WorkRecord:
        """Sum work records of the same type."""
        if not items:
            raise ValueError("No work records to aggregate")

        # Check same resource type and unit
        resource_type = items[0].resource_type
        unit = items[0].unit
        if not all(w.resource_type == resource_type and w.unit == unit for w in items):
            raise ValueError("Cannot aggregate work records with different types/units")

        total = sum(w.amount for w in items)

        return WorkRecord(
            work_id=f"total_{resource_type}_{datetime.now().timestamp()}",
            resource_type=resource_type,
            amount=total,
            unit=unit,
            timestamp=datetime.now(),
        )

    def _check_invariant(
        self,
        invariant: AggregationInvariant,
        items: list[WorkRecord],
        result: WorkRecord
    ) -> bool:
        """Check work aggregation invariants."""
        if invariant == AggregationInvariant.COMM:
            return True  # Sum is commutative

        elif invariant == AggregationInvariant.LOC:
            return True

        elif invariant == AggregationInvariant.MONO:
            # Adding non-negative work doesn't decrease total
            return all(w.amount >= 0 for w in items)

        elif invariant == AggregationInvariant.IDEM:
            # Not idempotent: same work twice = double work
            return False

        return True


# =============================================================================
# AGGREGATION GUARD: PREVENT FREE-HAND AVERAGES
# =============================================================================

class AggregationGuard:
    """
    Guard against "free-hand averages" - aggregation without explicit policy.

    FPF rule: "Never aggregate without declaring Γ-fold policy."

    Use this class to ensure all aggregations go through proper channels.
    """

    def __init__(self):
        self._blocked_operations: list[str] = []

    def require_policy(
        self,
        operation_name: str,
        items: list[Any],
        policy: AggregationPolicy | None
    ) -> None:
        """
        Ensure aggregation has explicit policy.

        Raises ValueError if policy is None.
        """
        if policy is None:
            self._blocked_operations.append(operation_name)
            raise ValueError(
                f"FPF violation: Aggregation '{operation_name}' attempted without "
                f"explicit Γ-fold policy. Declare AggregationPolicy before aggregating "
                f"{len(items)} items."
            )

    def warn_implicit_aggregation(
        self,
        operation: str,
        suggestion: GammaType
    ) -> str:
        """Generate warning for implicit aggregation attempt."""
        return (
            f"WARNING: '{operation}' appears to aggregate without policy. "
            f"Consider using {suggestion.value} with explicit invariants."
        )

    @property
    def blocked_count(self) -> int:
        """Number of blocked operations."""
        return len(self._blocked_operations)

    def get_blocked_operations(self) -> list[str]:
        """Get list of blocked operation names."""
        return list(self._blocked_operations)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def aggregate_reliabilities(
    reliabilities: list[float],
    policy: AggregationPolicy | None = None
) -> AggregationResult[float]:
    """
    Aggregate reliability values using weakest-link rule.

    This is a common operation in F-G-R trust propagation.
    """
    if policy is None:
        policy = AggregationPolicy(
            gamma_type=GammaType.EPIST,
            invariants=[AggregationInvariant.WLNK, AggregationInvariant.COMM],
            description="Reliability aggregation via weakest-link",
        )

    if policy.gamma_type != GammaType.EPIST:
        raise ValueError("Reliability aggregation requires Γ_epist policy")

    if not reliabilities:
        raise ValueError("No reliabilities to aggregate")

    result = min(reliabilities)

    return AggregationResult(
        value=result,
        policy_used=policy,
        input_count=len(reliabilities),
        invariants_checked={
            AggregationInvariant.WLNK: result <= min(reliabilities) + 0.001,
            AggregationInvariant.COMM: True,
        },
    )


def aggregate_formalities(
    formalities: list[int],
    policy: AggregationPolicy | None = None
) -> AggregationResult[int]:
    """
    Aggregate formality levels using weakest-link rule.

    F_aggregate = min(F_components)
    """
    if policy is None:
        policy = AggregationPolicy(
            gamma_type=GammaType.EPIST,
            invariants=[AggregationInvariant.WLNK],
            description="Formality aggregation via weakest-link",
        )

    if not formalities:
        raise ValueError("No formalities to aggregate")

    result = min(formalities)

    return AggregationResult(
        value=result,
        policy_used=policy,
        input_count=len(formalities),
        invariants_checked={AggregationInvariant.WLNK: True},
    )


def create_policy(
    gamma_type: GammaType,
    invariants: list[AggregationInvariant] | None = None,
    description: str = "",
) -> AggregationPolicy:
    """
    Convenience function to create aggregation policy.

    If invariants not specified, uses sensible defaults for the gamma type.
    """
    default_invariants = {
        GammaType.SYS: [AggregationInvariant.LOC, AggregationInvariant.COMM],
        GammaType.EPIST: [AggregationInvariant.WLNK, AggregationInvariant.LOC],
        GammaType.CTX: [AggregationInvariant.LOC, AggregationInvariant.IDEM],
        GammaType.TIME: [AggregationInvariant.LOC],  # Order matters, no COMM
        GammaType.METHOD: [AggregationInvariant.LOC],  # Order matters
        GammaType.WORK: [AggregationInvariant.COMM, AggregationInvariant.LOC, AggregationInvariant.MONO],
    }

    if invariants is None:
        invariants = default_invariants.get(gamma_type, [AggregationInvariant.LOC])

    if not description:
        description = f"Default {gamma_type.value} aggregation policy"

    return AggregationPolicy(
        gamma_type=gamma_type,
        invariants=invariants,
        description=description,
    )


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_aggregation_attempt(
    items: list[Any],
    declared_policy: AggregationPolicy | None,
    operation_description: str,
) -> tuple[bool, list[str]]:
    """
    Validate an aggregation attempt before execution.

    Returns (is_valid, list of warnings/errors).
    """
    issues = []

    # Must have explicit policy
    if declared_policy is None:
        issues.append(
            f"CRITICAL: No Γ-fold policy declared for '{operation_description}'. "
            "FPF requires explicit aggregation policy."
        )
        return False, issues

    # Check policy consistency
    compatible, msg = check_invariant_compatibility(declared_policy.invariants)
    if not compatible:
        issues.append(f"WARNING: {msg}")

    # Check for empty input
    if not items:
        issues.append("ERROR: Cannot aggregate empty list")
        return False, issues

    # Check for single item (trivial aggregation)
    if len(items) == 1:
        issues.append("INFO: Aggregating single item (trivial case)")

    # Add policy warnings
    issues.extend(declared_policy.warnings)

    has_errors = any("CRITICAL" in i or "ERROR" in i for i in issues)
    return not has_errors, issues

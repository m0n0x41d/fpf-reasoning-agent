"""
Extended FPF Pydantic Schemas — Comprehensive type coverage for FPF concepts.

These schemas encode the FPF ontology for structured reasoning.
Use with Schema-Guided Reasoning (SGR) for type-safe FPF reasoning.
"""

from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# PART A: KERNEL ARCHITECTURE
# =============================================================================

class HolonType(str, Enum):
    """A.1: Core holon types — everything is simultaneously whole and part."""
    SYSTEM = "System"           # Physical/operational entity
    EPISTEME = "Episteme"       # Knowledge artifact
    ROLE = "Role"               # Contextual responsibility
    METHOD = "Method"           # Abstract way of doing
    WORK = "Work"               # Execution record
    SERVICE = "Service"         # External promise
    COMMUNITY = "Community"     # Collective of agents


class Holon(BaseModel):
    """A.1: Holonic Foundation — entity that is both whole and part."""
    holon_type: HolonType
    identifier: str = Field(description="Unique identifier within context")
    description: str = Field(description="What this holon represents")
    boundary_description: str = Field(description="What defines the boundary")
    parent_holon: str | None = Field(default=None, description="Parent in holarchy")
    sub_holons: list[str] = Field(default_factory=list, description="Child holons")


class StrictDistinction(BaseModel):
    """A.7: The eight core categorical distinctions that must never be conflated."""
    category_a: str
    category_b: str
    violation_risk: str = Field(description="What goes wrong if conflated")


# The core distinctions - reference for validation
STRICT_DISTINCTIONS = [
    StrictDistinction(
        category_a="System", category_b="Episteme",
        violation_risk="Confusing physical thing with knowledge about it"
    ),
    StrictDistinction(
        category_a="Role", category_b="Holder",
        violation_risk="Confusing function with entity filling it"
    ),
    StrictDistinction(
        category_a="Method", category_b="MethodDescription",
        violation_risk="Confusing abstract way with recipe document"
    ),
    StrictDistinction(
        category_a="MethodDescription", category_b="Work",
        violation_risk="Confusing recipe with execution record"
    ),
    StrictDistinction(
        category_a="Design-time", category_b="Run-time",
        violation_risk="Creating design-run chimeras"
    ),
    StrictDistinction(
        category_a="Object", category_b="Description",
        violation_risk="Confusing the thing with claims about it"
    ),
    StrictDistinction(
        category_a="Intension", category_b="Extension",
        violation_risk="Confusing definition with set of instances"
    ),
    StrictDistinction(
        category_a="Context", category_b="Content",
        violation_risk="Confusing frame of meaning with claims within"
    ),
]


class CharacteristicType(str, Enum):
    """A.17-A.19: Types of characteristics (measurements) in FPF."""
    ORDINAL = "ordinal"         # Ranked but not measurable distance
    INTERVAL = "interval"       # Measurable distance, no true zero
    RATIO = "ratio"             # True zero, meaningful ratios


class Characteristic(BaseModel):
    """A.17-A.19: A measurable property of a holon."""
    name: str
    char_type: CharacteristicType
    unit: str | None = Field(default=None, description="Unit of measurement if applicable")
    value: float | str | None = None
    scale_min: float | None = None
    scale_max: float | None = None


# =============================================================================
# PART B: TRANS-DISCIPLINARY REASONING
# =============================================================================

class AggregationInvariant(str, Enum):
    """B.1: Invariants for safe aggregation (Γ algebra)."""
    IDEM = "IDEM"       # Idempotence: x + x = x
    COMM = "COMM"       # Commutativity: x + y = y + x
    LOC = "LOC"         # Locality: result depends only on declared inputs
    WLNK = "WLNK"       # Weakest-link: aggregate ≤ min(parts)
    MONO = "MONO"       # Monotonicity: adding parts doesn't decrease


class AggregationPolicy(BaseModel):
    """B.1: Policy for how to aggregate holons."""
    gamma_type: Literal["Γ_sys", "Γ_epist", "Γ_ctx", "Γ_time", "Γ_method", "Γ_work"]
    invariants: list[AggregationInvariant]
    description: str
    warnings: list[str] = Field(default_factory=list)


class MetaHolonTransitionType(str, Enum):
    """B.2: Types of emergence/transition."""
    MST = "MST"  # Meta-System Transition (physical emergence)
    MET = "MET"  # Meta-Epistemic Transition (knowledge emergence)
    MFT = "MFT"  # Meta-Functional Transition (capability emergence)


class MetaHolonTransition(BaseModel):
    """B.2: Recognition of emergence — when a collection becomes a new whole."""
    transition_type: MetaHolonTransitionType
    source_holons: list[str]
    emergent_holon: str
    bosc_trigger: str = Field(description="Boundary-Objective-Supervisor-Complexity trigger")
    evidence: str = Field(description="Why this constitutes genuine emergence")


# =============================================================================
# F-G-R TRUST MODEL (B.3) — Extended
# =============================================================================

class FormalityLevel(int, Enum):
    """B.3: Formality scale F0-F9."""
    F0_VAGUE_PROSE = 0
    F1_STRUCTURED_PROSE = 1
    F2_SEMI_FORMAL = 2
    F3_TYPED_SCHEMA = 3
    F4_CONSTRAINED_MODEL = 4
    F5_FORMAL_SPEC = 5
    F6_EXECUTABLE_SPEC = 6
    F7_VERIFIED_MODEL = 7
    F8_MACHINE_CHECKED = 8
    F9_MACHINE_VERIFIED_PROOF = 9


class AssuranceLevel(str, Enum):
    """B.3.3: Assurance levels for claims."""
    L0_UNSUBSTANTIATED = "L0_unsubstantiated"
    L1_PARTIAL = "L1_partial"
    L2_ASSURED = "L2_assured"


class AssuranceSubtype(str, Enum):
    """B.3.3: Types of assurance."""
    TA = "TA"  # Type Assurance (syntactic correctness)
    VA = "VA"  # Validation Assurance (fit for purpose)
    LA = "LA"  # Logical Assurance (deductive soundness)


class EvidenceNode(BaseModel):
    """A.10: A node in the evidence graph."""
    evidence_id: str
    evidence_type: Literal["observation", "test_result", "document", "testimony", "derivation"]
    content_summary: str
    source: str
    timestamp: str | None = None
    reliability: Annotated[float, Field(ge=0, le=1)]


class Claim(BaseModel):
    """B.3: A claim with F-G-R trust assessment."""
    claim_id: str
    statement: str
    formality: FormalityLevel
    scope: str = Field(description="Bounded context + applicability conditions (G)")
    reliability: Annotated[float, Field(ge=0, le=1)]
    assurance_level: AssuranceLevel
    assurance_subtypes: list[AssuranceSubtype] = Field(default_factory=list)
    evidence_anchors: list[str] = Field(description="Evidence IDs supporting this claim")
    decay_rate: float | None = Field(default=None, description="How fast reliability degrades")


class EpistemicDebt(BaseModel):
    """B.3.4: Tracking of unvalidated assumptions."""
    debt_id: str
    description: str = Field(description="What assumption is unvalidated")
    created_at: str
    severity: Literal["low", "medium", "high", "critical"]
    mitigation_plan: str | None = None


# =============================================================================
# CANONICAL REASONING CYCLE (B.5) — Extended
# =============================================================================

class ReasoningPhaseType(str, Enum):
    """B.5: Phases of the ADI reasoning cycle."""
    ABDUCTION = "Abduction"   # Generate hypotheses
    DEDUCTION = "Deduction"   # Derive consequences
    INDUCTION = "Induction"   # Test against evidence


class ArtifactLifecycleState(str, Enum):
    """B.5.1: States in the artifact lifecycle."""
    EXPLORATION = "Exploration"  # Generating candidates
    SHAPING = "Shaping"          # Refining into concrete form
    EVIDENCE = "Evidence"        # Testing, validating
    OPERATION = "Operation"      # Using in production


class Hypothesis(BaseModel):
    """B.5.2: A candidate explanation generated by abduction."""
    hypothesis_id: str
    statement: str
    anomaly_explained: str = Field(description="What anomaly this addresses")
    plausibility_score: Annotated[float, Field(ge=0, le=1)]
    plausibility_rationale: str
    testable_predictions: list[str]
    competing_hypotheses: list[str] = Field(default_factory=list)


class AbductiveLoopResult(BaseModel):
    """B.5.2: Result of abductive hypothesis generation."""
    anomaly: str
    hypotheses_generated: list[Hypothesis]
    selection_criteria: str
    selected_hypothesis: str | None


# =============================================================================
# CREATIVITY PATTERNS (C.17-C.19)
# =============================================================================

class CreativityScore(BaseModel):
    """C.17: Scoring creative output on Novelty-Quality-Diversity dimensions."""
    novelty: Annotated[float, Field(ge=0, le=1, description="How new is this?")]
    use_value: Annotated[float, Field(ge=0, le=1, description="How useful is this?")]
    surprise: Annotated[float, Field(ge=0, le=1, description="How unexpected is this?")]
    constraint_fit: Annotated[float, Field(ge=0, le=1, description="How well does it fit constraints?")]


class NQDSearchConfig(BaseModel):
    """C.18: Configuration for Novelty-Quality-Diversity search."""
    exploration_budget: int = Field(description="How many candidates to generate")
    diversity_weight: Annotated[float, Field(ge=0, le=1)] = 0.3
    quality_threshold: Annotated[float, Field(ge=0, le=1)] = 0.5
    novelty_threshold: Annotated[float, Field(ge=0, le=1)] = 0.3
    pareto_only: bool = Field(default=True, description="Only keep Pareto-optimal candidates")


class ExploreExploitPolicy(BaseModel):
    """C.19: Explore-Exploit governance policy."""
    exploration_share: Annotated[float, Field(ge=0, le=1)] = Field(
        description="Fraction of effort on exploration vs exploitation"
    )
    switch_criteria: str = Field(description="When to switch from explore to exploit")
    current_phase: Literal["explore", "exploit", "balanced"]
    rationale: str


class ScalingLawLens(BaseModel):
    """C.18.1: Scale-aware reasoning (Bitter Lesson integration)."""
    scale_variables: list[str] = Field(description="Variables that govern scaling")
    expected_elasticities: dict[str, float] = Field(description="How output scales with each variable")
    scale_regime: Literal["linear", "sublinear", "diminishing_returns", "unknown"]
    general_method_preference: bool = Field(
        default=True,
        description="BLP: Prefer general scalable methods over domain-specific heuristics"
    )


# =============================================================================
# EVOLUTION LOOP (B.4)
# =============================================================================

class EvolutionStep(BaseModel):
    """B.4: A step in the canonical evolution loop."""
    step_type: Literal["observe", "refine", "deploy", "monitor"]
    description: str
    inputs: list[str]
    outputs: list[str]
    success_criteria: str


class DesignRationaleRecord(BaseModel):
    """E.9: Record of why a decision was made."""
    drr_id: str
    decision: str
    context: str = Field(description="What situation led to this decision")
    options_considered: list[str]
    selected_option: str
    rationale: str = Field(description="Why this option was chosen")
    consequences: list[str]
    trade_offs: list[str]
    timestamp: str


# =============================================================================
# BRIDGES AND CROSS-CONTEXT (F.9)
# =============================================================================

class CongruenceLossType(str, Enum):
    """F.9: Types of information loss when crossing contexts."""
    SEMANTIC = "semantic"       # Meaning changes
    PRECISION = "precision"     # Detail is lost
    SCOPE = "scope"             # Applicability narrows
    TRUST = "trust"             # Reliability degrades


class BridgeMapping(BaseModel):
    """F.9: Mapping between concepts across bounded contexts."""
    source_context: str
    target_context: str
    source_term: str
    target_term: str
    relation: Literal["equivalent", "subsumes", "subsumed_by", "overlaps", "disjoint"]
    congruence_loss: list[CongruenceLossType] = Field(default_factory=list)
    reliability_penalty: Annotated[float, Field(ge=0, le=1)] = 0.0


# =============================================================================
# CG-SPEC (G.0) — Comparability
# =============================================================================

class ComparabilityMode(str, Enum):
    """G.0: How things can be compared."""
    IDENTICAL = "identical"     # Same measurement frame
    CONGRUENT = "congruent"     # Mappable with known loss
    INCOMPARABLE = "incomparable"  # Cannot be meaningfully compared


class CGSpec(BaseModel):
    """G.0: Comparability-Gauge Specification."""
    cg_id: str
    description: str
    reference_plane: str = Field(description="The common ground for comparison")
    characteristics_measured: list[str]
    comparability_mode: ComparabilityMode
    normalization_rules: list[str] = Field(default_factory=list)


# =============================================================================
# EXTENDED REASONING CONTEXT
# =============================================================================

class FPFReasoningContext(BaseModel):
    """Extended context for FPF reasoning with all relevant type information."""
    object_of_talk: Holon
    bounded_context_id: str
    bounded_context_terms: list[str] = Field(default_factory=list)
    temporal_stance: Literal["design_time", "run_time"]
    active_claims: list[Claim] = Field(default_factory=list)
    evidence_graph: list[EvidenceNode] = Field(default_factory=list)
    epistemic_debts: list[EpistemicDebt] = Field(default_factory=list)
    current_phase: ReasoningPhaseType
    artifact_state: ArtifactLifecycleState
    active_hypotheses: list[Hypothesis] = Field(default_factory=list)
    explore_exploit_policy: ExploreExploitPolicy | None = None
    aggregation_policy: AggregationPolicy | None = None
    cg_spec: CGSpec | None = None


class FPFViolation(BaseModel):
    """A detected violation of FPF principles."""
    violation_type: Literal[
        "strict_distinction", "context_leak", "untraceable_claim",
        "naive_aggregation", "design_run_chimera", "trust_inflation",
        "premature_convergence", "missing_evidence", "scope_creep"
    ]
    description: str
    severity: Literal["warning", "error", "critical"]
    fpf_pattern: str = Field(description="Which FPF pattern is violated")
    remediation: str = Field(description="How to fix this")


class FPFReasoningValidation(BaseModel):
    """Result of validating reasoning against FPF principles."""
    is_valid: bool
    violations: list[FPFViolation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fpf_coverage: dict[str, bool] = Field(
        default_factory=dict,
        description="Which FPF principles were applied"
    )

"""
FPF Integration Layer

Integrates all FPF operational components into a cohesive reasoning pipeline.

This module wires together:
- F-G-R Trust Calculus (fgr.py)
- ADI Cycle Controller (adi_cycle.py)
- Strict Distinction Validator (validator.py)
- Bounded Context Registry (contexts.py)
- Γ Aggregation Operators (aggregation.py)
- Creativity Patterns (creativity.py)

The FPFReasoningPipeline class provides the main entry point for
FPF-guided reasoning with full operational mechanics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import logging

from .schemas import (
    FPFReasoningStep,
    FPFResponse,
    ObjectOfTalk,
    BoundedContext,
    TemporalStance,
    FGRAssessment,
    EvidenceReference,
)
from .fgr import (
    FormalityLevel,
    EvidenceNode,
    EvidenceGraph,
    ClaimScope,
    compute_fgr_assessment,
    compute_formality,
)
from .adi_cycle import (
    ReasoningPhase,
    ArtifactLifecycle,
    ADICycleController,
    Hypothesis,
    DeductionResult,
    InductionResult,
)
from .validator import (
    FPFReasoningValidator,
    ValidationResult,
    ViolationSeverity,
    FPFViolation,
)
from .contexts import (
    BoundedContext as CtxBoundedContext,
    ContextRegistry,
    BridgeMapping,
    ContextTransitionTracker,
    TermDefinition,
)
from .aggregation import (
    AggregationGuard,
    AggregationPolicy,
    GammaType,
    AggregationInvariant,
    GammaEpist,
    aggregate_reliabilities,
)
from .creativity import (
    NQDSearchConfig,
    NQDSearchEngine,
    HypothesisNQDScorer,
    HypothesisCandidate,
    ExploreExploitPolicy,
    ExploreExploitController,
    ExploreExploitPhase,
)


logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE STATE
# =============================================================================

@dataclass
class PipelineState:
    """Complete state of an FPF reasoning pipeline execution."""
    session_id: str
    started_at: datetime = field(default_factory=datetime.now)

    # ADI Cycle
    current_phase: ReasoningPhase = ReasoningPhase.ABDUCTION
    artifact_state: ArtifactLifecycle = ArtifactLifecycle.EXPLORATION
    cycle_count: int = 0

    # Context
    current_context_id: str = "default"
    context_transitions: list[dict[str, Any]] = field(default_factory=list)

    # Evidence and Trust
    evidence_count: int = 0
    average_reliability: float = 0.0
    epistemic_debt: float = 0.0

    # Validation
    total_violations: int = 0
    critical_violations: int = 0
    fpf_coverage: dict[str, bool] = field(default_factory=dict)

    # Creativity (for abduction)
    hypotheses_generated: int = 0
    ee_phase: str = "explore"

    # Steps
    step_count: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepResult:
    """Result of processing a single reasoning step."""
    step_number: int
    validation: ValidationResult
    adi_phase: ReasoningPhase
    artifact_state: ArtifactLifecycle

    # Computed values
    computed_fgr: FGRAssessment | None = None
    context_penalty: float = 0.0  # CL penalty if context changed
    gate_passed: bool = True
    gate_message: str = ""

    # Warnings/notes
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# FPF REASONING PIPELINE
# =============================================================================

class FPFReasoningPipeline:
    """
    Main integration class for FPF-guided reasoning.

    Orchestrates all FPF operational components:
    - Validates each step against FPF principles
    - Manages ADI cycle with proper gates
    - Tracks evidence and computes F-G-R
    - Manages bounded contexts with bridges
    - Enforces Γ aggregation policies
    - Supports NQD creativity during abduction

    Usage:
        pipeline = FPFReasoningPipeline()
        result = pipeline.process_step(step_dict)
        final = pipeline.finalize()
    """

    def __init__(
        self,
        session_id: str | None = None,
        max_adi_cycles: int = 3,
        strict_mode: bool = False,
    ):
        """
        Initialize the reasoning pipeline.

        Args:
            session_id: Unique session identifier
            max_adi_cycles: Maximum ADI cycles before forcing completion
            strict_mode: If True, fail on any validation error
        """
        self.session_id = session_id or f"fpf_{datetime.now().timestamp()}"
        self.strict_mode = strict_mode

        # Initialize components
        self.adi_controller = ADICycleController(max_cycles=max_adi_cycles)
        self.validator = FPFReasoningValidator()
        self.evidence_graph = EvidenceGraph()
        self.context_registry = ContextRegistry()
        self.context_tracker = ContextTransitionTracker(self.context_registry)
        self.aggregation_guard = AggregationGuard()

        # Creativity (for abduction phase)
        self.nqd_config = NQDSearchConfig()
        self.nqd_engine: NQDSearchEngine | None = None
        self.ee_controller: ExploreExploitController | None = None

        # State
        self.state = PipelineState(session_id=self.session_id)
        self._step_results: list[StepResult] = []

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def process_step(self, step: dict[str, Any]) -> StepResult:
        """
        Process a single reasoning step through the FPF pipeline.

        This is the main method called for each reasoning step.
        It validates, updates ADI state, computes F-G-R, and returns results.
        """
        self.state.step_count += 1
        step_num = self.state.step_count
        warnings = []

        # 1. Validate the step
        validation = self.validator.validate_reasoning_step(step)
        self.state.total_violations += len(validation.violations)
        self.state.critical_violations += len(validation.critical_violations)
        self.state.fpf_coverage.update(validation.fpf_coverage)

        if self.strict_mode and validation.has_critical:
            raise FPFValidationError(
                f"Critical FPF violation in step {step_num}: "
                f"{validation.critical_violations[0].description}"
            )

        for v in validation.violations:
            if v.severity == ViolationSeverity.WARNING:
                warnings.append(f"FPF Warning: {v.description}")

        # 2. Update ADI cycle based on action type
        action = step.get("next_action", {})
        action_type = action.get("action_type", "")
        gate_passed = True
        gate_message = ""

        if action_type == "generate_hypothesis":
            # Abduction phase
            result = self._handle_abduction(action)
            gate_passed = result.passed
            gate_message = result.message

        elif action_type == "deduce_consequences":
            # Deduction phase
            result = self._handle_deduction(action)
            gate_passed = result.passed
            gate_message = result.message

        # 3. Compute F-G-R if trust_assessment provided
        computed_fgr = None
        if "trust_assessment" in step and step["trust_assessment"]:
            computed_fgr = self._compute_fgr(step["trust_assessment"])

        # 4. Handle context transitions
        context_penalty = 0.0
        if "context" in step:
            ctx = step["context"]
            new_context_id = ctx.get("context_id", self.state.current_context_id)
            if new_context_id != self.state.current_context_id:
                context_penalty = self._handle_context_transition(
                    self.state.current_context_id,
                    new_context_id,
                )
                self.state.current_context_id = new_context_id

        # 5. Record step
        self.state.steps.append(step)
        self.state.current_phase = self.adi_controller.get_current_phase()
        self.state.artifact_state = self.adi_controller.get_artifact_state()

        result = StepResult(
            step_number=step_num,
            validation=validation,
            adi_phase=self.adi_controller.get_current_phase(),
            artifact_state=self.adi_controller.get_artifact_state(),
            computed_fgr=computed_fgr,
            context_penalty=context_penalty,
            gate_passed=gate_passed,
            gate_message=gate_message,
            warnings=warnings,
        )

        self._step_results.append(result)
        return result

    def finalize(self) -> dict[str, Any]:
        """
        Finalize the reasoning pipeline and return summary.

        Call this after all steps have been processed.
        """
        return {
            "session_id": self.session_id,
            "total_steps": self.state.step_count,
            "adi_cycles": self.adi_controller.get_cycle_count(),
            "final_phase": self.state.current_phase.value,
            "artifact_state": self.state.artifact_state.value,
            "validation": {
                "total_violations": self.state.total_violations,
                "critical_violations": self.state.critical_violations,
                "fpf_coverage": self.state.fpf_coverage,
            },
            "evidence": {
                "count": len(self.evidence_graph.nodes),
                "epistemic_debt": self.evidence_graph.compute_epistemic_debt(),
            },
            "context": {
                "current": self.state.current_context_id,
                "transitions": len(self.state.context_transitions),
            },
            "creativity": {
                "hypotheses_generated": self.state.hypotheses_generated,
                "ee_phase": self.state.ee_phase,
            },
        }

    # =========================================================================
    # ADI CYCLE HANDLING
    # =========================================================================

    def _handle_abduction(self, action: dict) -> Any:
        """Handle abduction (hypothesis generation) action."""
        from .adi_cycle import GateCheckResult

        hypothesis = Hypothesis(
            hypothesis_id=f"hyp_{self.state.step_count}",
            statement=action.get("hypothesis_statement", ""),
            anomaly_addressed=action.get("anomaly", ""),
            testable_predictions=action.get("testable_predictions", []),
            plausibility_score=action.get("plausibility_score", 0.5),
            plausibility_rationale=action.get("plausibility_rationale", ""),
            generation_method="llm_abduction",
        )

        result = self.adi_controller.submit_hypothesis(hypothesis)
        self.state.hypotheses_generated += 1

        # Track for NQD if in explore phase
        if self.nqd_engine is None:
            self.nqd_engine = NQDSearchEngine(
                self.nqd_config,
                HypothesisNQDScorer(),
            )

        self.nqd_engine.add_candidate(
            HypothesisCandidate(
                hypothesis_id=hypothesis.hypothesis_id,
                statement=hypothesis.statement,
                anomaly_addressed=hypothesis.anomaly_addressed,
                testable_predictions=hypothesis.testable_predictions,
                plausibility_score=hypothesis.plausibility_score,
            ),
            generation_method="abduction",
        )

        return result

    def _handle_deduction(self, action: dict) -> Any:
        """Handle deduction (consequence derivation) action."""
        deduction = DeductionResult(
            hypothesis_id=action.get("hypothesis", ""),
            derived_consequences=action.get("consequences", []),
            testable_predictions=action.get("testable_predictions", []),
        )

        return self.adi_controller.submit_deduction(deduction)

    # =========================================================================
    # F-G-R COMPUTATION
    # =========================================================================

    def _compute_fgr(self, trust_assessment: dict) -> FGRAssessment | None:
        """Compute actual F-G-R from trust assessment."""
        try:
            # Extract evidence references
            evidence_refs = trust_assessment.get("evidence_references", [])

            # Add evidence to graph
            for ref in evidence_refs:
                node = EvidenceNode(
                    evidence_id=ref.get("evidence_id", f"ev_{datetime.now().timestamp()}"),
                    evidence_type=ref.get("evidence_type", "observation"),
                    content_summary=ref.get("summary", ""),
                    source=ref.get("source", ""),
                    reliability=ref.get("reliability", 0.5),
                )
                self.evidence_graph.add_evidence(node)

            # Compute reliability using weakest-link
            if evidence_refs:
                reliabilities = [r.get("reliability", 0.5) for r in evidence_refs]
                agg_result = aggregate_reliabilities(reliabilities)
                computed_reliability = agg_result.value
            else:
                computed_reliability = trust_assessment.get("reliability", 0.0)

            # Return computed assessment
            return FGRAssessment(
                formality=trust_assessment.get("formality", 0),
                formality_rationale=trust_assessment.get("formality_rationale", ""),
                scope_context=trust_assessment.get("scope_context", ""),
                scope_conditions=trust_assessment.get("scope_conditions", []),
                scope_exclusions=trust_assessment.get("scope_exclusions", []),
                reliability=computed_reliability,
                reliability_rationale=f"Computed via weakest-link from {len(evidence_refs)} evidence nodes",
                evidence_references=[
                    EvidenceReference(**r) for r in evidence_refs
                ] if evidence_refs else [],
                assurance_level=trust_assessment.get("assurance_level", "L0_unsubstantiated"),
                has_stale_evidence=self.evidence_graph.has_stale_evidence(),
                epistemic_debt_note=f"Debt: {self.evidence_graph.compute_epistemic_debt():.2f}" if self.evidence_graph.compute_epistemic_debt() > 0 else None,
            )

        except Exception as e:
            logger.warning(f"Error computing F-G-R: {e}")
            return None

    # =========================================================================
    # CONTEXT HANDLING
    # =========================================================================

    def _handle_context_transition(
        self,
        from_context: str,
        to_context: str,
    ) -> float:
        """
        Handle transition between bounded contexts.

        Returns CL penalty for the transition.
        """
        # Record transition
        self.state.context_transitions.append({
            "from": from_context,
            "to": to_context,
            "step": self.state.step_count,
            "timestamp": datetime.now().isoformat(),
        })

        # Check for bridge
        bridge = self.context_registry.get_bridge(from_context, to_context)
        if bridge:
            return bridge.aggregate_cl

        # No explicit bridge - warn and return default penalty
        logger.warning(
            f"Context transition {from_context} -> {to_context} "
            "without explicit bridge. Applying default CL penalty."
        )
        return 0.1  # Default 10% CL penalty

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def add_context(self, context: CtxBoundedContext) -> None:
        """Add a bounded context to the registry."""
        self.context_registry.add_context(context)

    def add_bridge(self, bridge: BridgeMapping) -> None:
        """Add a bridge between contexts."""
        self.context_registry.add_bridge(bridge)

    def add_evidence(self, node: EvidenceNode) -> None:
        """Add evidence to the graph."""
        self.evidence_graph.add_evidence(node)

    def get_step_results(self) -> list[StepResult]:
        """Get all step results."""
        return list(self._step_results)

    def get_validation_summary(self) -> str:
        """Get human-readable validation summary."""
        lines = [
            f"## FPF Validation Summary",
            f"- Total steps: {self.state.step_count}",
            f"- Total violations: {self.state.total_violations}",
            f"- Critical violations: {self.state.critical_violations}",
            f"",
            "### FPF Principle Coverage:",
        ]

        for principle, covered in self.state.fpf_coverage.items():
            status = "✓" if covered else "✗"
            lines.append(f"  - {principle}: {status}")

        return "\n".join(lines)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class FPFValidationError(Exception):
    """Raised when FPF validation fails in strict mode."""
    pass


class FPFGateError(Exception):
    """Raised when an ADI gate check fails."""
    pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline(
    strict_mode: bool = False,
    max_cycles: int = 3,
) -> FPFReasoningPipeline:
    """Create a new FPF reasoning pipeline with defaults."""
    return FPFReasoningPipeline(
        strict_mode=strict_mode,
        max_adi_cycles=max_cycles,
    )


def validate_step_standalone(step: dict) -> ValidationResult:
    """Validate a single step without full pipeline."""
    validator = FPFReasoningValidator()
    return validator.validate_reasoning_step(step)


def compute_fgr_standalone(
    claim: str,
    evidence: list[dict],
    context: str = "default",
) -> dict[str, Any]:
    """
    Compute F-G-R for a claim with evidence.

    Returns dict with formality, scope, reliability, and assurance_level.
    """
    # Estimate formality
    formality = compute_formality(claim)

    # Compute reliability via weakest-link
    if evidence:
        reliabilities = [e.get("reliability", 0.5) for e in evidence]
        reliability = min(reliabilities)
        assurance = "L2_assured" if reliability > 0.7 else "L1_partial"
    else:
        reliability = 0.0
        assurance = "L0_unsubstantiated"

    return {
        "formality": formality.value,
        "formality_name": formality.name,
        "scope_context": context,
        "reliability": reliability,
        "assurance_level": assurance,
        "evidence_count": len(evidence),
    }

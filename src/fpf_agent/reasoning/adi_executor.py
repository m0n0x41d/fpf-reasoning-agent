"""
ADI Cycle Executor with SGR Schemas

This module integrates the SGR Pydantic schemas with the ADI cycle controller.
It provides the bridge between:
- LLM calls (using Pydantic-AI or similar)
- SGR schemas (structured outputs)
- ADICycleController (state machine)
- EpistemeStore (persistence)

SGR Cascade: One LLM, different schemas per phase.
Python orchestration controls phase order and gate enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel

from .schemas.abduction import AbductionOutput, ABDUCTION_SYSTEM_PROMPT
from .schemas.deduction import DeductionOutput, DEDUCTION_SYSTEM_PROMPT
from .schemas.induction import InductionOutput, INDUCTION_SYSTEM_PROMPT
from .schemas.synthesis import SynthesisOutput, SYNTHESIS_SYSTEM_PROMPT
from .lifecycle import LifecycleManager

from ..kernel.types import AssuranceLevel, LifecycleState

if TYPE_CHECKING:
    from ..kernel.holons import UEpisteme
    from ..kernel.types import HolonId
    from ..persistence.episteme_store import EpistemeStore
    from ..persistence.evidence_graph import EvidenceGraph


T = TypeVar("T", bound=BaseModel)


# =============================================================================
# LLM CLIENT PROTOCOL
# =============================================================================

class LLMClient(Protocol):
    """Protocol for LLM client that supports structured output."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.2,
    ) -> T:
        """
        Generate structured output from LLM.

        Args:
            system_prompt: System instructions
            user_prompt: User query/context
            response_model: Pydantic model for structured output
            temperature: Sampling temperature

        Returns:
            Instance of response_model
        """
        ...


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================

@dataclass
class ExecutionContext:
    """Context for ADI cycle execution."""

    context_id: str
    problem: str
    existing_knowledge: list["UEpisteme"] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    max_iterations: int = 3

    # State tracking
    current_iteration: int = 0
    phase_outputs: dict[str, BaseModel] = field(default_factory=dict)
    episteme_ids: list[UUID] = field(default_factory=list)

    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


@dataclass
class PhaseResult:
    """Result from executing a single phase."""

    phase: str
    success: bool
    output: BaseModel | None
    episteme: "UEpisteme | None"
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class CycleResult:
    """Result from executing a complete ADI cycle."""

    success: bool
    final_episteme: "UEpisteme | None"
    phase_results: list[PhaseResult]
    iterations: int
    verdict: str
    total_duration_ms: float = 0.0


# =============================================================================
# ADI EXECUTOR
# =============================================================================

class ADIExecutor:
    """
    Executes ADI reasoning cycles using SGR schemas.

    Integrates:
    - LLM client for structured generation
    - SGR schemas for each phase
    - EpistemeStore for persistence
    - EvidenceGraph for evidence links
    - LifecycleManager for state transitions
    """

    def __init__(
        self,
        llm_client: LLMClient,
        episteme_store: "EpistemeStore",
        evidence_graph: "EvidenceGraph",
        lifecycle_manager: LifecycleManager | None = None,
    ):
        self.llm = llm_client
        self.store = episteme_store
        self.evidence = evidence_graph
        self.lifecycle = lifecycle_manager or LifecycleManager()

    async def execute_cycle(
        self,
        context: ExecutionContext,
    ) -> CycleResult:
        """
        Execute a complete ADI cycle.

        Runs Abduction → Deduction → Induction, with possible iterations
        if hypothesis is refuted.
        """
        import time

        start_time = time.time()
        phase_results: list[PhaseResult] = []
        current_problem = context.problem
        final_episteme = None

        while context.current_iteration < context.max_iterations:
            context.current_iteration += 1

            # ABDUCTION
            abd_result = await self._execute_abduction(context, current_problem)
            phase_results.append(abd_result)

            if not abd_result.success or abd_result.episteme is None:
                break

            episteme = abd_result.episteme

            # DEDUCTION
            ded_result = await self._execute_deduction(context, episteme, abd_result.output)
            phase_results.append(ded_result)

            if not ded_result.success:
                break

            episteme = ded_result.episteme or episteme

            # INDUCTION
            ind_result = await self._execute_induction(
                context, episteme, ded_result.output
            )
            phase_results.append(ind_result)

            if not ind_result.success:
                break

            episteme = ind_result.episteme or episteme
            final_episteme = episteme

            # Check verdict
            if isinstance(ind_result.output, InductionOutput):
                verdict = ind_result.output.overall_verdict.value

                if verdict == "corroborated":
                    # Success!
                    context.completed_at = datetime.utcnow()
                    return CycleResult(
                        success=True,
                        final_episteme=final_episteme,
                        phase_results=phase_results,
                        iterations=context.current_iteration,
                        verdict=verdict,
                        total_duration_ms=(time.time() - start_time) * 1000,
                    )

                elif verdict == "refuted":
                    # Loop back with new anomaly
                    new_anomaly = ind_result.output.if_refuted_new_anomaly
                    if new_anomaly:
                        current_problem = new_anomaly
                    else:
                        current_problem = f"[Refuted] {current_problem}"
                    continue

                else:
                    # Inconclusive or weakened - try refinement
                    if ind_result.output.refinement_suggestions:
                        # Could implement refinement logic here
                        pass
                    continue

        # Max iterations or error
        total_time = (time.time() - start_time) * 1000
        context.completed_at = datetime.utcnow()

        return CycleResult(
            success=final_episteme is not None,
            final_episteme=final_episteme,
            phase_results=phase_results,
            iterations=context.current_iteration,
            verdict="max_iterations" if context.current_iteration >= context.max_iterations else "error",
            total_duration_ms=total_time,
        )

    async def _execute_abduction(
        self,
        context: ExecutionContext,
        problem: str,
    ) -> PhaseResult:
        """Execute abduction phase."""
        import time

        start = time.time()

        try:
            # Build user prompt
            knowledge_context = self._format_knowledge(context.existing_knowledge)

            user_prompt = f"""## Problem to Explain
{problem}

## Existing Knowledge in Context
{knowledge_context}

## Constraints
- Generate 2-5 competing hypotheses
- Each must have at least one testable prediction
- Document why alternatives are less preferred

Generate hypotheses that could explain this problem."""

            # LLM call with SGR schema
            output: AbductionOutput = await self.llm.generate(
                system_prompt=ABDUCTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=AbductionOutput,
                temperature=0.7,  # Higher for creativity in abduction
            )

            # Create episteme for best hypothesis
            episteme = await self._create_hypothesis_episteme(
                context, problem, output
            )

            context.phase_outputs["abduction"] = output
            context.episteme_ids.append(episteme.holon_id.id)

            return PhaseResult(
                phase="abduction",
                success=True,
                output=output,
                episteme=episteme,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return PhaseResult(
                phase="abduction",
                success=False,
                output=None,
                episteme=None,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def _execute_deduction(
        self,
        context: ExecutionContext,
        episteme: "UEpisteme",
        abduction_output: AbductionOutput | None,
    ) -> PhaseResult:
        """Execute deduction phase."""
        import time

        start = time.time()

        try:
            # Get hypothesis from episteme or abduction output
            if abduction_output:
                hypothesis = abduction_output.best_hypothesis
                hypothesis_text = hypothesis.statement
                mechanism = hypothesis.mechanism
            else:
                hypothesis_text = episteme.claim_graph.get("statement", "")
                mechanism = episteme.claim_graph.get("mechanism", "")

            user_prompt = f"""## Hypothesis to Analyze
{hypothesis_text}

## Proposed Mechanism
{mechanism}

## Task
1. List all premises explicitly (including the hypothesis)
2. Derive logical consequences step by step
3. Identify testable predictions with falsification criteria
4. Check for internal contradictions

Derive the logical consequences and testable predictions."""

            # LLM call with SGR schema
            output: DeductionOutput = await self.llm.generate(
                system_prompt=DEDUCTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=DeductionOutput,
                temperature=0.3,  # Lower for logical precision
            )

            # Update episteme with deduction results
            updated_episteme = await self._update_episteme_with_deduction(
                episteme, output
            )

            context.phase_outputs["deduction"] = output

            return PhaseResult(
                phase="deduction",
                success=True,
                output=output,
                episteme=updated_episteme,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return PhaseResult(
                phase="deduction",
                success=False,
                output=None,
                episteme=None,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def _execute_induction(
        self,
        context: ExecutionContext,
        episteme: "UEpisteme",
        deduction_output: DeductionOutput | None,
    ) -> PhaseResult:
        """Execute induction phase."""
        import time

        start = time.time()

        try:
            # Get predictions from deduction
            if deduction_output:
                predictions_text = "\n".join([
                    f"{i+1}. {p.prediction} (expect: {p.expected_outcome}, falsify if: {p.falsification_criterion})"
                    for i, p in enumerate(deduction_output.predictions)
                ])
                hypothesis_text = deduction_output.hypothesis_statement
            else:
                predictions = episteme.claim_graph.get("predictions", [])
                predictions_text = "\n".join([
                    f"{i+1}. {p}" if isinstance(p, str) else f"{i+1}. {p.get('prediction', p)}"
                    for i, p in enumerate(predictions)
                ])
                hypothesis_text = episteme.claim_graph.get("statement", "")

            sources_text = "\n".join(context.evidence_sources) if context.evidence_sources else "No specific sources provided - use general knowledge"

            user_prompt = f"""## Hypothesis Being Tested
{hypothesis_text}

## Predictions to Test
{predictions_text}

## Available Evidence Sources
{sources_text}

## Task
1. Test each prediction against available evidence
2. Be specific about what evidence supports or refutes each prediction
3. Determine overall verdict (corroborated/refuted/weakened/inconclusive)
4. If refuted, identify the new anomaly for the next cycle

Test each prediction and determine the verdict."""

            # LLM call with SGR schema
            output: InductionOutput = await self.llm.generate(
                system_prompt=INDUCTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=InductionOutput,
                temperature=0.2,  # Low for objective assessment
            )

            # Update episteme with induction results
            updated_episteme = await self._update_episteme_with_induction(
                episteme, output
            )

            # Link evidence to episteme
            await self._link_evidence(updated_episteme, output)

            context.phase_outputs["induction"] = output

            return PhaseResult(
                phase="induction",
                success=True,
                output=output,
                episteme=updated_episteme,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return PhaseResult(
                phase="induction",
                success=False,
                output=None,
                episteme=None,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _parse_assurance_level(self, level: str) -> AssuranceLevel:
        """Parse assurance level string to enum."""
        mapping = {"L0": AssuranceLevel.L0, "L1": AssuranceLevel.L1, "L2": AssuranceLevel.L2}
        return mapping.get(level, AssuranceLevel.L0)

    async def _create_hypothesis_episteme(
        self,
        context: ExecutionContext,
        problem: str,
        output: AbductionOutput,
    ) -> "UEpisteme":
        """Create an episteme for the best hypothesis."""
        from ..kernel.holons import UEpisteme
        from ..kernel.types import HolonId, TemporalStance

        best = output.best_hypothesis

        episteme = UEpisteme(
            holon_id=HolonId(context_id=context.context_id),
            described_entity=problem[:100],
            claim_graph={
                "type": "hypothesis",
                "statement": best.statement,
                "mechanism": best.mechanism,
                "assumptions": best.assumptions,
                "testable_predictions": best.testable_predictions,
                "potential_issues": best.potential_issues,
                "source_problem": problem,
                "competing_hypotheses": [h.statement for h in output.alternative_hypotheses],
                "selection_rationale": output.selection_rationale,
                "plausibility_score": output.plausibility_score,
            },
            lifecycle_state=LifecycleState.EXPLORATION,
            assurance_level=AssuranceLevel.L0,
            temporal_stance=TemporalStance.DESIGN_TIME,
        )

        return self.store.create(episteme)

    async def _update_episteme_with_deduction(
        self,
        episteme: "UEpisteme",
        output: DeductionOutput,
    ) -> "UEpisteme":
        """Update episteme with deduction results."""
        updated_claim_graph = dict(episteme.claim_graph)

        updated_claim_graph["predictions"] = [
            {
                "prediction_id": p.prediction_id,
                "prediction": p.prediction,
                "expected_outcome": p.expected_outcome,
                "falsification_criterion": p.falsification_criterion,
                "criticality": p.criticality,
            }
            for p in output.predictions
        ]
        updated_claim_graph["derivation_chain"] = [
            {
                "step": s.step_number,
                "from": s.from_premises,
                "inference": s.inference_type.value,
                "conclusion": s.conclusion,
            }
            for s in output.derivation_chain
        ]
        updated_claim_graph["premises"] = [
            {
                "id": p.premise_id,
                "statement": p.statement,
                "source": p.source.value,
                "confidence": p.confidence,
            }
            for p in output.premises
        ]
        updated_claim_graph["internal_consistency"] = output.internal_consistency
        updated_claim_graph["formality_achieved"] = output.formality_achieved
        updated_claim_graph["deduction_confidence"] = output.overall_confidence

        updated = episteme.model_copy(
            update={
                "claim_graph": updated_claim_graph,
                "lifecycle_state": LifecycleState.SHAPING,
            }
        )

        return self.store.update(updated)

    async def _update_episteme_with_induction(
        self,
        episteme: "UEpisteme",
        output: InductionOutput,
    ) -> "UEpisteme":
        """Update episteme with induction results."""
        updated_claim_graph = dict(episteme.claim_graph)

        updated_claim_graph["test_results"] = [
            {
                "prediction_id": t.prediction_id,
                "prediction": t.prediction_text,
                "observed": t.observed_outcome,
                "match": t.match_result.value,
                "confidence": t.confidence_in_assessment,
            }
            for t in output.prediction_tests
        ]
        updated_claim_graph["test_verdict"] = output.overall_verdict.value
        updated_claim_graph["verdict_rationale"] = output.verdict_rationale
        updated_claim_graph["reliability_delta"] = output.reliability_delta
        updated_claim_graph["refinement_suggestions"] = output.refinement_suggestions

        if output.if_refuted_new_anomaly:
            updated_claim_graph["refutation_anomaly"] = output.if_refuted_new_anomaly

        # Determine new lifecycle state and assurance
        if output.overall_verdict.value == "corroborated":
            new_state = LifecycleState.EVIDENCE
            new_assurance = self._parse_assurance_level(output.recommended_assurance)
        elif output.overall_verdict.value == "refuted":
            new_state = LifecycleState.EXPLORATION
            new_assurance = AssuranceLevel.L0
        else:
            new_state = LifecycleState.EVIDENCE
            new_assurance = AssuranceLevel.L0

        updated = episteme.model_copy(
            update={
                "claim_graph": updated_claim_graph,
                "lifecycle_state": new_state,
                "assurance_level": new_assurance,
            }
        )

        return self.store.update(updated)

    async def _link_evidence(
        self,
        episteme: "UEpisteme",
        output: InductionOutput,
    ) -> None:
        """Link evidence items to episteme via evidence graph."""
        from ..kernel.holons import UEpisteme
        from ..kernel.types import HolonId
        from ..trust.fgr import CongruenceLevel

        for test in output.prediction_tests:
            for ev in test.evidence:
                # Create evidence episteme
                ev_episteme = UEpisteme(
                    holon_id=HolonId(context_id=episteme.holon_id.context_id),
                    described_entity=f"Evidence: {ev.description[:50]}...",
                    claim_graph={
                        "type": "evidence",
                        "description": ev.description,
                        "source": ev.source,
                        "source_type": ev.source_type.value,
                        "reliability": ev.reliability,
                    },
                    assurance_level="L0",
                )
                ev_episteme = self.store.create(ev_episteme)

                # Link to claim
                relation = (
                    "supports" if test.match_result.value == "match"
                    else "refutes" if test.match_result.value == "mismatch"
                    else "qualifies"
                )

                # Map reliability to CL
                cl = (
                    CongruenceLevel.CL5_EXACT if ev.reliability > 0.9
                    else CongruenceLevel.CL4_ALIGNED if ev.reliability > 0.7
                    else CongruenceLevel.CL3_PARTIAL if ev.reliability > 0.5
                    else CongruenceLevel.CL2_LOOSE
                )

                self.evidence.add_link(
                    claim_id=episteme.holon_id.id,
                    evidence_id=ev_episteme.holon_id.id,
                    relation=relation,
                    strength=ev.reliability * ev.relevance,
                    cl=cl,
                )

    def _format_knowledge(self, epistemes: list["UEpisteme"]) -> str:
        """Format existing knowledge for prompt context."""
        if not epistemes:
            return "No prior knowledge available."

        lines = []
        for e in epistemes:
            fgr = f"[{e.assurance_level}]"
            statement = e.claim_graph.get("statement", e.described_entity)
            lines.append(f"- {statement} {fgr}")

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_adi_cycle(
    llm_client: LLMClient,
    episteme_store: "EpistemeStore",
    evidence_graph: "EvidenceGraph",
    problem: str,
    context_id: str,
    existing_knowledge: list["UEpisteme"] | None = None,
    evidence_sources: list[str] | None = None,
    max_iterations: int = 3,
) -> CycleResult:
    """
    Convenience function to run a complete ADI cycle.

    Args:
        llm_client: LLM client supporting structured output
        episteme_store: Persistence store for epistemes
        evidence_graph: Graph for evidence relationships
        problem: The problem/question to investigate
        context_id: Bounded context ID
        existing_knowledge: Optional prior epistemes
        evidence_sources: Optional evidence sources to consider
        max_iterations: Max ADI iterations before stopping

    Returns:
        CycleResult with final episteme and phase details
    """
    executor = ADIExecutor(llm_client, episteme_store, evidence_graph)

    context = ExecutionContext(
        context_id=context_id,
        problem=problem,
        existing_knowledge=existing_knowledge or [],
        evidence_sources=evidence_sources or [],
        max_iterations=max_iterations,
    )

    return await executor.execute_cycle(context)

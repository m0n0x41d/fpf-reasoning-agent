"""
Artifact Lifecycle Management (FPF B.5.1)

Manages the lifecycle states of epistemes:
    Exploration → Shaping → Evidence → Operate

Each transition has gates that must be satisfied.
This module enforces FPF compliance for state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from ..kernel.types import LifecycleState

if TYPE_CHECKING:
    from ..kernel.holons import UEpisteme


@dataclass
class TransitionRule:
    """
    Rule for a state transition.

    Defines what checks must pass to transition between states.
    """

    from_state: LifecycleState
    to_state: LifecycleState
    required_checks: list[str]
    driven_by: str  # What triggers this transition (e.g., "abduction_complete")

    def __str__(self) -> str:
        return f"{self.from_state.value} → {self.to_state.value} (driven by {self.driven_by})"


# =============================================================================
# NORMATIVE TRANSITIONS (FPF B.5.1)
# =============================================================================

TRANSITIONS: list[TransitionRule] = [
    # Forward transitions
    TransitionRule(
        from_state=LifecycleState.EXPLORATION,
        to_state=LifecycleState.SHAPING,
        required_checks=[
            "has_hypothesis",
            "hypothesis_has_predictions",
            "alternatives_documented",
        ],
        driven_by="abduction_complete",
    ),
    TransitionRule(
        from_state=LifecycleState.SHAPING,
        to_state=LifecycleState.EVIDENCE,
        required_checks=[
            "has_testable_predictions",
            "derivation_complete",
            "no_internal_contradictions",
        ],
        driven_by="deduction_complete",
    ),
    TransitionRule(
        from_state=LifecycleState.EVIDENCE,
        to_state=LifecycleState.OPERATE,
        required_checks=[
            "has_test_results",
            "verdict_is_corroborated",
            "assurance_level_ge_L1",
            "evidence_links_exist",
        ],
        driven_by="induction_passed",
    ),
    # Backward transitions (refinement loops)
    TransitionRule(
        from_state=LifecycleState.SHAPING,
        to_state=LifecycleState.EXPLORATION,
        required_checks=[
            "refinement_needed",
        ],
        driven_by="hypothesis_needs_revision",
    ),
    TransitionRule(
        from_state=LifecycleState.EVIDENCE,
        to_state=LifecycleState.EXPLORATION,
        required_checks=[
            "verdict_is_refuted",
        ],
        driven_by="induction_failed",
    ),
    TransitionRule(
        from_state=LifecycleState.EVIDENCE,
        to_state=LifecycleState.SHAPING,
        required_checks=[
            "verdict_is_partial",
            "refinement_possible",
        ],
        driven_by="induction_partial",
    ),
    # Degradation (e.g., evidence becomes stale)
    TransitionRule(
        from_state=LifecycleState.OPERATE,
        to_state=LifecycleState.EVIDENCE,
        required_checks=[
            "evidence_stale_or_challenged",
        ],
        driven_by="revalidation_needed",
    ),
]


@dataclass
class TransitionResult:
    """Result of attempting a state transition."""

    success: bool
    from_state: LifecycleState
    to_state: LifecycleState
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "BLOCKED"
        return f"[{status}] {self.from_state.value} → {self.to_state.value}: {self.reason}"


# =============================================================================
# CHECK FUNCTIONS
# =============================================================================

def _check_has_hypothesis(episteme: "UEpisteme") -> bool:
    """Check if episteme has a hypothesis statement."""
    cg = episteme.claim_graph
    return bool(cg.get("statement") or cg.get("hypothesis"))


def _check_hypothesis_has_predictions(episteme: "UEpisteme") -> bool:
    """Check if hypothesis has testable predictions."""
    cg = episteme.claim_graph
    predictions = cg.get("predictions") or cg.get("testable_predictions") or []
    return len(predictions) > 0


def _check_alternatives_documented(episteme: "UEpisteme") -> bool:
    """Check if alternative hypotheses are documented."""
    cg = episteme.claim_graph
    # Allow pass if explicitly marked as only hypothesis or has alternatives
    alternatives = cg.get("competing_hypotheses") or cg.get("alternatives") or []
    only_option = cg.get("only_viable_option", False)
    return len(alternatives) > 0 or only_option


def _check_has_testable_predictions(episteme: "UEpisteme") -> bool:
    """Check if there are testable predictions from deduction."""
    cg = episteme.claim_graph
    predictions = cg.get("predictions") or []
    # Must have at least one with falsification criterion
    return any(
        isinstance(p, dict) and p.get("falsification_criterion")
        for p in predictions
    ) or len(predictions) > 0


def _check_derivation_complete(episteme: "UEpisteme") -> bool:
    """Check if deduction derivation is complete."""
    cg = episteme.claim_graph
    return bool(cg.get("derivation_chain") or cg.get("derived_consequences"))


def _check_no_internal_contradictions(episteme: "UEpisteme") -> bool:
    """Check for internal consistency."""
    cg = episteme.claim_graph
    # Default to True unless explicitly marked inconsistent
    return cg.get("internal_consistency", True)


def _check_has_test_results(episteme: "UEpisteme") -> bool:
    """Check if test results exist."""
    cg = episteme.claim_graph
    return bool(cg.get("test_results") or cg.get("prediction_tests"))


def _check_verdict_is_corroborated(episteme: "UEpisteme") -> bool:
    """Check if hypothesis was corroborated."""
    cg = episteme.claim_graph
    verdict = cg.get("test_verdict") or cg.get("overall_verdict")
    return verdict in ("corroborated", "confirmed")


def _check_verdict_is_refuted(episteme: "UEpisteme") -> bool:
    """Check if hypothesis was refuted."""
    cg = episteme.claim_graph
    verdict = cg.get("test_verdict") or cg.get("overall_verdict")
    return verdict == "refuted"


def _check_verdict_is_partial(episteme: "UEpisteme") -> bool:
    """Check if verdict is partial (some support, some refutation)."""
    cg = episteme.claim_graph
    verdict = cg.get("test_verdict") or cg.get("overall_verdict")
    return verdict in ("partial", "weakened")


def _check_assurance_level_ge_L1(episteme: "UEpisteme") -> bool:
    """Check if assurance level is at least L1."""
    return episteme.assurance_level in ("L1", "L2")


def _check_evidence_links_exist(episteme: "UEpisteme") -> bool:
    """Check if evidence IDs are linked."""
    return len(episteme.evidence_ids) > 0


def _check_refinement_needed(episteme: "UEpisteme") -> bool:
    """Check if refinement is needed (always true if requested)."""
    cg = episteme.claim_graph
    return bool(cg.get("refinement_needed") or cg.get("refinement_suggestions"))


def _check_refinement_possible(episteme: "UEpisteme") -> bool:
    """Check if refinement is possible."""
    cg = episteme.claim_graph
    suggestions = cg.get("refinement_suggestions") or []
    return len(suggestions) > 0


def _check_evidence_stale_or_challenged(episteme: "UEpisteme") -> bool:
    """Check if evidence is stale or challenged."""
    cg = episteme.claim_graph
    return bool(
        cg.get("evidence_stale")
        or cg.get("evidence_challenged")
        or cg.get("revalidation_needed")
    )


# Check registry
CHECK_REGISTRY: dict[str, Callable[["UEpisteme"], bool]] = {
    "has_hypothesis": _check_has_hypothesis,
    "hypothesis_has_predictions": _check_hypothesis_has_predictions,
    "alternatives_documented": _check_alternatives_documented,
    "has_testable_predictions": _check_has_testable_predictions,
    "derivation_complete": _check_derivation_complete,
    "no_internal_contradictions": _check_no_internal_contradictions,
    "has_test_results": _check_has_test_results,
    "verdict_is_corroborated": _check_verdict_is_corroborated,
    "verdict_is_refuted": _check_verdict_is_refuted,
    "verdict_is_partial": _check_verdict_is_partial,
    "assurance_level_ge_L1": _check_assurance_level_ge_L1,
    "evidence_links_exist": _check_evidence_links_exist,
    "refinement_needed": _check_refinement_needed,
    "refinement_possible": _check_refinement_possible,
    "evidence_stale_or_challenged": _check_evidence_stale_or_challenged,
}


# =============================================================================
# LIFECYCLE MANAGER
# =============================================================================

class LifecycleManager:
    """
    Manages artifact lifecycle transitions.

    Ensures FPF B.5.1 compliance by enforcing gates between states.
    """

    def __init__(
        self,
        transitions: list[TransitionRule] | None = None,
        check_registry: dict[str, Callable[["UEpisteme"], bool]] | None = None,
    ):
        """
        Initialize lifecycle manager.

        Args:
            transitions: Custom transition rules (default: FPF normative)
            check_registry: Custom check functions (default: built-in)
        """
        self.transitions = transitions or TRANSITIONS
        self.check_registry = check_registry or CHECK_REGISTRY
        self._history: list[TransitionResult] = []

    def get_current_state(self, episteme: "UEpisteme") -> LifecycleState:
        """Get current lifecycle state of an episteme."""
        return LifecycleState(episteme.lifecycle_state)

    def get_allowed_transitions(
        self,
        episteme: "UEpisteme",
    ) -> list[LifecycleState]:
        """Get states this episteme can transition to."""
        current = self.get_current_state(episteme)
        return [
            rule.to_state
            for rule in self.transitions
            if rule.from_state == current
        ]

    def find_transition_rule(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
    ) -> TransitionRule | None:
        """Find the rule for a specific transition."""
        for rule in self.transitions:
            if rule.from_state == from_state and rule.to_state == to_state:
                return rule
        return None

    def check_transition(
        self,
        episteme: "UEpisteme",
        target_state: LifecycleState,
    ) -> TransitionResult:
        """
        Check if transition is allowed.

        Returns TransitionResult with pass/fail details.
        """
        current = self.get_current_state(episteme)

        # Find applicable rule
        rule = self.find_transition_rule(current, target_state)

        if rule is None:
            return TransitionResult(
                success=False,
                from_state=current,
                to_state=target_state,
                reason=f"No transition rule from {current.value} to {target_state.value}",
            )

        # Check each requirement
        checks_passed = []
        checks_failed = []

        for check_name in rule.required_checks:
            check_fn = self.check_registry.get(check_name)
            if check_fn is None:
                checks_failed.append(f"{check_name} (unknown check)")
                continue

            try:
                if check_fn(episteme):
                    checks_passed.append(check_name)
                else:
                    checks_failed.append(check_name)
            except Exception as e:
                checks_failed.append(f"{check_name} (error: {e})")

        success = len(checks_failed) == 0

        result = TransitionResult(
            success=success,
            from_state=current,
            to_state=target_state,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            reason="All checks passed" if success else f"Failed: {', '.join(checks_failed)}",
        )

        self._history.append(result)
        return result

    def can_transition(
        self,
        episteme: "UEpisteme",
        target_state: LifecycleState,
    ) -> bool:
        """Quick check if transition is possible."""
        return self.check_transition(episteme, target_state).success

    def get_blocking_checks(
        self,
        episteme: "UEpisteme",
        target_state: LifecycleState,
    ) -> list[str]:
        """Get list of checks blocking a transition."""
        result = self.check_transition(episteme, target_state)
        return result.checks_failed

    def suggest_next_state(
        self,
        episteme: "UEpisteme",
    ) -> LifecycleState | None:
        """
        Suggest the next state based on current checks.

        Returns the first allowed transition, or None if blocked.
        """
        allowed = self.get_allowed_transitions(episteme)
        for state in allowed:
            if self.can_transition(episteme, state):
                return state
        return None

    def get_transition_history(self) -> list[TransitionResult]:
        """Get history of transition attempts."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear transition history."""
        self._history.clear()

    def get_state_requirements(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
    ) -> list[str]:
        """Get requirements for a specific transition."""
        rule = self.find_transition_rule(from_state, to_state)
        if rule is None:
            return []
        return rule.required_checks

    def format_transition_guide(
        self,
        episteme: "UEpisteme",
    ) -> str:
        """
        Format a guide showing what's needed to progress.
        """
        current = self.get_current_state(episteme)
        lines = [f"Current state: {current.value}", ""]

        for state in self.get_allowed_transitions(episteme):
            result = self.check_transition(episteme, state)
            status = "✓" if result.success else "✗"
            lines.append(f"{status} → {state.value}")

            if result.checks_passed:
                for check in result.checks_passed:
                    lines.append(f"    ✓ {check}")
            if result.checks_failed:
                for check in result.checks_failed:
                    lines.append(f"    ✗ {check}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_lifecycle_manager() -> LifecycleManager:
    """Get a default lifecycle manager instance."""
    return LifecycleManager()


def check_can_transition(
    episteme: "UEpisteme",
    target_state: LifecycleState,
) -> tuple[bool, list[str]]:
    """
    Convenience function to check if transition is allowed.

    Returns (allowed, list of failed checks).
    """
    manager = LifecycleManager()
    result = manager.check_transition(episteme, target_state)
    return result.success, result.checks_failed


def get_next_state(episteme: "UEpisteme") -> LifecycleState | None:
    """Get the suggested next state for an episteme."""
    manager = LifecycleManager()
    return manager.suggest_next_state(episteme)

"""
FPF B.3: Trust Calculus — composition rules.

When composing epistemes (serial or parallel), trust propagates
according to strict rules. Key invariant: WLNK (Weakest Link).

Serial composition: A depends on B depends on C
- F = min(F_i)
- G = intersect(G_i)
- R = min(R_i) * product(CL_penalty_i)

Parallel composition: Independent sources A, B, C
- F = min(F_i)
- G = union where individually supported
- R = probabilistic combination with conservative cap
"""
from __future__ import annotations

from typing import Sequence

from .fgr import ClaimScope, CongruenceLevel, FGRTuple, FormalityLevel


class TrustCalculus:
    """
    Computes trust for composed epistemes.

    Invariants:
    - WLNK (Weakest Link): Result never better than worst input
    - F/G invariant under CL: Only R is penalized
    """

    @staticmethod
    def weakest_link(fgr_list: Sequence[FGRTuple]) -> FGRTuple:
        """
        WLNK invariant — result is bounded by worst input.

        Used when composition creates a chain of dependencies.
        This is the foundation of FPF trust composition.
        """
        if not fgr_list:
            raise ValueError("Cannot compute WLNK for empty list")

        min_f = min(fgr.formality for fgr in fgr_list)
        min_r = min(fgr.reliability for fgr in fgr_list)

        combined_scope = fgr_list[0].claim_scope
        for fgr in fgr_list[1:]:
            combined_scope = combined_scope.intersect(fgr.claim_scope)

        return FGRTuple(
            formality=FormalityLevel(min_f),
            claim_scope=combined_scope,
            reliability=min_r
        )

    @staticmethod
    def compose_serial(
        fgr_list: Sequence[FGRTuple],
        cl_edges: Sequence[CongruenceLevel]
    ) -> FGRTuple:
        """
        Compose epistemes in series (A depends on B depends on C).

        F = min(F_i)
        G = intersect(G_i)
        R = min(R_i) * product(1 - CL_penalty_i)

        Args:
            fgr_list: List of F-G-R tuples in dependency order
            cl_edges: Congruence levels for N-1 edges between N nodes

        Returns:
            Composed F-G-R tuple

        Raises:
            ValueError: If cl_edges count doesn't match fgr_list - 1
        """
        if len(cl_edges) != len(fgr_list) - 1:
            raise ValueError(
                f"Need {len(fgr_list) - 1} CL edges for {len(fgr_list)} epistemes, "
                f"got {len(cl_edges)}"
            )

        base = TrustCalculus.weakest_link(fgr_list)

        for cl in cl_edges:
            base = base.apply_cl_penalty(cl)

        return base

    @staticmethod
    def compose_parallel(fgr_list: Sequence[FGRTuple]) -> FGRTuple:
        """
        Compose independent parallel sources.

        Multiple independent sources can increase confidence,
        but conservatively (not by simple averaging).

        F = min(F_i)
        G = union of scopes where individually supported
        R = 1 - product(1 - R_i), capped at max(R_i) + 0.1

        The cap prevents unrealistic reliability boosts from
        combining weak sources.
        """
        if not fgr_list:
            raise ValueError("Cannot compose empty list")

        min_f = min(fgr.formality for fgr in fgr_list)

        combined_r = 1.0
        for fgr in fgr_list:
            combined_r *= (1 - fgr.reliability)
        combined_r = 1 - combined_r

        max_individual_r = max(fgr.reliability for fgr in fgr_list)
        combined_r = min(combined_r, max_individual_r + 0.1)

        all_contexts: set[str] = set()
        for fgr in fgr_list:
            all_contexts |= fgr.claim_scope.contexts

        combined_scope = ClaimScope(contexts=all_contexts)

        return FGRTuple(
            formality=FormalityLevel(min_f),
            claim_scope=combined_scope,
            reliability=min(1.0, combined_r)
        )

    @staticmethod
    def aggregate_with_weights(
        fgr_list: Sequence[FGRTuple],
        weights: Sequence[float]
    ) -> FGRTuple:
        """
        Weighted aggregation for ranked sources.

        NOTE: This is NOT standard FPF — provided for compatibility
        with systems that rank source authority. Use with caution.

        F = min(F_i) (no weighting — formality is not arithmetic)
        G = union of scopes
        R = sum(w_i * R_i) / sum(w_i), capped at max(R_i)
        """
        if len(weights) != len(fgr_list):
            raise ValueError("Weights must match fgr_list length")
        if not fgr_list:
            raise ValueError("Cannot aggregate empty list")

        min_f = min(fgr.formality for fgr in fgr_list)

        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")

        weighted_r = sum(w * fgr.reliability for w, fgr in zip(weights, fgr_list))
        weighted_r /= total_weight

        max_r = max(fgr.reliability for fgr in fgr_list)
        weighted_r = min(weighted_r, max_r)

        all_contexts: set[str] = set()
        for fgr in fgr_list:
            all_contexts |= fgr.claim_scope.contexts

        return FGRTuple(
            formality=FormalityLevel(min_f),
            claim_scope=ClaimScope(contexts=all_contexts),
            reliability=weighted_r
        )


def check_trust_consistency(
    claimed: FGRTuple,
    evidence: Sequence[FGRTuple]
) -> tuple[bool, list[str]]:
    """
    Check if claimed trust is consistent with evidence.

    Returns (is_consistent, list of violations).
    """
    violations = []

    if not evidence:
        if claimed.reliability > 0:
            violations.append("Non-zero reliability claimed with no evidence")
        return len(violations) == 0, violations

    min_evidence_f = min(e.formality for e in evidence)
    if claimed.formality > min_evidence_f:
        violations.append(
            f"Claimed F{claimed.formality.value} exceeds evidence max F{min_evidence_f.value}"
        )

    min_evidence_r = min(e.reliability for e in evidence)
    if claimed.reliability > min_evidence_r + 0.01:
        violations.append(
            f"Claimed R={claimed.reliability:.2f} exceeds WLNK bound {min_evidence_r:.2f}"
        )

    evidence_scope = evidence[0].claim_scope
    for e in evidence[1:]:
        evidence_scope = evidence_scope.intersect(e.claim_scope)

    if not evidence_scope.contains(claimed.claim_scope):
        violations.append("Claimed scope exceeds evidence intersection")

    return len(violations) == 0, violations

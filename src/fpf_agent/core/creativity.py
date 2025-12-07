"""
FPF Creativity Patterns (C.17-C.19)

Implements NQD (Novelty-Quality-Diversity) search and Explore-Exploit governance.

Key FPF principles:
- C.17: Creativity is measurable on N-Q-D dimensions
- C.18: NQD search balances novelty vs quality vs diversity
- C.19: Explore-Exploit governance controls resource allocation

The creativity module supports the Abduction phase of ADI cycle,
where multiple hypotheses are generated and evaluated.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Generic, TypeVar, Callable
import math


# =============================================================================
# NQD SCORING (C.17)
# =============================================================================

@dataclass
class NQDScore:
    """
    C.17: Novelty-Quality-Diversity score for creative output.

    Each dimension is [0, 1]:
    - Novelty (N): How new/original is this?
    - Quality (Q): How good/useful is this?
    - Diversity (D): How different from existing candidates?
    """
    novelty: float
    quality: float
    diversity: float

    # Optional sub-scores for quality
    use_value: float = 0.0      # Practical utility
    surprise: float = 0.0       # Unexpectedness
    constraint_fit: float = 0.0 # How well it fits constraints

    def __post_init__(self):
        """Validate scores are in [0, 1]."""
        for name, value in [
            ("novelty", self.novelty),
            ("quality", self.quality),
            ("diversity", self.diversity),
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def composite(self) -> float:
        """Simple average composite score."""
        return (self.novelty + self.quality + self.diversity) / 3

    def weighted_composite(
        self,
        w_novelty: float = 1.0,
        w_quality: float = 1.0,
        w_diversity: float = 1.0,
    ) -> float:
        """Weighted composite score."""
        total_weight = w_novelty + w_quality + w_diversity
        if total_weight == 0:
            return 0.0
        return (
            w_novelty * self.novelty +
            w_quality * self.quality +
            w_diversity * self.diversity
        ) / total_weight

    def dominates(self, other: "NQDScore") -> bool:
        """
        Pareto dominance: self dominates other if better or equal on all,
        and strictly better on at least one.
        """
        better_or_equal = (
            self.novelty >= other.novelty and
            self.quality >= other.quality and
            self.diversity >= other.diversity
        )
        strictly_better = (
            self.novelty > other.novelty or
            self.quality > other.quality or
            self.diversity > other.diversity
        )
        return better_or_equal and strictly_better


# =============================================================================
# CREATIVE CANDIDATE
# =============================================================================

T = TypeVar('T')


@dataclass
class CreativeCandidate(Generic[T]):
    """A candidate solution with NQD scoring."""
    candidate_id: str
    content: T
    score: NQDScore

    # Metadata
    generation_method: str = ""  # How was this generated?
    parent_candidates: list[str] = field(default_factory=list)  # Evolution lineage
    created_at: datetime = field(default_factory=datetime.now)

    # ADI integration
    testable_predictions: list[str] = field(default_factory=list)
    plausibility_rationale: str = ""

    @property
    def is_pareto_optimal(self) -> bool:
        """Whether this candidate is on the Pareto frontier (set externally)."""
        return getattr(self, "_pareto_optimal", False)

    def mark_pareto_optimal(self, value: bool = True) -> None:
        """Mark as Pareto optimal."""
        self._pareto_optimal = value


# =============================================================================
# NQD SEARCH CONFIG (C.18)
# =============================================================================

@dataclass
class NQDSearchConfig:
    """
    C.18: Configuration for NQD search.

    Controls how candidates are generated and selected.
    """
    # Budget
    exploration_budget: int = 10  # Max candidates to generate
    max_iterations: int = 5       # Max refinement iterations

    # Weights for composite scoring
    novelty_weight: float = 1.0
    quality_weight: float = 1.5   # Usually prioritize quality
    diversity_weight: float = 0.5

    # Thresholds
    quality_threshold: float = 0.5    # Minimum quality to keep
    novelty_threshold: float = 0.3    # Minimum novelty to keep
    diversity_threshold: float = 0.2  # Minimum diversity to keep

    # Selection strategy
    pareto_only: bool = True          # Only keep Pareto-optimal candidates
    keep_top_k: int = 5               # Keep at most K candidates

    # Diversity computation
    diversity_metric: str = "jaccard"  # "jaccard", "cosine", "hamming"

    def validate(self) -> list[str]:
        """Validate config, return list of issues."""
        issues = []
        if self.exploration_budget < 1:
            issues.append("exploration_budget must be >= 1")
        if self.quality_threshold > 1 or self.quality_threshold < 0:
            issues.append("quality_threshold must be in [0, 1]")
        if self.novelty_weight + self.quality_weight + self.diversity_weight == 0:
            issues.append("At least one weight must be > 0")
        return issues


# =============================================================================
# NQD SCORER
# =============================================================================

class NQDScorer(ABC, Generic[T]):
    """
    Abstract base for scoring candidates on NQD dimensions.

    Implement this for your specific domain/candidate type.
    """

    @abstractmethod
    def score_novelty(self, candidate: T, existing: list[T]) -> float:
        """Score novelty relative to existing candidates. [0, 1]"""
        pass

    @abstractmethod
    def score_quality(self, candidate: T) -> float:
        """Score intrinsic quality of candidate. [0, 1]"""
        pass

    @abstractmethod
    def score_diversity(self, candidate: T, population: list[T]) -> float:
        """Score diversity contribution to population. [0, 1]"""
        pass

    def compute_score(
        self,
        candidate: T,
        existing: list[T],
        population: list[T] | None = None
    ) -> NQDScore:
        """Compute full NQD score for a candidate."""
        if population is None:
            population = existing

        return NQDScore(
            novelty=self.score_novelty(candidate, existing),
            quality=self.score_quality(candidate),
            diversity=self.score_diversity(candidate, population),
        )


# =============================================================================
# STRING/TEXT SCORER (Default Implementation)
# =============================================================================

class TextNQDScorer(NQDScorer[str]):
    """NQD scorer for text/string candidates using simple heuristics."""

    def __init__(self, reference_corpus: list[str] | None = None):
        self.reference_corpus = reference_corpus or []

    def score_novelty(self, candidate: str, existing: list[str]) -> float:
        """
        Score novelty based on token overlap with existing.

        Lower overlap = higher novelty.
        """
        if not existing:
            return 1.0  # First candidate is maximally novel

        candidate_tokens = set(candidate.lower().split())
        if not candidate_tokens:
            return 0.0

        # Compute average Jaccard distance
        distances = []
        for ex in existing:
            ex_tokens = set(ex.lower().split())
            if not ex_tokens:
                continue
            intersection = len(candidate_tokens & ex_tokens)
            union = len(candidate_tokens | ex_tokens)
            jaccard = intersection / union if union > 0 else 0
            distances.append(1 - jaccard)  # Distance = 1 - similarity

        return sum(distances) / len(distances) if distances else 1.0

    def score_quality(self, candidate: str) -> float:
        """
        Score quality based on heuristics:
        - Length (too short = low quality)
        - Structure (has punctuation, proper sentences)
        - Specificity (contains specific terms vs vague)
        """
        if not candidate:
            return 0.0

        score = 0.5  # Base score

        # Length factor
        word_count = len(candidate.split())
        if word_count < 5:
            score -= 0.2
        elif word_count > 10:
            score += 0.1

        # Structure factor
        if "." in candidate or ":" in candidate:
            score += 0.1
        if candidate[0].isupper():
            score += 0.05

        # Specificity factor (contains numbers or technical terms)
        if any(c.isdigit() for c in candidate):
            score += 0.1
        if any(word in candidate.lower() for word in ["because", "therefore", "since", "given"]):
            score += 0.15

        return max(0.0, min(1.0, score))

    def score_diversity(self, candidate: str, population: list[str]) -> float:
        """
        Score diversity as minimum distance to any population member.

        Higher = more diverse (further from all existing).
        """
        if not population:
            return 1.0

        candidate_tokens = set(candidate.lower().split())
        if not candidate_tokens:
            return 0.0

        min_distance = 1.0
        for member in population:
            member_tokens = set(member.lower().split())
            if not member_tokens:
                continue
            intersection = len(candidate_tokens & member_tokens)
            union = len(candidate_tokens | member_tokens)
            similarity = intersection / union if union > 0 else 0
            distance = 1 - similarity
            min_distance = min(min_distance, distance)

        return min_distance


# =============================================================================
# HYPOTHESIS SCORER
# =============================================================================

@dataclass
class HypothesisCandidate:
    """A hypothesis candidate for NQD scoring."""
    hypothesis_id: str
    statement: str
    anomaly_addressed: str
    testable_predictions: list[str]
    plausibility_score: float
    generation_method: str = ""


class HypothesisNQDScorer(NQDScorer[HypothesisCandidate]):
    """NQD scorer specialized for hypothesis candidates."""

    def score_novelty(
        self,
        candidate: HypothesisCandidate,
        existing: list[HypothesisCandidate]
    ) -> float:
        """Score novelty based on statement similarity."""
        if not existing:
            return 1.0

        candidate_words = set(candidate.statement.lower().split())
        if not candidate_words:
            return 0.0

        similarities = []
        for ex in existing:
            ex_words = set(ex.statement.lower().split())
            if not ex_words:
                continue
            intersection = len(candidate_words & ex_words)
            union = len(candidate_words | ex_words)
            similarities.append(intersection / union if union > 0 else 0)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        return 1.0 - avg_similarity

    def score_quality(self, candidate: HypothesisCandidate) -> float:
        """
        Score quality based on:
        - Plausibility score (from generation)
        - Number of testable predictions
        - Specificity of statement
        """
        score = candidate.plausibility_score * 0.5  # Base from plausibility

        # Testable predictions bonus
        pred_count = len(candidate.testable_predictions)
        if pred_count >= 3:
            score += 0.25
        elif pred_count >= 1:
            score += 0.15

        # Specificity bonus
        word_count = len(candidate.statement.split())
        if 10 <= word_count <= 50:
            score += 0.15
        if any(c.isdigit() for c in candidate.statement):
            score += 0.1

        return max(0.0, min(1.0, score))

    def score_diversity(
        self,
        candidate: HypothesisCandidate,
        population: list[HypothesisCandidate]
    ) -> float:
        """Score diversity based on prediction overlap."""
        if not population:
            return 1.0

        candidate_preds = set(p.lower() for p in candidate.testable_predictions)
        if not candidate_preds:
            return 0.5  # Neutral if no predictions

        diversities = []
        for member in population:
            member_preds = set(p.lower() for p in member.testable_predictions)
            if not member_preds:
                continue
            intersection = len(candidate_preds & member_preds)
            union = len(candidate_preds | member_preds)
            overlap = intersection / union if union > 0 else 0
            diversities.append(1.0 - overlap)

        return sum(diversities) / len(diversities) if diversities else 1.0


# =============================================================================
# PARETO SELECTION
# =============================================================================

def compute_pareto_frontier(
    candidates: list[CreativeCandidate[T]]
) -> list[CreativeCandidate[T]]:
    """
    Compute Pareto-optimal frontier from candidates.

    A candidate is Pareto-optimal if no other candidate dominates it
    (i.e., is better or equal on all NQD dimensions and strictly better on at least one).
    """
    if not candidates:
        return []

    frontier = []

    for candidate in candidates:
        is_dominated = False

        for other in candidates:
            if other.candidate_id == candidate.candidate_id:
                continue
            if other.score.dominates(candidate.score):
                is_dominated = True
                break

        if not is_dominated:
            candidate.mark_pareto_optimal(True)
            frontier.append(candidate)

    return frontier


def select_diverse_subset(
    candidates: list[CreativeCandidate[T]],
    k: int,
    scorer: NQDScorer[T],
) -> list[CreativeCandidate[T]]:
    """
    Select k diverse candidates using greedy max-min diversity.

    Starts with highest quality, then adds candidates that maximize
    minimum distance to selected set.
    """
    if len(candidates) <= k:
        return candidates

    if not candidates:
        return []

    # Sort by quality descending
    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.score.quality,
        reverse=True
    )

    # Greedy selection
    selected = [sorted_candidates[0]]
    remaining = sorted_candidates[1:]

    while len(selected) < k and remaining:
        # Find candidate with max min-distance to selected
        best_idx = 0
        best_min_dist = -1

        for i, candidate in enumerate(remaining):
            min_dist = min(
                1 - _jaccard_similarity(
                    str(candidate.content),
                    str(sel.content)
                )
                for sel in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def _jaccard_similarity(s1: str, s2: str) -> float:
    """Simple Jaccard similarity for strings."""
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


# =============================================================================
# NQD SEARCH ENGINE
# =============================================================================

class NQDSearchEngine(Generic[T]):
    """
    C.18: Main NQD search engine for generating and selecting candidates.

    Usage:
        config = NQDSearchConfig(exploration_budget=10)
        engine = NQDSearchEngine(config, scorer)

        # Add initial candidates
        engine.add_candidate(candidate1)
        engine.add_candidate(candidate2)

        # Get best candidates
        best = engine.get_best_candidates()
    """

    def __init__(
        self,
        config: NQDSearchConfig,
        scorer: NQDScorer[T],
    ):
        self.config = config
        self.scorer = scorer
        self._candidates: list[CreativeCandidate[T]] = []
        self._generation_count = 0

    def add_candidate(
        self,
        content: T,
        generation_method: str = "manual",
        parent_ids: list[str] | None = None,
    ) -> CreativeCandidate[T]:
        """
        Add a candidate and compute its NQD score.

        Returns the wrapped CreativeCandidate.
        """
        self._generation_count += 1

        # Compute NQD score
        existing_contents = [c.content for c in self._candidates]
        score = self.scorer.compute_score(
            content,
            existing_contents,
            existing_contents,
        )

        candidate = CreativeCandidate(
            candidate_id=f"cand_{self._generation_count}",
            content=content,
            score=score,
            generation_method=generation_method,
            parent_candidates=parent_ids or [],
        )

        self._candidates.append(candidate)
        return candidate

    def filter_by_thresholds(self) -> list[CreativeCandidate[T]]:
        """Filter candidates by quality/novelty/diversity thresholds."""
        return [
            c for c in self._candidates
            if (c.score.quality >= self.config.quality_threshold and
                c.score.novelty >= self.config.novelty_threshold and
                c.score.diversity >= self.config.diversity_threshold)
        ]

    def get_pareto_frontier(self) -> list[CreativeCandidate[T]]:
        """Get Pareto-optimal candidates."""
        filtered = self.filter_by_thresholds()
        return compute_pareto_frontier(filtered)

    def get_best_candidates(self) -> list[CreativeCandidate[T]]:
        """
        Get best candidates according to config.

        If pareto_only: returns Pareto frontier (up to keep_top_k)
        Otherwise: returns top K by weighted composite score
        """
        if self.config.pareto_only:
            frontier = self.get_pareto_frontier()
            if len(frontier) <= self.config.keep_top_k:
                return frontier
            return select_diverse_subset(frontier, self.config.keep_top_k, self.scorer)
        else:
            filtered = self.filter_by_thresholds()
            sorted_candidates = sorted(
                filtered,
                key=lambda c: c.score.weighted_composite(
                    self.config.novelty_weight,
                    self.config.quality_weight,
                    self.config.diversity_weight,
                ),
                reverse=True
            )
            return sorted_candidates[:self.config.keep_top_k]

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.config.exploration_budget - len(self._candidates))

    def get_statistics(self) -> dict[str, Any]:
        """Get search statistics."""
        if not self._candidates:
            return {"count": 0}

        scores = [c.score for c in self._candidates]
        return {
            "count": len(self._candidates),
            "avg_novelty": sum(s.novelty for s in scores) / len(scores),
            "avg_quality": sum(s.quality for s in scores) / len(scores),
            "avg_diversity": sum(s.diversity for s in scores) / len(scores),
            "pareto_count": len(self.get_pareto_frontier()),
            "budget_remaining": self.budget_remaining,
        }


# =============================================================================
# EXPLORE-EXPLOIT GOVERNANCE (C.19)
# =============================================================================

class ExploreExploitPhase(Enum):
    """Current phase of explore-exploit cycle."""
    EXPLORE = "explore"       # Generating diverse candidates
    EXPLOIT = "exploit"       # Refining best candidates
    BALANCED = "balanced"     # Mixed strategy


@dataclass
class ExploreExploitPolicy:
    """
    C.19: Policy governing explore vs exploit resource allocation.

    Exploration: Generate new diverse candidates (high N, D)
    Exploitation: Refine best candidates (high Q)
    """
    # Resource allocation
    exploration_share: float = 0.3  # 30% on exploration
    total_budget: int = 100         # Total resource units

    # Phase switching
    switch_to_exploit_threshold: float = 0.7  # Quality threshold to switch
    min_exploration_rounds: int = 3           # Min exploration before switching
    max_exploration_rounds: int = 10          # Max exploration before forcing switch

    # Current state
    current_phase: ExploreExploitPhase = ExploreExploitPhase.EXPLORE
    exploration_rounds: int = 0
    exploitation_rounds: int = 0

    # Rationale tracking
    rationale: str = ""

    @property
    def exploration_budget(self) -> int:
        """Budget allocated to exploration."""
        return int(self.total_budget * self.exploration_share)

    @property
    def exploitation_budget(self) -> int:
        """Budget allocated to exploitation."""
        return self.total_budget - self.exploration_budget


class ExploreExploitController:
    """
    C.19: Controller for explore-exploit governance.

    Tracks decisions and manages phase transitions.
    """

    def __init__(self, policy: ExploreExploitPolicy):
        self.policy = policy
        self._decision_log: list[dict[str, Any]] = []
        self._resources_used = {"explore": 0, "exploit": 0}

    def should_explore(self) -> bool:
        """Whether to explore (generate new) vs exploit (refine best)."""
        # Always explore during minimum exploration rounds
        if self.policy.exploration_rounds < self.policy.min_exploration_rounds:
            return True

        # Force exploit after max exploration
        if self.policy.exploration_rounds >= self.policy.max_exploration_rounds:
            return False

        # Otherwise, based on current phase
        return self.policy.current_phase in [
            ExploreExploitPhase.EXPLORE,
            ExploreExploitPhase.BALANCED
        ]

    def record_decision(
        self,
        action: str,
        phase: ExploreExploitPhase,
        resource_cost: int,
        outcome_quality: float,
        notes: str = "",
    ) -> None:
        """Record an explore/exploit decision for auditability."""
        self._decision_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "phase": phase.value,
            "resource_cost": resource_cost,
            "outcome_quality": outcome_quality,
            "notes": notes,
        })

        if phase == ExploreExploitPhase.EXPLORE:
            self.policy.exploration_rounds += 1
            self._resources_used["explore"] += resource_cost
        else:
            self.policy.exploitation_rounds += 1
            self._resources_used["exploit"] += resource_cost

    def update_phase(self, best_quality: float) -> ExploreExploitPhase:
        """
        Update phase based on best candidate quality.

        Switches to exploit if quality threshold reached.
        """
        old_phase = self.policy.current_phase

        if best_quality >= self.policy.switch_to_exploit_threshold:
            self.policy.current_phase = ExploreExploitPhase.EXPLOIT
            self.policy.rationale = f"Quality {best_quality:.2f} >= threshold"
        elif self.policy.exploration_rounds >= self.policy.max_exploration_rounds:
            self.policy.current_phase = ExploreExploitPhase.EXPLOIT
            self.policy.rationale = "Max exploration rounds reached"
        elif self.policy.exploration_rounds >= self.policy.min_exploration_rounds:
            # Could switch to balanced
            if best_quality >= 0.5:
                self.policy.current_phase = ExploreExploitPhase.BALANCED
                self.policy.rationale = "Moderate quality, balancing E/E"

        if old_phase != self.policy.current_phase:
            self._decision_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "phase_transition",
                "from_phase": old_phase.value,
                "to_phase": self.policy.current_phase.value,
                "rationale": self.policy.rationale,
            })

        return self.policy.current_phase

    def get_remaining_budget(self) -> dict[str, int]:
        """Get remaining budget for explore and exploit."""
        return {
            "explore": self.policy.exploration_budget - self._resources_used["explore"],
            "exploit": self.policy.exploitation_budget - self._resources_used["exploit"],
            "total": (
                self.policy.total_budget -
                self._resources_used["explore"] -
                self._resources_used["exploit"]
            ),
        }

    def get_decision_log(self) -> list[dict[str, Any]]:
        """Get full decision log for auditability."""
        return list(self._decision_log)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of E/E state."""
        return {
            "current_phase": self.policy.current_phase.value,
            "exploration_rounds": self.policy.exploration_rounds,
            "exploitation_rounds": self.policy.exploitation_rounds,
            "resources_used": self._resources_used.copy(),
            "remaining_budget": self.get_remaining_budget(),
            "decision_count": len(self._decision_log),
        }


# =============================================================================
# INTEGRATION WITH ADI CYCLE
# =============================================================================

@dataclass
class CreativitySession:
    """
    A creativity session integrating NQD search with E/E governance.

    Used during Abduction phase of ADI cycle.
    """
    session_id: str
    anomaly: str                     # What anomaly we're explaining
    config: NQDSearchConfig
    ee_policy: ExploreExploitPolicy

    # State
    started_at: datetime = field(default_factory=datetime.now)
    hypotheses: list[HypothesisCandidate] = field(default_factory=list)
    selected_hypothesis: HypothesisCandidate | None = None

    def is_complete(self) -> bool:
        """Whether session has produced a selected hypothesis."""
        return self.selected_hypothesis is not None


def create_creativity_session(
    anomaly: str,
    exploration_budget: int = 10,
    quality_threshold: float = 0.5,
) -> CreativitySession:
    """Create a new creativity session with default configs."""
    return CreativitySession(
        session_id=f"creative_{datetime.now().timestamp()}",
        anomaly=anomaly,
        config=NQDSearchConfig(
            exploration_budget=exploration_budget,
            quality_threshold=quality_threshold,
        ),
        ee_policy=ExploreExploitPolicy(
            total_budget=exploration_budget * 2,
        ),
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def score_hypothesis_set(
    hypotheses: list[HypothesisCandidate],
) -> list[tuple[HypothesisCandidate, NQDScore]]:
    """Score a set of hypotheses and return with scores."""
    scorer = HypothesisNQDScorer()
    results = []

    for i, hyp in enumerate(hypotheses):
        existing = hypotheses[:i]
        score = scorer.compute_score(hyp, existing, hypotheses)
        results.append((hyp, score))

    return results


def select_best_hypothesis(
    hypotheses: list[HypothesisCandidate],
    prefer_quality: bool = True,
) -> HypothesisCandidate | None:
    """Select best hypothesis from a set."""
    if not hypotheses:
        return None

    scored = score_hypothesis_set(hypotheses)

    if prefer_quality:
        # Sort by quality, then novelty
        sorted_scored = sorted(
            scored,
            key=lambda x: (x[1].quality, x[1].novelty),
            reverse=True
        )
    else:
        # Sort by composite
        sorted_scored = sorted(
            scored,
            key=lambda x: x[1].composite,
            reverse=True
        )

    return sorted_scored[0][0] if sorted_scored else None

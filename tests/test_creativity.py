"""
Tests for Creativity Patterns (C.17-C.19)
"""

import pytest

from fpf_agent.core.creativity import (
    NQDScore,
    CreativeCandidate,
    NQDSearchConfig,
    NQDSearchEngine,
    TextNQDScorer,
    HypothesisNQDScorer,
    HypothesisCandidate,
    ExploreExploitPhase,
    ExploreExploitPolicy,
    ExploreExploitController,
    compute_pareto_frontier,
    select_diverse_subset,
    score_hypothesis_set,
    select_best_hypothesis,
)


class TestNQDScore:
    """Test NQD score dataclass."""

    def test_score_bounds(self):
        """Scores must be in [0, 1]."""
        score = NQDScore(novelty=0.5, quality=0.7, diversity=0.3)
        assert 0 <= score.novelty <= 1
        assert 0 <= score.quality <= 1
        assert 0 <= score.diversity <= 1

    def test_score_out_of_bounds_error(self):
        """Out of bounds scores should raise error."""
        with pytest.raises(ValueError):
            NQDScore(novelty=1.5, quality=0.5, diversity=0.5)

    def test_composite_score(self):
        """Composite score is average of N, Q, D."""
        score = NQDScore(novelty=0.6, quality=0.9, diversity=0.3)
        expected = (0.6 + 0.9 + 0.3) / 3
        assert score.composite == pytest.approx(expected, rel=0.01)

    def test_weighted_composite(self):
        """Can compute weighted composite."""
        score = NQDScore(novelty=0.5, quality=0.8, diversity=0.2)
        # Weight quality more
        weighted = score.weighted_composite(w_novelty=1, w_quality=2, w_diversity=1)
        expected = (0.5 * 1 + 0.8 * 2 + 0.2 * 1) / 4
        assert weighted == pytest.approx(expected, rel=0.01)

    def test_pareto_dominance(self):
        """Test Pareto dominance logic."""
        # Score A dominates Score B
        score_a = NQDScore(novelty=0.8, quality=0.8, diversity=0.8)
        score_b = NQDScore(novelty=0.5, quality=0.5, diversity=0.5)
        assert score_a.dominates(score_b)
        assert not score_b.dominates(score_a)

    def test_pareto_incomparable(self):
        """Incomparable scores don't dominate each other."""
        score_a = NQDScore(novelty=0.9, quality=0.3, diversity=0.5)
        score_b = NQDScore(novelty=0.3, quality=0.9, diversity=0.5)
        assert not score_a.dominates(score_b)
        assert not score_b.dominates(score_a)


class TestTextNQDScorer:
    """Test text-based NQD scorer."""

    def test_novelty_first_candidate(self):
        """First candidate should have high novelty."""
        scorer = TextNQDScorer()
        novelty = scorer.score_novelty("A novel idea", existing=[])
        assert novelty == 1.0

    def test_novelty_decreases_with_similar(self):
        """Novelty should decrease with similar existing."""
        scorer = TextNQDScorer()
        existing = ["A similar idea about databases"]
        novelty = scorer.score_novelty("A similar idea about databases", existing)
        assert novelty < 0.5  # Low novelty for exact match

    def test_quality_scoring(self):
        """Quality should reflect text structure."""
        scorer = TextNQDScorer()

        short_text = "ok"
        long_text = "The system processes requests through a queue mechanism. Because of buffering."

        quality_short = scorer.score_quality(short_text)
        quality_long = scorer.score_quality(long_text)

        assert quality_long > quality_short

    def test_diversity_scoring(self):
        """Diversity measures distance from population."""
        scorer = TextNQDScorer()
        population = [
            "Machine learning model",
            "Neural network classifier",
        ]

        similar = "Machine learning classifier"
        different = "Database optimization strategy"

        div_similar = scorer.score_diversity(similar, population)
        div_different = scorer.score_diversity(different, population)

        assert div_different > div_similar


class TestHypothesisNQDScorer:
    """Test hypothesis-specific scorer."""

    def test_hypothesis_quality(self):
        """Quality depends on predictions and plausibility."""
        scorer = HypothesisNQDScorer()

        good_hyp = HypothesisCandidate(
            hypothesis_id="h1",
            statement="Memory leak causes OOM errors due to unclosed connections.",
            anomaly_addressed="High memory usage",
            testable_predictions=["Memory grows", "Connections remain open", "OOM at threshold"],
            plausibility_score=0.9,
        )

        weak_hyp = HypothesisCandidate(
            hypothesis_id="h2",
            statement="Something is wrong",
            anomaly_addressed="Issues",
            testable_predictions=[],
            plausibility_score=0.2,
        )

        quality_good = scorer.score_quality(good_hyp)
        quality_weak = scorer.score_quality(weak_hyp)

        assert quality_good > quality_weak


class TestNQDSearchEngine:
    """Test NQD search engine."""

    def test_add_candidates(self):
        """Can add candidates to engine."""
        config = NQDSearchConfig(exploration_budget=10)
        engine = NQDSearchEngine(config, TextNQDScorer())

        engine.add_candidate("First idea")
        engine.add_candidate("Second idea")

        assert engine.candidate_count == 2

    def test_budget_tracking(self):
        """Engine tracks remaining budget."""
        config = NQDSearchConfig(exploration_budget=5)
        engine = NQDSearchEngine(config, TextNQDScorer())

        engine.add_candidate("Idea 1")
        engine.add_candidate("Idea 2")

        assert engine.budget_remaining == 3

    def test_pareto_frontier(self):
        """Engine can compute Pareto frontier."""
        config = NQDSearchConfig()
        engine = NQDSearchEngine(config, TextNQDScorer())

        engine.add_candidate("Short")
        engine.add_candidate("A medium length idea about systems")
        engine.add_candidate("A very different and longer idea about completely unrelated topics")

        frontier = engine.get_pareto_frontier()
        assert len(frontier) >= 1


class TestParetoSelection:
    """Test Pareto frontier computation."""

    def test_pareto_with_dominant(self):
        """Dominated candidates should be excluded."""
        candidates = [
            CreativeCandidate(
                candidate_id="c1",
                content="dominant",
                score=NQDScore(novelty=0.9, quality=0.9, diversity=0.9),
            ),
            CreativeCandidate(
                candidate_id="c2",
                content="dominated",
                score=NQDScore(novelty=0.3, quality=0.3, diversity=0.3),
            ),
        ]

        frontier = compute_pareto_frontier(candidates)
        assert len(frontier) == 1
        assert frontier[0].candidate_id == "c1"

    def test_pareto_incomparable(self):
        """Incomparable candidates should all be on frontier."""
        candidates = [
            CreativeCandidate(
                candidate_id="c1",
                content="high novelty",
                score=NQDScore(novelty=0.9, quality=0.3, diversity=0.5),
            ),
            CreativeCandidate(
                candidate_id="c2",
                content="high quality",
                score=NQDScore(novelty=0.3, quality=0.9, diversity=0.5),
            ),
        ]

        frontier = compute_pareto_frontier(candidates)
        assert len(frontier) == 2


class TestExploreExploitPolicy:
    """Test E/E policy."""

    def test_budget_allocation(self):
        """Budget should be allocated between E and E."""
        policy = ExploreExploitPolicy(
            exploration_share=0.4,
            total_budget=100,
        )

        assert policy.exploration_budget == 40
        assert policy.exploitation_budget == 60


class TestExploreExploitController:
    """Test E/E controller."""

    def test_initial_explore(self):
        """Should start in explore mode."""
        policy = ExploreExploitPolicy(min_exploration_rounds=3)
        controller = ExploreExploitController(policy)

        assert controller.should_explore()

    def test_phase_transition(self):
        """Should transition to exploit at quality threshold."""
        policy = ExploreExploitPolicy(
            min_exploration_rounds=1,
            switch_to_exploit_threshold=0.7,
        )
        controller = ExploreExploitController(policy)

        # Record some exploration
        controller.record_decision(
            action="generate",
            phase=ExploreExploitPhase.EXPLORE,
            resource_cost=1,
            outcome_quality=0.5,
        )

        # Update phase with high quality
        new_phase = controller.update_phase(best_quality=0.8)
        assert new_phase == ExploreExploitPhase.EXPLOIT

    def test_decision_logging(self):
        """All decisions should be logged."""
        policy = ExploreExploitPolicy()
        controller = ExploreExploitController(policy)

        controller.record_decision(
            action="explore",
            phase=ExploreExploitPhase.EXPLORE,
            resource_cost=1,
            outcome_quality=0.5,
        )

        log = controller.get_decision_log()
        assert len(log) == 1
        assert log[0]["action"] == "explore"


class TestConvenienceFunctions:
    """Test helper functions."""

    def test_score_hypothesis_set(self):
        """Can score a set of hypotheses."""
        hypotheses = [
            HypothesisCandidate(
                hypothesis_id="h1",
                statement="Hypothesis one",
                anomaly_addressed="Problem",
                testable_predictions=["P1"],
                plausibility_score=0.7,
            ),
            HypothesisCandidate(
                hypothesis_id="h2",
                statement="Hypothesis two",
                anomaly_addressed="Problem",
                testable_predictions=["P2"],
                plausibility_score=0.6,
            ),
        ]

        scored = score_hypothesis_set(hypotheses)
        assert len(scored) == 2
        assert all(isinstance(s[1], NQDScore) for s in scored)

    def test_select_best_hypothesis(self):
        """Can select best hypothesis from set."""
        hypotheses = [
            HypothesisCandidate(
                hypothesis_id="h1",
                statement="Weak hypothesis",
                anomaly_addressed="Issue",
                testable_predictions=[],
                plausibility_score=0.2,
            ),
            HypothesisCandidate(
                hypothesis_id="h2",
                statement="Strong hypothesis with detailed predictions and evidence.",
                anomaly_addressed="Critical issue",
                testable_predictions=["P1", "P2", "P3"],
                plausibility_score=0.9,
            ),
        ]

        best = select_best_hypothesis(hypotheses)
        assert best is not None
        assert best.hypothesis_id == "h2"

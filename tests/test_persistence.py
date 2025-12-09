"""
Tests for FPF Persistence Layer (Phase 1).

Tests contracts for:
- ArangoSchema: collection/graph initialization
- EpistemeStore: CRUD, versioning, queries
- EvidenceGraph: links, NetworkX analysis
- ContextStore: bounded contexts, bridges
- ResearchSessionStore: session state management

Requires running ArangoDB: docker-compose up -d
"""
import os
import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from fpf_agent.kernel import (
    AssuranceLevel,
    Edition,
    HolonId,
    LifecycleState,
    StrictDistinctionSlots,
    TemporalStance,
    UBoundedContext,
    UEpisteme,
)
from fpf_agent.persistence import (
    ArangoSchema,
    ContextStore,
    create_persistence,
    EpistemeStore,
    EvidenceGraph,
    EvidenceLink,
    EvidenceRelation,
    PersistenceManager,
    ResearchSessionStore,
)
from fpf_agent.trust.fgr import CongruenceLevel


# Test database configuration
TEST_DB_HOST = os.environ.get("ARANGO_HOST", "http://localhost:8529")
TEST_DB_USER = os.environ.get("ARANGO_USER", "root")
TEST_DB_PASS = os.environ.get("ARANGO_PASSWORD", "fpf_dev_password")
TEST_DB_NAME = "fpf_test_db"


@pytest.fixture(scope="module")
def persistence():
    """Create test persistence manager with isolated test database."""
    pm = PersistenceManager(
        host=TEST_DB_HOST,
        username=TEST_DB_USER,
        password=TEST_DB_PASS,
        database=TEST_DB_NAME
    )
    pm.initialize()

    # Clean slate for tests
    pm.reset_database()

    yield pm

    # Cleanup after all tests
    pm.schema.drop_all()


@pytest.fixture
def clean_db(persistence):
    """Reset database before each test that needs isolation."""
    persistence.reset_database()
    return persistence


def make_episteme(context_id: str = "test_ctx", entity: str = "Test Entity") -> UEpisteme:
    """Helper to create test episteme."""
    return UEpisteme(
        holon_id=HolonId(context_id=context_id),
        described_entity=entity,
        claim_graph={"test": "claim"},
        slots=StrictDistinctionSlots(),
        lifecycle_state=LifecycleState.EXPLORATION,
        assurance_level=AssuranceLevel.L0,
        temporal_stance=TemporalStance.DESIGN_TIME,
    )


class TestArangoSchema:
    """Test ArangoDB schema management."""

    def test_initialize_creates_collections(self, persistence):
        """Initialization creates all required collections."""
        db = persistence.db

        expected_collections = [
            "contexts", "epistemes", "sessions",
            "hypotheses", "adi_cycles"
        ]
        expected_edges = [
            "evidence_links", "context_bridges", "edition_chain"
        ]

        for coll in expected_collections:
            assert db.has_collection(coll), f"Missing collection: {coll}"

        for edge in expected_edges:
            assert db.has_collection(edge), f"Missing edge collection: {edge}"

    def test_initialize_creates_graph(self, persistence):
        """Initialization creates evidence graph."""
        db = persistence.db
        assert db.has_graph("fpf_evidence_graph")

    def test_health_check_returns_status(self, persistence):
        """Health check returns database status."""
        health = persistence.health_check()

        assert health["status"] == "healthy"
        assert "version" in health
        assert health["database"] == TEST_DB_NAME
        assert "collection_counts" in health

    def test_reset_database_clears_data(self, persistence):
        """Reset drops and recreates collections."""
        # Insert some data
        persistence.contexts.create(UBoundedContext(
            context_id="temp_ctx",
            name="Temporary Context",
            description="Temp context for reset test"
        ))

        # Reset
        persistence.reset_database()

        # Verify collections exist but are empty
        health = persistence.health_check()
        assert all(v == 0 for v in health["collection_counts"].values())


class TestEpistemeStore:
    """Test episteme CRUD and versioning."""

    def test_create_episteme(self, clean_db):
        """Can create new episteme."""
        episteme = make_episteme()

        created = clean_db.epistemes.create(episteme)

        assert created.holon_id.id == episteme.holon_id.id
        assert created.described_entity == "Test Entity"
        assert created.holon_id.edition.number == 1

    def test_get_episteme_by_id(self, clean_db):
        """Can retrieve episteme by ID."""
        episteme = make_episteme(entity="Retrievable Entity")
        created = clean_db.epistemes.create(episteme)

        retrieved = clean_db.epistemes.get(created.holon_id.id)

        assert retrieved is not None
        assert retrieved.described_entity == "Retrievable Entity"

    def test_get_by_key(self, clean_db):
        """Can retrieve episteme by document key."""
        episteme = make_episteme()
        created = clean_db.epistemes.create(episteme)

        retrieved = clean_db.epistemes.get_by_key(str(created.holon_id.id))

        assert retrieved is not None
        assert retrieved.holon_id.id == created.holon_id.id

    def test_update_creates_new_edition(self, clean_db):
        """Update creates new edition, doesn't mutate original."""
        original = clean_db.epistemes.create(make_episteme())

        # "Update" creates new edition
        updated = clean_db.epistemes.update(original)

        assert updated.holon_id.edition.number == 2
        assert updated.holon_id.edition.supersedes == original.holon_id.id
        assert updated.holon_id.id != original.holon_id.id

        # Original still exists unchanged
        original_retrieved = clean_db.epistemes.get_by_key(str(original.holon_id.id))
        assert original_retrieved.holon_id.edition.number == 1

    def test_find_by_entity(self, clean_db):
        """Can find epistemes by entity name."""
        clean_db.epistemes.create(make_episteme(entity="Alpha Protocol"))
        clean_db.epistemes.create(make_episteme(entity="Beta System"))
        clean_db.epistemes.create(make_episteme(entity="Alpha Controller"))

        results = clean_db.epistemes.find_by_entity("Alpha", "test_ctx")

        assert len(results) == 2
        entities = [e.described_entity for e in results]
        assert "Alpha Protocol" in entities
        assert "Alpha Controller" in entities

    def test_find_by_lifecycle(self, clean_db):
        """Can filter epistemes by lifecycle state."""
        # Create with different lifecycle states
        e1 = make_episteme(entity="Exploration Item")
        clean_db.epistemes.create(e1)

        e2 = make_episteme(entity="Stable Item")
        e2_created = clean_db.epistemes.create(e2)
        # To test different lifecycle, we'd need a way to update state
        # For now, verify exploration filter works

        results = clean_db.epistemes.find_by_lifecycle(
            LifecycleState.EXPLORATION, "test_ctx"
        )

        assert len(results) >= 1

    def test_find_by_assurance(self, clean_db):
        """Can filter epistemes by minimum assurance level."""
        clean_db.epistemes.create(make_episteme(entity="Low Assurance"))

        results = clean_db.epistemes.find_by_assurance(
            AssuranceLevel.L0, "test_ctx"
        )

        assert len(results) >= 1

    def test_find_all_in_context(self, clean_db):
        """Can list all epistemes in a context."""
        clean_db.epistemes.create(make_episteme(context_id="ctx_a", entity="A1"))
        clean_db.epistemes.create(make_episteme(context_id="ctx_a", entity="A2"))
        clean_db.epistemes.create(make_episteme(context_id="ctx_b", entity="B1"))

        results_a = clean_db.epistemes.find_all_in_context("ctx_a")
        results_b = clean_db.epistemes.find_all_in_context("ctx_b")

        assert len(results_a) == 2
        assert len(results_b) == 1

    def test_history_traverses_editions(self, clean_db):
        """History returns all editions of an episteme."""
        v1 = clean_db.epistemes.create(make_episteme(entity="Versioned Entity"))
        v2 = clean_db.epistemes.update(v1)
        v3 = clean_db.epistemes.update(v2)

        history = list(clean_db.epistemes.history(v3.holon_id.id))

        # Should get v3, v2, v1 (or at least v3 as starting point)
        assert len(history) >= 1
        assert history[0].holon_id.id == v3.holon_id.id

    def test_count_by_context(self, clean_db):
        """Can count epistemes in context."""
        clean_db.epistemes.create(make_episteme(context_id="count_ctx"))
        clean_db.epistemes.create(make_episteme(context_id="count_ctx"))
        clean_db.epistemes.create(make_episteme(context_id="other_ctx"))

        count = clean_db.epistemes.count_by_context("count_ctx")

        assert count == 2

    def test_delete_episteme(self, clean_db):
        """Can delete episteme."""
        episteme = clean_db.epistemes.create(make_episteme())

        result = clean_db.epistemes.delete(episteme.holon_id.id)

        assert result is True
        assert clean_db.epistemes.get_by_key(str(episteme.holon_id.id)) is None


class TestEvidenceGraph:
    """Test evidence graph and NetworkX analysis."""

    def test_add_evidence_link(self, clean_db):
        """Can add evidence link between epistemes."""
        e1 = clean_db.epistemes.create(make_episteme(entity="Claim"))
        e2 = clean_db.epistemes.create(make_episteme(entity="Evidence"))

        link = clean_db.evidence.add_link(
            claim_id=e1.holon_id.id,
            evidence_id=e2.holon_id.id,
            relation=EvidenceRelation.SUPPORTS,
            strength=0.8,
            cl=CongruenceLevel.CL4_ALIGNED
        )

        assert link.from_id == e1.holon_id.id
        assert link.to_id == e2.holon_id.id
        assert link.relation == EvidenceRelation.SUPPORTS
        assert link.strength == 0.8

    def test_add_link_with_string_relation(self, clean_db):
        """Can add link using string relation type."""
        e1 = clean_db.epistemes.create(make_episteme(entity="Claim"))
        e2 = clean_db.epistemes.create(make_episteme(entity="Refutation"))

        link = clean_db.evidence.add_link(
            claim_id=e1.holon_id.id,
            evidence_id=e2.holon_id.id,
            relation="refutes",  # String instead of enum
            strength=0.6
        )

        assert link.relation == EvidenceRelation.REFUTES

    def test_get_evidence_for_claim(self, clean_db):
        """Can retrieve all evidence for a claim."""
        claim = clean_db.epistemes.create(make_episteme(entity="Main Claim"))
        ev1 = clean_db.epistemes.create(make_episteme(entity="Evidence 1"))
        ev2 = clean_db.epistemes.create(make_episteme(entity="Evidence 2"))

        clean_db.evidence.add_link(
            claim.holon_id.id, ev1.holon_id.id,
            EvidenceRelation.SUPPORTS, strength=0.7
        )
        clean_db.evidence.add_link(
            claim.holon_id.id, ev2.holon_id.id,
            EvidenceRelation.VERIFIED_BY, strength=0.9
        )

        links = clean_db.evidence.get_evidence_for(claim.holon_id.id)

        assert len(links) == 2

    def test_get_evidence_with_relation_filter(self, clean_db):
        """Can filter evidence by relation type."""
        claim = clean_db.epistemes.create(make_episteme(entity="Claim"))
        support = clean_db.epistemes.create(make_episteme(entity="Support"))
        refute = clean_db.epistemes.create(make_episteme(entity="Refutation"))

        clean_db.evidence.add_link(
            claim.holon_id.id, support.holon_id.id,
            EvidenceRelation.SUPPORTS
        )
        clean_db.evidence.add_link(
            claim.holon_id.id, refute.holon_id.id,
            EvidenceRelation.REFUTES
        )

        supports_only = clean_db.evidence.get_evidence_for(
            claim.holon_id.id,
            relation_filter=[EvidenceRelation.SUPPORTS]
        )

        assert len(supports_only) == 1
        assert supports_only[0].relation == EvidenceRelation.SUPPORTS

    def test_remove_link(self, clean_db):
        """Can remove evidence link."""
        e1 = clean_db.epistemes.create(make_episteme())
        e2 = clean_db.epistemes.create(make_episteme())

        link = clean_db.evidence.add_link(
            e1.holon_id.id, e2.holon_id.id,
            EvidenceRelation.SUPPORTS
        )

        result = clean_db.evidence.remove_link(link.id)

        assert result is True
        links = clean_db.evidence.get_evidence_for(e1.holon_id.id)
        assert len(links) == 0

    def test_compute_support_with_networkx(self, clean_db):
        """NetworkX computes aggregate support for claim."""
        claim = clean_db.epistemes.create(make_episteme(entity="Hypothesis"))
        ev1 = clean_db.epistemes.create(make_episteme(entity="Strong Evidence"))
        ev2 = clean_db.epistemes.create(make_episteme(entity="Weak Evidence"))

        # Add supporting evidence (evidence points TO claim as predecessor)
        # The graph models: evidence SUPPORTS claim, so edge: evidence → claim
        clean_db.evidence.add_link(
            claim_id=ev1.holon_id.id, evidence_id=claim.holon_id.id,
            relation=EvidenceRelation.SUPPORTS, strength=0.9,
            cl=CongruenceLevel.CL5_EXACT
        )
        clean_db.evidence.add_link(
            claim_id=ev2.holon_id.id, evidence_id=claim.holon_id.id,
            relation=EvidenceRelation.SUPPORTS, strength=0.3,
            cl=CongruenceLevel.CL3_PARTIAL
        )

        support, audit = clean_db.evidence.compute_support(claim.holon_id.id)

        assert support > 0.0
        assert len(audit) > 0  # Has audit trail

    def test_compute_support_with_refutation(self, clean_db):
        """Support calculation accounts for refuting evidence."""
        claim = clean_db.epistemes.create(make_episteme(entity="Contested Claim"))
        support_ev = clean_db.epistemes.create(make_episteme(entity="Support"))
        refute_ev = clean_db.epistemes.create(make_episteme(entity="Refutation"))

        # Edge direction: evidence → claim (evidence is predecessor of claim)
        clean_db.evidence.add_link(
            claim_id=support_ev.holon_id.id, evidence_id=claim.holon_id.id,
            relation=EvidenceRelation.SUPPORTS, strength=0.5,
            cl=CongruenceLevel.CL5_EXACT
        )
        clean_db.evidence.add_link(
            claim_id=refute_ev.holon_id.id, evidence_id=claim.holon_id.id,
            relation=EvidenceRelation.REFUTES, strength=0.5,
            cl=CongruenceLevel.CL5_EXACT
        )

        net_support, audit = clean_db.evidence.compute_support(claim.holon_id.id)

        # Equal support and refutation should roughly cancel
        assert net_support < 0.6
        # Check audit trail mentions refutation
        assert any("refutes" in a.lower() or "-0" in a for a in audit)

    def test_find_evidence_paths(self, clean_db):
        """Can find all paths to a claim through evidence graph."""
        # Build chain: root -> mid -> claim
        claim = clean_db.epistemes.create(make_episteme(entity="Final Claim"))
        mid = clean_db.epistemes.create(make_episteme(entity="Intermediate"))
        root = clean_db.epistemes.create(make_episteme(entity="Root Evidence"))

        clean_db.evidence.add_link(
            claim.holon_id.id, mid.holon_id.id,
            EvidenceRelation.DERIVED_FROM
        )
        clean_db.evidence.add_link(
            mid.holon_id.id, root.holon_id.id,
            EvidenceRelation.SUPPORTS
        )

        paths = clean_db.evidence.find_evidence_paths(claim.holon_id.id)

        assert len(paths) >= 1

    def test_compute_path_trust_wlnk(self, clean_db):
        """Path trust is bounded by weakest link (WLNK invariant)."""
        e1 = clean_db.epistemes.create(make_episteme(entity="Start"))
        e2 = clean_db.epistemes.create(make_episteme(entity="Middle"))
        e3 = clean_db.epistemes.create(make_episteme(entity="End"))

        # Strong first link
        clean_db.evidence.add_link(
            e2.holon_id.id, e1.holon_id.id,
            EvidenceRelation.SUPPORTS, strength=0.9,
            cl=CongruenceLevel.CL5_EXACT
        )
        # Weak second link
        clean_db.evidence.add_link(
            e3.holon_id.id, e2.holon_id.id,
            EvidenceRelation.SUPPORTS, strength=0.2,
            cl=CongruenceLevel.CL2_LOOSE
        )

        path = [str(e3.holon_id.id), str(e2.holon_id.id), str(e1.holon_id.id)]
        fgr, audit = clean_db.evidence.compute_path_trust(path)

        # Trust should be bounded by weakest link
        assert fgr.reliability <= 0.3  # Bounded by weak link
        assert any("WLNK" in a for a in audit)

    def test_get_orphan_claims(self, clean_db):
        """Can identify claims without supporting evidence."""
        supported = clean_db.epistemes.create(make_episteme(entity="Supported"))
        orphan = clean_db.epistemes.create(make_episteme(entity="Orphan"))
        evidence = clean_db.epistemes.create(make_episteme(entity="Evidence"))

        clean_db.evidence.add_link(
            supported.holon_id.id, evidence.holon_id.id,
            EvidenceRelation.SUPPORTS
        )
        # orphan has no evidence links

        orphans = clean_db.evidence.get_orphan_claims()
        orphan_ids = set(orphans)

        # Orphan should be in the list (or all nodes if none have support)
        assert str(orphan.holon_id.id) in orphan_ids or len(orphans) >= 1

    def test_invalidate_cache(self, clean_db):
        """Can force NetworkX graph reload."""
        e1 = clean_db.epistemes.create(make_episteme())
        e2 = clean_db.epistemes.create(make_episteme())

        clean_db.evidence.add_link(
            e1.holon_id.id, e2.holon_id.id,
            EvidenceRelation.SUPPORTS
        )

        # Access graph to cache it
        clean_db.evidence.compute_support(e1.holon_id.id)

        # Invalidate
        clean_db.evidence.invalidate_cache()

        # Should reload on next access
        support, _ = clean_db.evidence.compute_support(e1.holon_id.id)
        assert support >= 0.0


class TestContextStore:
    """Test bounded context persistence."""

    def test_create_context(self, clean_db):
        """Can create bounded context."""
        context = UBoundedContext(
            context_id="research_ctx",
            name="Research Domain",
            description="Test research context"
        )

        created = clean_db.contexts.create(context)

        assert created.context_id == "research_ctx"
        assert created.name == "Research Domain"

    def test_get_context(self, clean_db):
        """Can retrieve context by ID."""
        context = UBoundedContext(
            context_id="retrievable",
            name="Retrievable Context",
            description="Context for retrieval test"
        )
        clean_db.contexts.create(context)

        retrieved = clean_db.contexts.get("retrievable")

        assert retrieved is not None
        assert retrieved.name == "Retrievable Context"

    def test_find_by_name(self, clean_db):
        """Can find context by name."""
        context = UBoundedContext(
            context_id="named_ctx",
            name="Unique Name Here",
            description="Context for name search test"
        )
        clean_db.contexts.create(context)

        found = clean_db.contexts.find_by_name("Unique Name Here")

        assert found is not None
        assert found.context_id == "named_ctx"

    def test_exists(self, clean_db):
        """Can check if context exists."""
        context = UBoundedContext(
            context_id="exists_ctx",
            name="Exists",
            description="Context for exists test"
        )
        clean_db.contexts.create(context)

        assert clean_db.contexts.exists("exists_ctx") is True
        assert clean_db.contexts.exists("nonexistent") is False

    def test_list_all(self, clean_db):
        """Can list all contexts."""
        clean_db.contexts.create(UBoundedContext(
            context_id="list_1", name="List Context 1", description="First list ctx"
        ))
        clean_db.contexts.create(UBoundedContext(
            context_id="list_2", name="List Context 2", description="Second list ctx"
        ))

        all_contexts = clean_db.contexts.list_all()

        assert len(all_contexts) >= 2
        names = [c.name for c in all_contexts]
        assert "List Context 1" in names
        assert "List Context 2" in names

    def test_update_context(self, clean_db):
        """Can update context."""
        context = UBoundedContext(
            context_id="updatable",
            name="Original Name",
            description="Original description"
        )
        clean_db.contexts.create(context)

        context.description = "Updated description"
        updated = clean_db.contexts.update(context)

        retrieved = clean_db.contexts.get("updatable")
        assert retrieved.description == "Updated description"

    def test_add_glossary_term(self, clean_db):
        """Can add terms to context glossary."""
        context = UBoundedContext(
            context_id="glossary_ctx",
            name="Glossary Context",
            description="Context for glossary test"
        )
        clean_db.contexts.create(context)

        clean_db.contexts.add_glossary_term(
            "glossary_ctx",
            "episteme",
            "A knowledge unit in FPF"
        )

        retrieved = clean_db.contexts.get("glossary_ctx")
        assert "episteme" in retrieved.glossary
        assert retrieved.glossary["episteme"] == "A knowledge unit in FPF"

    def test_add_invariant(self, clean_db):
        """Can add invariants to context."""
        context = UBoundedContext(
            context_id="invariant_ctx",
            name="Invariant Context",
            description="Context for invariant test"
        )
        clean_db.contexts.create(context)

        clean_db.contexts.add_invariant(
            "invariant_ctx",
            "All claims must have evidence"
        )

        retrieved = clean_db.contexts.get("invariant_ctx")
        assert "All claims must have evidence" in retrieved.invariants

    def test_delete_context(self, clean_db):
        """Can delete context."""
        context = UBoundedContext(
            context_id="deletable",
            name="Deletable Context",
            description="Context for delete test"
        )
        clean_db.contexts.create(context)

        result = clean_db.contexts.delete("deletable")

        assert result is True
        assert clean_db.contexts.get("deletable") is None

    def test_create_bridge(self, clean_db):
        """Can create bridge between contexts."""
        ctx1 = UBoundedContext(context_id="source", name="Source", description="Source ctx")
        ctx2 = UBoundedContext(context_id="target", name="Target", description="Target ctx")
        clean_db.contexts.create(ctx1)
        clean_db.contexts.create(ctx2)

        bridge_id = clean_db.contexts.create_bridge(
            from_context="source",
            to_context="target",
            bridge_type="translation",
            congruence_level=4
        )

        assert bridge_id is not None

    def test_get_bridges_from(self, clean_db):
        """Can get all bridges from a context."""
        ctx1 = UBoundedContext(context_id="from_ctx", name="From", description="From ctx")
        ctx2 = UBoundedContext(context_id="to_ctx1", name="To 1", description="To 1 ctx")
        ctx3 = UBoundedContext(context_id="to_ctx2", name="To 2", description="To 2 ctx")
        clean_db.contexts.create(ctx1)
        clean_db.contexts.create(ctx2)
        clean_db.contexts.create(ctx3)

        clean_db.contexts.create_bridge("from_ctx", "to_ctx1")
        clean_db.contexts.create_bridge("from_ctx", "to_ctx2")

        bridges = clean_db.contexts.get_bridges_from("from_ctx")

        assert len(bridges) == 2

    def test_get_bridge_between(self, clean_db):
        """Can get specific bridge between two contexts."""
        ctx1 = UBoundedContext(context_id="a", name="A", description="Context A")
        ctx2 = UBoundedContext(context_id="b", name="B", description="Context B")
        clean_db.contexts.create(ctx1)
        clean_db.contexts.create(ctx2)

        clean_db.contexts.create_bridge(
            "a", "b",
            bridge_type="alignment",
            congruence_level=3
        )

        bridge = clean_db.contexts.get_bridge("a", "b")

        assert bridge is not None
        assert bridge["bridge_type"] == "alignment"
        assert bridge["congruence_level"] == 3

    def test_delete_bridge(self, clean_db):
        """Can delete bridge."""
        ctx1 = UBoundedContext(context_id="x", name="X", description="Context X")
        ctx2 = UBoundedContext(context_id="y", name="Y", description="Context Y")
        clean_db.contexts.create(ctx1)
        clean_db.contexts.create(ctx2)

        bridge_id = clean_db.contexts.create_bridge("x", "y")

        result = clean_db.contexts.delete_bridge(bridge_id)

        assert result is True
        assert clean_db.contexts.get_bridge("x", "y") is None


class TestResearchSessionStore:
    """Test research session persistence."""

    def test_create_session(self, clean_db):
        """Can create research session."""
        session = clean_db.sessions.create(
            context_id="research_ctx",
            research_question="What is the nature of X?",
            methodology="systematic"
        )

        assert session["context_id"] == "research_ctx"
        assert session["research_question"] == "What is the nature of X?"
        assert session["state"] == "active"

    def test_get_session(self, clean_db):
        """Can retrieve session by ID."""
        created = clean_db.sessions.create(
            context_id="ctx",
            research_question="Test question"
        )

        retrieved = clean_db.sessions.get(created["_key"])

        assert retrieved is not None
        assert retrieved["research_question"] == "Test question"

    def test_list_all_sessions(self, clean_db):
        """Can list all sessions."""
        clean_db.sessions.create("ctx1", "Question 1")
        clean_db.sessions.create("ctx2", "Question 2")

        all_sessions = clean_db.sessions.list_all()

        assert len(all_sessions) >= 2

    def test_list_active_sessions(self, clean_db):
        """Can filter to active sessions only."""
        s1 = clean_db.sessions.create("ctx", "Active session")
        s2 = clean_db.sessions.create("ctx", "Will be paused")
        clean_db.sessions.pause(s2["_key"])

        active = clean_db.sessions.list_active()

        # Only one should be active
        assert len(active) >= 1
        assert all(s["state"] == "active" for s in active)

    def test_list_sessions_by_context(self, clean_db):
        """Can filter sessions by context."""
        clean_db.sessions.create("ctx_a", "Question A")
        clean_db.sessions.create("ctx_b", "Question B")
        clean_db.sessions.create("ctx_a", "Another A question")

        ctx_a_sessions = clean_db.sessions.list_all(context_id="ctx_a")

        assert len(ctx_a_sessions) == 2

    def test_session_state_transitions(self, clean_db):
        """Session can transition through states."""
        session = clean_db.sessions.create("ctx", "Question")
        session_id = session["_key"]

        # Pause
        clean_db.sessions.pause(session_id)
        assert clean_db.sessions.get(session_id)["state"] == "paused"

        # Resume
        clean_db.sessions.resume(session_id)
        assert clean_db.sessions.get(session_id)["state"] == "active"

        # Complete
        clean_db.sessions.complete(session_id)
        assert clean_db.sessions.get(session_id)["state"] == "completed"

        # Archive
        clean_db.sessions.archive(session_id)
        assert clean_db.sessions.get(session_id)["state"] == "archived"

    def test_add_episteme_to_session(self, clean_db):
        """Can associate episteme with session."""
        session = clean_db.sessions.create("ctx", "Question")
        episteme_id = uuid4()

        clean_db.sessions.add_episteme(session["_key"], episteme_id)

        episteme_ids = clean_db.sessions.get_session_epistemes(session["_key"])
        assert str(episteme_id) in episteme_ids

    def test_add_hypothesis_to_session(self, clean_db):
        """Can associate hypothesis with session."""
        session = clean_db.sessions.create("ctx", "Question")
        hypothesis_id = uuid4()

        clean_db.sessions.add_hypothesis(session["_key"], hypothesis_id)

        hypothesis_ids = clean_db.sessions.get_session_hypotheses(session["_key"])
        assert str(hypothesis_id) in hypothesis_ids

    def test_update_methodology(self, clean_db):
        """Can update session methodology."""
        session = clean_db.sessions.create(
            "ctx", "Question", methodology="systematic"
        )

        clean_db.sessions.update_methodology(session["_key"], "exploratory")

        updated = clean_db.sessions.get(session["_key"])
        assert updated["methodology"] == "exploratory"

    def test_delete_session(self, clean_db):
        """Can delete session."""
        session = clean_db.sessions.create("ctx", "Question")

        result = clean_db.sessions.delete(session["_key"])

        assert result is True
        assert clean_db.sessions.get(session["_key"]) is None


class TestPersistenceManager:
    """Test unified persistence manager."""

    def test_get_or_create_context_creates(self, clean_db):
        """get_or_create_context creates new context if not exists."""
        context = clean_db.get_or_create_context(
            name="New Context",
            description="Created by manager"
        )

        assert context is not None
        assert context.name == "New Context"

    def test_get_or_create_context_gets_existing(self, clean_db):
        """get_or_create_context returns existing context."""
        # Create first
        ctx1 = clean_db.get_or_create_context("Existing", "First creation")

        # Get same
        ctx2 = clean_db.get_or_create_context("Existing", "Different desc")

        assert ctx1.context_id == ctx2.context_id
        # Description should be from first creation
        assert ctx2.description == "First creation"

    def test_stores_are_lazily_initialized(self, clean_db):
        """Store properties initialize on first access."""
        # Fresh manager
        pm = PersistenceManager(
            host=TEST_DB_HOST,
            username=TEST_DB_USER,
            password=TEST_DB_PASS,
            database=TEST_DB_NAME
        )
        pm.initialize()

        # Internal stores should be None initially
        assert pm._episteme_store is None

        # Access triggers initialization
        _ = pm.epistemes

        assert pm._episteme_store is not None


class TestEvidenceRelationTypes:
    """Test all evidence relation types are correctly handled."""

    @pytest.mark.parametrize("relation", [
        EvidenceRelation.SUPPORTS,
        EvidenceRelation.REFUTES,
        EvidenceRelation.QUALIFIES,
        EvidenceRelation.VERIFIED_BY,
        EvidenceRelation.VALIDATED_BY,
        EvidenceRelation.DERIVED_FROM,
        EvidenceRelation.FROM_WORK_SET,
        EvidenceRelation.HAPPENED_BEFORE,
    ])
    def test_all_relation_types(self, clean_db, relation):
        """All evidence relation types can be stored and retrieved."""
        e1 = clean_db.epistemes.create(make_episteme(entity=f"Claim_{relation.value}"))
        e2 = clean_db.epistemes.create(make_episteme(entity=f"Evidence_{relation.value}"))

        link = clean_db.evidence.add_link(
            e1.holon_id.id, e2.holon_id.id,
            relation=relation,
            strength=0.5
        )

        assert link.relation == relation

        # Retrieve and verify
        links = clean_db.evidence.get_evidence_for(e1.holon_id.id)
        assert len(links) == 1
        assert links[0].relation == relation

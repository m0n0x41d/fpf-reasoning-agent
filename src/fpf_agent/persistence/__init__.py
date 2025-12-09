"""
Unified persistence layer for FPF Agent.

Provides single entry point for all persistence operations:
- ArangoDB for document and graph storage
- NetworkX for in-memory graph algorithms

Components:
- PersistenceManager: Unified access to all stores
- EpistemeStore: Versioned episteme CRUD
- EvidenceGraph: Evidence relationships with NetworkX analysis
- ContextStore: Bounded contexts and bridges
- ResearchSessionStore: Cross-session research tracking
"""
from __future__ import annotations

from typing import Optional
from uuid import uuid4

from .arango_schema import ArangoSchema
from .context_store import ContextStore
from .episteme_store import EpistemeStore
from .evidence_graph import EvidenceGraph, EvidenceLink, EvidenceRelation
from .session_store import ResearchSessionStore


class PersistenceManager:
    """
    Unified persistence manager.

    Provides:
    - Single ArangoDB connection
    - All stores (epistemes, evidence, contexts, sessions)
    - Cross-session state management

    Usage:
        pm = create_persistence()
        context = pm.get_or_create_context("My Research")
        episteme = pm.epistemes.create(...)
        pm.evidence.add_link(...)
    """

    def __init__(
        self,
        host: str = "http://localhost:8529",
        username: str = "root",
        password: str = "",
        database: str = "fpf_agent"
    ):
        self.schema = ArangoSchema(host, username, password, database)

        self._episteme_store: Optional[EpistemeStore] = None
        self._evidence_graph: Optional[EvidenceGraph] = None
        self._context_store: Optional[ContextStore] = None
        self._session_store: Optional[ResearchSessionStore] = None

    def initialize(self) -> None:
        """Initialize database schema."""
        self.schema.initialize()

    @property
    def db(self):
        """Get database connection."""
        return self.schema.db

    @property
    def epistemes(self) -> EpistemeStore:
        """Get episteme store."""
        if self._episteme_store is None:
            self._episteme_store = EpistemeStore(self.db)
        return self._episteme_store

    @property
    def evidence(self) -> EvidenceGraph:
        """Get evidence graph."""
        if self._evidence_graph is None:
            self._evidence_graph = EvidenceGraph(self.db)
        return self._evidence_graph

    @property
    def contexts(self) -> ContextStore:
        """Get context store."""
        if self._context_store is None:
            self._context_store = ContextStore(self.db)
        return self._context_store

    @property
    def sessions(self) -> ResearchSessionStore:
        """Get session store."""
        if self._session_store is None:
            self._session_store = ResearchSessionStore(self.db)
        return self._session_store

    def get_or_create_context(
        self,
        name: str,
        description: str = ""
    ) -> "UBoundedContext":
        """Get existing context by name or create new one."""
        from ..kernel.bounded_context import UBoundedContext

        existing = self.contexts.find_by_name(name)
        if existing:
            return existing

        context = UBoundedContext(
            context_id=f"ctx_{uuid4().hex[:8]}",
            name=name,
            description=description
        )
        return self.contexts.create(context)

    def health_check(self) -> dict:
        """Check persistence health."""
        return self.schema.health_check()

    def reset_database(self) -> None:
        """
        Drop and recreate all collections.

        WARNING: Destroys all data. Use only for testing.
        """
        self.schema.drop_all()
        self.schema.initialize()

        self._episteme_store = None
        self._evidence_graph = None
        self._context_store = None
        self._session_store = None

    def close(self) -> None:
        """Close connections (ArangoDB manages pooling)."""
        pass


def create_persistence(
    host: str = "http://localhost:8529",
    username: str = "root",
    password: str = "",
    database: str = "fpf_agent"
) -> PersistenceManager:
    """Create and initialize persistence manager."""
    pm = PersistenceManager(host, username, password, database)
    pm.initialize()
    return pm


__all__ = [
    "ArangoSchema",
    "ContextStore",
    "create_persistence",
    "EpistemeStore",
    "EvidenceGraph",
    "EvidenceLink",
    "EvidenceRelation",
    "PersistenceManager",
    "ResearchSessionStore",
]

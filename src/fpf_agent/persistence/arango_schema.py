"""
ArangoDB schema for FPF persistent storage.

Collections:
- Documents: contexts, epistemes, sessions, hypotheses, adi_cycles
- Graphs: fpf_evidence_graph (edges: evidence_links, context_bridges, edition_chain)

FPF Compliance:
- A.1.1: Bounded contexts as documents
- A.4: Edition chains via edges
- A.10: Evidence graph for provenance
- B.3: F-G-R stored per episteme
"""
from __future__ import annotations

import logging
from typing import Optional

from arango import ArangoClient
from arango.database import StandardDatabase

logger = logging.getLogger(__name__)


COLLECTIONS = {
    "contexts": {
        "type": "document",
        "indexes": [
            {"type": "persistent", "fields": ["name"], "unique": True},
        ]
    },
    "epistemes": {
        "type": "document",
        "indexes": [
            {"type": "persistent", "fields": ["context_id"]},
            {"type": "persistent", "fields": ["lifecycle_state"]},
            {"type": "persistent", "fields": ["edition.number"]},
            {"type": "persistent", "fields": ["edition.supersedes"]},
        ]
    },
    "sessions": {
        "type": "document",
        "indexes": [
            {"type": "persistent", "fields": ["context_id"]},
            {"type": "persistent", "fields": ["state"]},
        ]
    },
    "hypotheses": {
        "type": "document",
        "indexes": [
            {"type": "persistent", "fields": ["episteme_id"]},
            {"type": "persistent", "fields": ["status"]},
        ]
    },
    "adi_cycles": {
        "type": "document",
        "indexes": [
            {"type": "persistent", "fields": ["context_id"]},
            {"type": "persistent", "fields": ["current_phase"]},
        ]
    }
}

EDGE_COLLECTIONS = {
    "evidence_links": {
        "type": "edge",
        "indexes": [
            {"type": "persistent", "fields": ["relation"]},
            {"type": "persistent", "fields": ["congruence_level"]},
        ]
    },
    "context_bridges": {
        "type": "edge",
        "indexes": [
            {"type": "persistent", "fields": ["congruence_level"]},
        ]
    },
    "edition_chain": {
        "type": "edge",
        "indexes": [
            {"type": "persistent", "fields": ["edition_number"]},
        ]
    }
}

GRAPH_DEFINITION = {
    "name": "fpf_evidence_graph",
    "edge_definitions": [
        {
            "edge_collection": "evidence_links",
            "from_vertex_collections": ["epistemes"],
            "to_vertex_collections": ["epistemes"]
        },
        {
            "edge_collection": "context_bridges",
            "from_vertex_collections": ["contexts"],
            "to_vertex_collections": ["contexts"]
        },
        {
            "edge_collection": "edition_chain",
            "from_vertex_collections": ["epistemes"],
            "to_vertex_collections": ["epistemes"]
        }
    ]
}


class ArangoSchema:
    """
    Manages ArangoDB schema initialization and migrations.
    """

    def __init__(
        self,
        host: str = "http://localhost:8529",
        username: str = "root",
        password: str = "",
        database: str = "fpf_agent"
    ):
        self.client = ArangoClient(hosts=host)
        self.sys_db = self.client.db("_system", username=username, password=password)
        self.db_name = database
        self.username = username
        self.password = password
        self._db: Optional[StandardDatabase] = None

    @property
    def db(self) -> StandardDatabase:
        """Get or create the database connection."""
        if self._db is None:
            if not self.sys_db.has_database(self.db_name):
                self.sys_db.create_database(self.db_name)
                logger.info(f"Created database: {self.db_name}")

            self._db = self.client.db(
                self.db_name,
                username=self.username,
                password=self.password
            )
        return self._db

    def initialize(self) -> None:
        """Initialize all collections, indexes, and graph."""
        for name, config in COLLECTIONS.items():
            self._ensure_collection(name, edge=False, indexes=config.get("indexes", []))

        for name, config in EDGE_COLLECTIONS.items():
            self._ensure_collection(name, edge=True, indexes=config.get("indexes", []))

        self._ensure_graph()

        logger.info("ArangoDB schema initialized")

    def _ensure_collection(
        self,
        name: str,
        edge: bool,
        indexes: list
    ) -> None:
        """Ensure collection exists with indexes."""
        if not self.db.has_collection(name):
            collection = self.db.create_collection(name, edge=edge)
            logger.info(f"Created {'edge ' if edge else ''}collection: {name}")
        else:
            collection = self.db.collection(name)

        existing_indexes = {
            tuple(idx.get("fields", [])): idx
            for idx in collection.indexes()
            if idx.get("type") == "persistent"
        }

        for idx_config in indexes:
            fields = tuple(idx_config["fields"])
            if fields not in existing_indexes:
                collection.add_persistent_index(
                    fields=list(fields),
                    unique=idx_config.get("unique", False)
                )
                logger.info(f"Created index on {name}: {fields}")

    def _ensure_graph(self) -> None:
        """Ensure graph exists."""
        graph_name = GRAPH_DEFINITION["name"]
        if not self.db.has_graph(graph_name):
            self.db.create_graph(
                graph_name,
                edge_definitions=GRAPH_DEFINITION["edge_definitions"]
            )
            logger.info(f"Created graph: {graph_name}")

    def drop_all(self) -> None:
        """Drop all collections and graph (for testing/reset)."""
        graph_name = GRAPH_DEFINITION["name"]
        if self.db.has_graph(graph_name):
            self.db.delete_graph(graph_name, drop_collections=False)
            logger.info(f"Deleted graph: {graph_name}")

        all_collections = list(COLLECTIONS.keys()) + list(EDGE_COLLECTIONS.keys())
        for name in all_collections:
            if self.db.has_collection(name):
                self.db.delete_collection(name)
                logger.info(f"Deleted collection: {name}")

    def health_check(self) -> dict:
        """Check database health."""
        try:
            version = self.db.version()
            counts = {}
            for name in COLLECTIONS.keys():
                if self.db.has_collection(name):
                    counts[name] = self.db.collection(name).count()

            return {
                "status": "healthy",
                "version": version,
                "database": self.db_name,
                "collection_counts": counts
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

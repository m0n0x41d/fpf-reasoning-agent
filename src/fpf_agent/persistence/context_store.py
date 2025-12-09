"""
Bounded Context persistence.

FPF A.1.1: All meaning is local to a context.
FPF F.9: Cross-context alignment requires explicit bridges.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from arango.database import StandardDatabase

from ..kernel.bounded_context import UBoundedContext


class ContextStore:
    """
    Persistent storage for bounded contexts.

    Enables cross-session context retrieval and bridge management.
    """

    def __init__(self, db: StandardDatabase):
        self.db = db
        self._contexts = db.collection("contexts")
        self._bridges = db.collection("context_bridges")

    def create(self, context: UBoundedContext) -> UBoundedContext:
        """Create new bounded context."""
        doc = {
            "_key": context.context_id,
            "name": context.name,
            "description": context.description,
            "glossary": context.glossary,
            "invariants": context.invariants,
            "parent_context_id": context.parent_context_id,
            "version": context.version,
            "created_at": context.created_at.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        self._contexts.insert(doc)
        return context

    def get(self, context_id: str) -> Optional[UBoundedContext]:
        """Get context by ID."""
        try:
            doc = self._contexts.get(context_id)
            if doc:
                return self._doc_to_context(doc)
        except Exception:
            pass
        return None

    def exists(self, context_id: str) -> bool:
        """Check if context exists."""
        return self._contexts.has(context_id)

    def list_all(self) -> List[UBoundedContext]:
        """List all contexts."""
        query = """
        FOR c IN contexts
            SORT c.created_at DESC
            RETURN c
        """
        cursor = self.db.aql.execute(query)
        return [self._doc_to_context(doc) for doc in cursor]

    def find_by_name(self, name: str) -> Optional[UBoundedContext]:
        """Find context by name."""
        query = """
        FOR c IN contexts
            FILTER c.name == @name
            RETURN c
        """
        cursor = self.db.aql.execute(query, bind_vars={"name": name})
        docs = list(cursor)
        if docs:
            return self._doc_to_context(docs[0])
        return None

    def update(self, context: UBoundedContext) -> UBoundedContext:
        """Update existing context."""
        doc = {
            "name": context.name,
            "description": context.description,
            "glossary": context.glossary,
            "invariants": context.invariants,
            "parent_context_id": context.parent_context_id,
            "version": context.version,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        self._contexts.update({"_key": context.context_id, **doc})
        return context

    def add_glossary_term(
        self,
        context_id: str,
        term: str,
        definition: str
    ) -> bool:
        """Add term to context glossary."""
        query = """
        FOR c IN contexts
            FILTER c._key == @key
            UPDATE c WITH {
                glossary: MERGE(c.glossary, {@term: @def}),
                updated_at: @now
            } IN contexts
        """
        self.db.aql.execute(query, bind_vars={
            "key": context_id,
            "term": term,
            "def": definition,
            "now": datetime.now(timezone.utc).isoformat()
        })
        return True

    def add_invariant(
        self,
        context_id: str,
        invariant: str
    ) -> bool:
        """Add invariant to context."""
        query = """
        FOR c IN contexts
            FILTER c._key == @key
            UPDATE c WITH {
                invariants: APPEND(c.invariants, @inv, true),
                updated_at: @now
            } IN contexts
        """
        self.db.aql.execute(query, bind_vars={
            "key": context_id,
            "inv": invariant,
            "now": datetime.now(timezone.utc).isoformat()
        })
        return True

    def delete(self, context_id: str) -> bool:
        """Delete context."""
        try:
            del_bridges_query = """
            FOR b IN context_bridges
                FILTER b._from == @ctx OR b._to == @ctx
                REMOVE b IN context_bridges
            """
            self.db.aql.execute(
                del_bridges_query,
                bind_vars={"ctx": f"contexts/{context_id}"}
            )

            self._contexts.delete(context_id)
            return True
        except Exception:
            return False

    def create_bridge(
        self,
        from_context: str,
        to_context: str,
        bridge_type: str = "translation",
        congruence_level: int = 3,
        loss_policy: str = "penalize",
        mapping: Optional[dict] = None
    ) -> str:
        """
        Create bridge between contexts.

        FPF F.9: Cross-context alignment requires explicit bridge.
        """
        bridge_id = str(uuid4())

        doc = {
            "_key": bridge_id,
            "_from": f"contexts/{from_context}",
            "_to": f"contexts/{to_context}",
            "bridge_type": bridge_type,
            "congruence_level": congruence_level,
            "loss_policy": loss_policy,
            "mapping": mapping or {},
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        self._bridges.insert(doc)
        return bridge_id

    def get_bridges_from(self, context_id: str) -> List[dict]:
        """Get all bridges originating from context."""
        query = """
        FOR b IN context_bridges
            FILTER b._from == @ctx
            RETURN b
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={"ctx": f"contexts/{context_id}"}
        )
        return list(cursor)

    def get_bridges_to(self, context_id: str) -> List[dict]:
        """Get all bridges targeting context."""
        query = """
        FOR b IN context_bridges
            FILTER b._to == @ctx
            RETURN b
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={"ctx": f"contexts/{context_id}"}
        )
        return list(cursor)

    def get_bridge(self, from_context: str, to_context: str) -> Optional[dict]:
        """Get bridge between two specific contexts."""
        query = """
        FOR b IN context_bridges
            FILTER b._from == @from AND b._to == @to
            RETURN b
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "from": f"contexts/{from_context}",
                "to": f"contexts/{to_context}"
            }
        )
        docs = list(cursor)
        if docs:
            return docs[0]
        return None

    def delete_bridge(self, bridge_id: str) -> bool:
        """Delete a bridge."""
        try:
            self._bridges.delete(bridge_id)
            return True
        except Exception:
            return False

    def _doc_to_context(self, doc: dict) -> UBoundedContext:
        """Convert document to UBoundedContext."""
        created_at_raw = doc.get("created_at")
        if created_at_raw:
            created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        else:
            created_at = datetime.now(timezone.utc)

        return UBoundedContext(
            context_id=doc["_key"],
            name=doc["name"],
            description=doc.get("description", ""),
            glossary=doc.get("glossary", {}),
            invariants=doc.get("invariants", []),
            parent_context_id=doc.get("parent_context_id"),
            version=doc.get("version", "1.0.0"),
            created_at=created_at
        )

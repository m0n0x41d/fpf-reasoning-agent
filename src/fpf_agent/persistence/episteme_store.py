"""
Versioned episteme storage with ArangoDB backend.

FPF Compliance:
- A.4: Temporal Duality — editions are immutable
- B.3: F-G-R stored and queryable
- A.10: Evidence links via graph edges
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator, List, Optional
from uuid import UUID

from arango.database import StandardDatabase

from ..kernel.holons import StrictDistinctionSlots, UEpisteme
from ..kernel.types import AssuranceLevel, Edition, HolonId, LifecycleState, TemporalStance


class EpistemeStore:
    """
    Persistent episteme storage with versioning.

    Uses ArangoDB for storage, provides clean Pythonic interface.
    Editions are immutable — updates create new editions linked via edition_chain.
    """

    def __init__(self, db: StandardDatabase):
        self.db = db
        self._epistemes = db.collection("epistemes")
        self._edition_chain = db.collection("edition_chain")

    def create(self, episteme: UEpisteme) -> UEpisteme:
        """
        Create new episteme (edition=1).

        Returns the created episteme with assigned ID.
        """
        doc = self._episteme_to_doc(episteme)
        result = self._epistemes.insert(doc, return_new=True)
        return self._doc_to_episteme(result["new"])

    def update(self, episteme: UEpisteme) -> UEpisteme:
        """
        Create new edition of episteme.

        FPF A.4: Old editions are immutable — we create new edition.
        """
        new_holon_id = episteme.holon_id.next_edition()

        new_episteme = episteme.model_copy(update={
            "holon_id": new_holon_id,
            "updated_at": datetime.now(timezone.utc)
        })

        doc = self._episteme_to_doc(new_episteme)
        result = self._epistemes.insert(doc, return_new=True)

        self._edition_chain.insert({
            "_from": f"epistemes/{new_holon_id.id}",
            "_to": f"epistemes/{episteme.holon_id.id}",
            "edition_number": new_holon_id.edition.number
        })

        return self._doc_to_episteme(result["new"])

    def get(
        self,
        holon_id: UUID | str,
        edition: Optional[int] = None
    ) -> Optional[UEpisteme]:
        """
        Get episteme by ID.

        edition=None returns latest edition.
        """
        str_id = str(holon_id)

        if edition is None:
            query = """
            FOR e IN epistemes
                FILTER e._key == @id OR e.edition.supersedes == @id
                SORT e.edition.number DESC
                LIMIT 1
                RETURN e
            """
            cursor = self.db.aql.execute(query, bind_vars={"id": str_id})
            docs = list(cursor)
            if not docs:
                return None
            return self._doc_to_episteme(docs[0])
        else:
            query = """
            FOR e IN epistemes
                FILTER e._key == @id AND e.edition.number == @edition
                RETURN e
            """
            cursor = self.db.aql.execute(
                query,
                bind_vars={"id": str_id, "edition": edition}
            )
            docs = list(cursor)
            if not docs:
                return None
            return self._doc_to_episteme(docs[0])

    def get_by_key(self, key: str) -> Optional[UEpisteme]:
        """Get episteme directly by document key."""
        try:
            doc = self._epistemes.get(key)
            if doc:
                return self._doc_to_episteme(doc)
        except Exception:
            pass
        return None

    def find_by_entity(
        self,
        entity: str,
        context_id: str,
        latest_only: bool = True
    ) -> List[UEpisteme]:
        """Find epistemes by described entity (substring match)."""
        if latest_only:
            query = """
            FOR e IN epistemes
                FILTER CONTAINS(LOWER(e.described_entity), LOWER(@search))
                FILTER e.context_id == @ctx
                FILTER e.edition.supersedes == null
                SORT e.created_at DESC
                RETURN e
            """
        else:
            query = """
            FOR e IN epistemes
                FILTER CONTAINS(LOWER(e.described_entity), LOWER(@search))
                FILTER e.context_id == @ctx
                SORT e.created_at DESC
                RETURN e
            """

        cursor = self.db.aql.execute(
            query,
            bind_vars={"search": entity, "ctx": context_id}
        )
        return [self._doc_to_episteme(doc) for doc in cursor]

    def find_by_lifecycle(
        self,
        state: LifecycleState | str,
        context_id: str
    ) -> List[UEpisteme]:
        """Find epistemes by lifecycle state."""
        state_val = state.value if isinstance(state, LifecycleState) else state

        query = """
        FOR e IN epistemes
            FILTER e.lifecycle_state == @state
            FILTER e.context_id == @ctx
            FILTER e.edition.supersedes == null
            SORT e.updated_at DESC
            RETURN e
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={"state": state_val, "ctx": context_id}
        )
        return [self._doc_to_episteme(doc) for doc in cursor]

    def find_by_assurance(
        self,
        min_level: AssuranceLevel | str,
        context_id: str
    ) -> List[UEpisteme]:
        """Find epistemes with minimum assurance level."""
        if isinstance(min_level, AssuranceLevel):
            min_val = min_level.value
        else:
            levels = {"L0": 0, "L1": 1, "L2": 2}
            min_val = levels.get(min_level, 0)

        query = """
        FOR e IN epistemes
            FILTER e.context_id == @ctx
            FILTER e.edition.supersedes == null
            LET level = (
                e.assurance_level == 2 ? 2 :
                e.assurance_level == 1 ? 1 : 0
            )
            FILTER level >= @min_level
            SORT level DESC, e.updated_at DESC
            RETURN e
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={"ctx": context_id, "min_level": min_val}
        )
        return [self._doc_to_episteme(doc) for doc in cursor]

    def find_all_in_context(
        self,
        context_id: str,
        latest_only: bool = True
    ) -> List[UEpisteme]:
        """Get all epistemes in a context."""
        if latest_only:
            query = """
            FOR e IN epistemes
                FILTER e.context_id == @ctx
                FILTER e.edition.supersedes == null
                SORT e.updated_at DESC
                RETURN e
            """
        else:
            query = """
            FOR e IN epistemes
                FILTER e.context_id == @ctx
                SORT e.edition.number DESC, e.updated_at DESC
                RETURN e
            """

        cursor = self.db.aql.execute(query, bind_vars={"ctx": context_id})
        return [self._doc_to_episteme(doc) for doc in cursor]

    def history(self, holon_id: UUID | str) -> Iterator[UEpisteme]:
        """
        Iterate through all editions of an episteme (newest first).

        Uses graph traversal on edition_chain.
        """
        query = """
        FOR v IN 0..100 OUTBOUND @start edition_chain
            RETURN v
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={"start": f"epistemes/{holon_id}"}
        )
        for doc in cursor:
            if doc:
                yield self._doc_to_episteme(doc)

    def count_by_context(self, context_id: str) -> int:
        """Count epistemes in context (latest editions only)."""
        query = """
        RETURN LENGTH(
            FOR e IN epistemes
                FILTER e.context_id == @ctx
                FILTER e.edition.supersedes == null
                RETURN 1
        )
        """
        cursor = self.db.aql.execute(query, bind_vars={"ctx": context_id})
        return list(cursor)[0]

    def delete(self, holon_id: UUID | str) -> bool:
        """
        Delete episteme and all its editions.

        WARNING: This violates immutability — use only for cleanup/testing.
        """
        str_id = str(holon_id)

        del_edges_query = """
        FOR e IN edition_chain
            FILTER e._from == @doc OR e._to == @doc
            REMOVE e IN edition_chain
        """
        self.db.aql.execute(
            del_edges_query,
            bind_vars={"doc": f"epistemes/{str_id}"}
        )

        try:
            self._epistemes.delete(str_id)
            return True
        except Exception:
            return False

    def _episteme_to_doc(self, episteme: UEpisteme) -> dict:
        """Convert UEpisteme to ArangoDB document."""
        return {
            "_key": str(episteme.holon_id.id),
            "context_id": episteme.holon_id.context_id,
            "edition": {
                "number": episteme.holon_id.edition.number,
                "supersedes": str(episteme.holon_id.edition.supersedes)
                    if episteme.holon_id.edition.supersedes else None,
                "created_at": episteme.holon_id.edition.created_at.isoformat()
            },
            "described_entity": episteme.described_entity,
            "claim_graph": episteme.claim_graph,
            "grounding_holon_id": str(episteme.grounding_holon_id)
                if episteme.grounding_holon_id else None,
            "viewpoint": episteme.viewpoint,
            "slots": {
                "structure": episteme.slots.structure,
                "order": episteme.slots.order,
                "time": episteme.slots.time,
                "work": episteme.slots.work,
                "values": episteme.slots.values
            },
            "lifecycle_state": episteme.lifecycle_state.value
                if isinstance(episteme.lifecycle_state, LifecycleState)
                else episteme.lifecycle_state,
            "assurance_level": episteme.assurance_level.value
                if isinstance(episteme.assurance_level, AssuranceLevel)
                else episteme.assurance_level,
            "temporal_stance": episteme.temporal_stance.value,
            "fgr": {
                "formality": 0,
                "claim_scope": {},
                "reliability": 0.0
            },
            "evidence_ids": [str(eid) for eid in episteme.evidence_ids],
            "created_at": episteme.created_at.isoformat(),
            "updated_at": episteme.updated_at.isoformat()
        }

    def _doc_to_episteme(self, doc: dict) -> UEpisteme:
        """Convert ArangoDB document to UEpisteme."""
        edition_data = doc.get("edition", {})

        supersedes_raw = edition_data.get("supersedes")
        supersedes = UUID(supersedes_raw) if supersedes_raw else None

        created_at_raw = edition_data.get("created_at")
        if created_at_raw:
            edition_created = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        else:
            edition_created = datetime.now(timezone.utc)

        lifecycle_raw = doc.get("lifecycle_state", 0)
        if isinstance(lifecycle_raw, int):
            lifecycle_state = LifecycleState(lifecycle_raw)
        elif isinstance(lifecycle_raw, str):
            lifecycle_state = LifecycleState[lifecycle_raw.upper()] if lifecycle_raw.upper() in LifecycleState.__members__ else LifecycleState.EXPLORATION
        else:
            lifecycle_state = LifecycleState.EXPLORATION

        assurance_raw = doc.get("assurance_level", 0)
        if isinstance(assurance_raw, int):
            assurance_level = AssuranceLevel(assurance_raw)
        elif isinstance(assurance_raw, str):
            assurance_level = AssuranceLevel[assurance_raw.upper()] if assurance_raw.upper() in AssuranceLevel.__members__ else AssuranceLevel.L0
        else:
            assurance_level = AssuranceLevel.L0

        doc_created = doc.get("created_at")
        if doc_created:
            created_at = datetime.fromisoformat(doc_created.replace("Z", "+00:00"))
        else:
            created_at = datetime.now(timezone.utc)

        doc_updated = doc.get("updated_at")
        if doc_updated:
            updated_at = datetime.fromisoformat(doc_updated.replace("Z", "+00:00"))
        else:
            updated_at = datetime.now(timezone.utc)

        evidence_ids = [
            UUID(eid) for eid in doc.get("evidence_ids", [])
            if eid
        ]

        grounding_raw = doc.get("grounding_holon_id")
        grounding_holon_id = UUID(grounding_raw) if grounding_raw else None

        slots_data = doc.get("slots", {})

        return UEpisteme(
            holon_id=HolonId(
                id=UUID(doc["_key"]),
                context_id=doc["context_id"],
                edition=Edition(
                    number=edition_data.get("number", 1),
                    supersedes=supersedes,
                    created_at=edition_created
                )
            ),
            described_entity=doc["described_entity"],
            claim_graph=doc.get("claim_graph", {}),
            grounding_holon_id=grounding_holon_id,
            viewpoint=doc.get("viewpoint"),
            slots=StrictDistinctionSlots(
                structure=slots_data.get("structure"),
                order=slots_data.get("order"),
                time=slots_data.get("time"),
                work=slots_data.get("work"),
                values=slots_data.get("values")
            ),
            lifecycle_state=lifecycle_state,
            assurance_level=assurance_level,
            temporal_stance=TemporalStance(doc.get("temporal_stance", 0)),
            evidence_ids=evidence_ids,
            created_at=created_at,
            updated_at=updated_at
        )

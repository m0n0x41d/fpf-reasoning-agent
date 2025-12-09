"""
Research session persistence for cross-session work.

Enables:
- Resume interrupted research
- Track long-running investigations
- Share sessions across instances
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4

from arango.database import StandardDatabase


class ResearchSessionStore:
    """
    Persists research sessions for cross-session continuity.

    Sessions track:
    - Research question and methodology
    - Associated hypotheses and epistemes
    - Current state (active, paused, completed, archived)
    """

    def __init__(self, db: StandardDatabase):
        self.db = db
        self._sessions = db.collection("sessions")

    def create(
        self,
        context_id: str,
        research_question: str,
        methodology: str = "systematic"
    ) -> dict:
        """Create new research session."""
        session_id = uuid4()
        now = datetime.now(timezone.utc)

        doc = {
            "_key": str(session_id),
            "context_id": context_id,
            "research_question": research_question,
            "methodology": methodology,
            "state": "active",
            "hypothesis_ids": [],
            "episteme_ids": [],
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }

        result = self._sessions.insert(doc, return_new=True)
        return result["new"]

    def get(self, session_id: UUID | str) -> Optional[dict]:
        """Get session by ID."""
        try:
            return self._sessions.get(str(session_id))
        except Exception:
            return None

    def list_all(self, context_id: Optional[str] = None) -> List[dict]:
        """List all sessions, optionally filtered by context."""
        if context_id:
            query = """
            FOR s IN sessions
                FILTER s.context_id == @ctx
                SORT s.updated_at DESC
                RETURN s
            """
            cursor = self.db.aql.execute(query, bind_vars={"ctx": context_id})
        else:
            query = """
            FOR s IN sessions
                SORT s.updated_at DESC
                RETURN s
            """
            cursor = self.db.aql.execute(query)

        return list(cursor)

    def list_active(self, context_id: Optional[str] = None) -> List[dict]:
        """List active sessions."""
        if context_id:
            query = """
            FOR s IN sessions
                FILTER s.state == "active"
                FILTER s.context_id == @ctx
                SORT s.updated_at DESC
                RETURN s
            """
            cursor = self.db.aql.execute(query, bind_vars={"ctx": context_id})
        else:
            query = """
            FOR s IN sessions
                FILTER s.state == "active"
                SORT s.updated_at DESC
                RETURN s
            """
            cursor = self.db.aql.execute(query)

        return list(cursor)

    def list_by_state(self, state: str) -> List[dict]:
        """List sessions by state."""
        query = """
        FOR s IN sessions
            FILTER s.state == @state
            SORT s.updated_at DESC
            RETURN s
        """
        cursor = self.db.aql.execute(query, bind_vars={"state": state})
        return list(cursor)

    def add_episteme(self, session_id: str, episteme_id: UUID) -> None:
        """Add episteme to session."""
        query = """
        FOR s IN sessions
            FILTER s._key == @key
            UPDATE s WITH {
                episteme_ids: APPEND(s.episteme_ids, @eid, true),
                updated_at: @now
            } IN sessions
        """
        self.db.aql.execute(query, bind_vars={
            "key": session_id,
            "eid": str(episteme_id),
            "now": datetime.now(timezone.utc).isoformat()
        })

    def add_hypothesis(self, session_id: str, hypothesis_id: UUID) -> None:
        """Add hypothesis to session."""
        query = """
        FOR s IN sessions
            FILTER s._key == @key
            UPDATE s WITH {
                hypothesis_ids: APPEND(s.hypothesis_ids, @hid, true),
                updated_at: @now
            } IN sessions
        """
        self.db.aql.execute(query, bind_vars={
            "key": session_id,
            "hid": str(hypothesis_id),
            "now": datetime.now(timezone.utc).isoformat()
        })

    def update_state(self, session_id: str, state: str) -> None:
        """Update session state."""
        query = """
        UPDATE @key WITH {
            state: @state,
            updated_at: @now
        } IN sessions
        """
        self.db.aql.execute(query, bind_vars={
            "key": session_id,
            "state": state,
            "now": datetime.now(timezone.utc).isoformat()
        })

    def update_methodology(self, session_id: str, methodology: str) -> None:
        """Update session methodology."""
        query = """
        UPDATE @key WITH {
            methodology: @methodology,
            updated_at: @now
        } IN sessions
        """
        self.db.aql.execute(query, bind_vars={
            "key": session_id,
            "methodology": methodology,
            "now": datetime.now(timezone.utc).isoformat()
        })

    def pause(self, session_id: str) -> None:
        """Pause session."""
        self.update_state(session_id, "paused")

    def resume(self, session_id: str) -> None:
        """Resume paused session."""
        self.update_state(session_id, "active")

    def complete(self, session_id: str) -> None:
        """Mark session as completed."""
        self.update_state(session_id, "completed")

    def archive(self, session_id: str) -> None:
        """Archive session."""
        self.update_state(session_id, "archived")

    def delete(self, session_id: str) -> bool:
        """Delete session."""
        try:
            self._sessions.delete(session_id)
            return True
        except Exception:
            return False

    def get_session_epistemes(self, session_id: str) -> List[str]:
        """Get all episteme IDs for a session."""
        session = self.get(session_id)
        if session:
            return session.get("episteme_ids", [])
        return []

    def get_session_hypotheses(self, session_id: str) -> List[str]:
        """Get all hypothesis IDs for a session."""
        session = self.get(session_id)
        if session:
            return session.get("hypothesis_ids", [])
        return []

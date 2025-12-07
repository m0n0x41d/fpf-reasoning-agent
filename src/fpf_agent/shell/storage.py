"""
SQLite storage for chat history.

Shell layer: handles all database I/O.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import aiosqlite


@dataclass
class ChatRecord:
    """Chat session record"""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass
class MessageRecord:
    """Chat message record"""
    id: str
    chat_id: str
    role: str  # "user" or "assistant"
    content: str
    metadata: dict
    created_at: datetime


class ChatStorage:
    """Async SQLite storage for chats."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
                )
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_chat_id
                ON messages(chat_id)
            """)

            await db.commit()

    async def create_chat(self, title: str | None = None) -> ChatRecord:
        """Create a new chat session."""
        chat_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        title = title or f"Chat {now[:10]}"

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (chat_id, title, now, now),
            )
            await db.commit()

        return ChatRecord(
            id=chat_id,
            title=title,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
        )

    async def get_chat(self, chat_id: str) -> ChatRecord | None:
        """Get chat by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM chats WHERE id = ?", (chat_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return ChatRecord(
                        id=row["id"],
                        title=row["title"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
        return None

    async def list_chats(self, limit: int = 50) -> list[ChatRecord]:
        """List recent chats."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM chats ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    ChatRecord(
                        id=row["id"],
                        title=row["title"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                    for row in rows
                ]

    async def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and its messages."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            cursor = await db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def update_chat_title(self, chat_id: str, title: str) -> bool:
        """Update chat title."""
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, chat_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> MessageRecord:
        """Add a message to a chat."""
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        metadata = metadata or {}

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO messages
                   (id, chat_id, role, content, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (message_id, chat_id, role, content, json.dumps(metadata), now),
            )
            await db.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?",
                (now, chat_id),
            )
            await db.commit()

        return MessageRecord(
            id=message_id,
            chat_id=chat_id,
            role=role,
            content=content,
            metadata=metadata,
            created_at=datetime.fromisoformat(now),
        )

    async def get_messages(self, chat_id: str) -> list[MessageRecord]:
        """Get all messages for a chat."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
                (chat_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    MessageRecord(
                        id=row["id"],
                        chat_id=row["chat_id"],
                        role=row["role"],
                        content=row["content"],
                        metadata=json.loads(row["metadata"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                    for row in rows
                ]

    async def get_message_history(self, chat_id: str) -> list[dict[str, str]]:
        """Get message history in format suitable for LLM context."""
        messages = await self.get_messages(chat_id)
        return [{"role": m.role, "content": m.content} for m in messages]

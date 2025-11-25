"""Database management for conversation history using libSQL/Turso."""

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

import libsql_client


@dataclass
class Session:
    """Represents a conversation session."""

    id: str
    name: str
    model: str
    created_at: int
    updated_at: int
    message_count: int = 0
    total_tokens: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.metadata:
            data["metadata"] = json.dumps(self.metadata)
        return data


@dataclass
class Message:
    """Represents a single message in a conversation."""

    id: int
    session_id: str
    role: str
    content: str
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_content: Optional[str] = None
    created_at: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.metadata:
            data["metadata"] = json.dumps(self.metadata)
        return data


class ConversationDatabase:
    """Manages conversation history using libSQL."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str,
        sync_url: Optional[str] = None,
        auth_token: Optional[str] = None,
    ):
        """Initialize the conversation database.

        Args:
            db_path: Path to the local database file
            sync_url: Optional remote Turso URL for syncing
            auth_token: Optional auth token for remote sync
        """
        # Expand user home directory
        self.db_path = os.path.expanduser(db_path)

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Build connection URL
        self.url = f"file:{self.db_path}"
        self.sync_url = sync_url
        self.auth_token = auth_token

        # Initialize database
        self._init_database()

    def _get_client(self):
        """Get a database client connection."""
        if self.sync_url and self.auth_token:
            return libsql_client.create_client_sync(
                self.url, sync_url=self.sync_url, auth_token=self.auth_token
            )
        else:
            return libsql_client.create_client_sync(self.url)

    def _init_database(self):
        """Initialize database schema if needed."""
        with self._get_client() as client:
            # Check if schema_version table exists
            result = client.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )

            if not result.rows:
                # New database - create schema
                self._create_schema(client)
            else:
                # Check schema version
                result = client.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
                current_version = result.rows[0]["version"] if result.rows else 0

                if current_version < self.SCHEMA_VERSION:
                    self._migrate_schema(client, current_version)

    def _create_schema(self, client):
        """Create the database schema."""
        # Enable foreign keys
        client.execute("PRAGMA foreign_keys = ON")

        # Create sessions table
        client.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                message_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)

        client.execute("CREATE INDEX idx_sessions_updated_at ON sessions(updated_at DESC)")

        # Create messages table
        client.execute("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                reasoning_content TEXT,
                created_at INTEGER NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)

        client.execute("CREATE INDEX idx_messages_session_id ON messages(session_id, created_at)")

        # Create schema_version table
        client.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at INTEGER NOT NULL
            )
        """)

        client.execute("INSERT INTO schema_version VALUES (?, ?)", [self.SCHEMA_VERSION, int(time.time())])

    def _migrate_schema(self, client, from_version: int):
        """Migrate schema from one version to another."""
        # Future migrations will go here
        pass

    def sync(self):
        """Sync local database with remote (if configured)."""
        if self.sync_url and self.auth_token:
            with self._get_client() as client:
                client.sync()

    # Session Management

    def create_session(
        self,
        session_id: str,
        name: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create a new conversation session.

        Args:
            session_id: Unique identifier for the session
            name: Display name for the session
            model: Model being used
            metadata: Optional metadata dictionary

        Returns:
            Created Session object
        """
        now = int(time.time())

        with self._get_client() as client:
            metadata_json = json.dumps(metadata) if metadata else None

            client.execute(
                """
                INSERT INTO sessions (id, name, model, created_at, updated_at, message_count, total_tokens, metadata)
                VALUES (?, ?, ?, ?, ?, 0, 0, ?)
                """,
                [session_id, name, model, now, now, metadata_json],
            )

        return Session(
            id=session_id,
            name=name,
            model=model,
            created_at=now,
            updated_at=now,
            message_count=0,
            total_tokens=0,
            metadata=metadata,
        )

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session object or None if not found
        """
        with self._get_client() as client:
            result = client.execute("SELECT * FROM sessions WHERE id = ?", [session_id])

            if not result.rows:
                return None

            row = result.rows[0]
            metadata = json.loads(row["metadata"]) if row["metadata"] else None

            return Session(
                id=row["id"],
                name=row["name"],
                model=row["model"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
                total_tokens=row["total_tokens"],
                metadata=metadata,
            )

    def list_sessions(self, limit: int = 50) -> List[Session]:
        """List all sessions ordered by last update.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of Session objects
        """
        with self._get_client() as client:
            result = client.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                [limit],
            )

            sessions = []
            for row in result.rows:
                metadata = json.loads(row["metadata"]) if row["metadata"] else None
                sessions.append(
                    Session(
                        id=row["id"],
                        name=row["name"],
                        model=row["model"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        message_count=row["message_count"],
                        total_tokens=row["total_tokens"],
                        metadata=metadata,
                    )
                )

            return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        with self._get_client() as client:
            # Enable foreign keys
            client.execute("PRAGMA foreign_keys = ON")

            # Delete messages first (manual cascade since PRAGMA might not work in libSQL)
            client.execute("DELETE FROM messages WHERE session_id = ?", [session_id])

            # Delete session
            result = client.execute("DELETE FROM sessions WHERE id = ?", [session_id])
            return result.rows_affected > 0

    def update_session(
        self,
        session_id: str,
        name: Optional[str] = None,
        model: Optional[str] = None,
        message_count: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> bool:
        """Update session metadata.

        Args:
            session_id: Session identifier
            name: New name (optional)
            model: New model (optional)
            message_count: New message count (optional)
            total_tokens: New total tokens (optional)

        Returns:
            True if session was updated, False if not found
        """
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if model is not None:
            updates.append("model = ?")
            params.append(model)
        if message_count is not None:
            updates.append("message_count = ?")
            params.append(message_count)
        if total_tokens is not None:
            updates.append("total_tokens = ?")
            params.append(total_tokens)

        if not updates:
            return False

        # Always update updated_at
        updates.append("updated_at = ?")
        params.append(int(time.time()))

        # Add session_id to params
        params.append(session_id)

        with self._get_client() as client:
            query = f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?"
            result = client.execute(query, params)
            return result.rows_affected > 0

    # Message Management

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        reasoning_content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a message to a session.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            model: Model used for this message
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            reasoning_content: Reasoning content for thinking models
            metadata: Optional metadata dictionary

        Returns:
            Created Message object
        """
        now = int(time.time())
        total_tokens = None

        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        metadata_json = json.dumps(metadata) if metadata else None

        with self._get_client() as client:
            result = client.execute(
                """
                INSERT INTO messages (
                    session_id, role, content, model,
                    prompt_tokens, completion_tokens, total_tokens,
                    reasoning_content, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    session_id,
                    role,
                    content,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    reasoning_content,
                    now,
                    metadata_json,
                ],
            )

            message_id = result.last_insert_rowid

            # Update session stats
            session = self.get_session(session_id)
            if session:
                new_message_count = session.message_count + 1
                new_total_tokens = session.total_tokens + (total_tokens or 0)
                self.update_session(
                    session_id,
                    message_count=new_message_count,
                    total_tokens=new_total_tokens,
                )

        return Message(
            id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_content=reasoning_content,
            created_at=now,
            metadata=metadata,
        )

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Message]:
        """Get messages for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return (most recent first)

        Returns:
            List of Message objects ordered by creation time
        """
        with self._get_client() as client:
            if limit:
                # Get most recent N messages
                result = client.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [session_id, limit],
                )
                # Reverse to get chronological order
                rows = list(reversed(result.rows))
            else:
                result = client.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
                    [session_id],
                )
                rows = result.rows

            messages = []
            for row in rows:
                metadata = json.loads(row["metadata"]) if row["metadata"] else None
                messages.append(
                    Message(
                        id=row["id"],
                        session_id=row["session_id"],
                        role=row["role"],
                        content=row["content"],
                        model=row["model"],
                        prompt_tokens=row["prompt_tokens"],
                        completion_tokens=row["completion_tokens"],
                        total_tokens=row["total_tokens"],
                        reasoning_content=row["reasoning_content"],
                        created_at=row["created_at"],
                        metadata=metadata,
                    )
                )

            return messages

    def get_message_history_for_llm(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get message history formatted for LLM API calls.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to include (most recent)

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        messages = self.get_messages(session_id, limit=limit)
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def clear_messages(self, session_id: str) -> bool:
        """Clear all messages in a session.

        Args:
            session_id: Session identifier

        Returns:
            True if messages were deleted
        """
        with self._get_client() as client:
            client.execute("DELETE FROM messages WHERE session_id = ?", [session_id])

            # Reset session stats
            self.update_session(session_id, message_count=0, total_tokens=0)

            return True

    # Export/Import

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a session and all its messages to a dictionary.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session data and messages, or None if not found
        """
        session = self.get_session(session_id)
        if not session:
            return None

        messages = self.get_messages(session_id)

        return {
            "session": session.to_dict(),
            "messages": [msg.to_dict() for msg in messages],
            "exported_at": int(time.time()),
            "version": 1,
        }

    def import_session(self, data: Dict[str, Any], overwrite: bool = False) -> Optional[Session]:
        """Import a session from exported data.

        Args:
            data: Exported session data
            overwrite: If True, overwrite existing session

        Returns:
            Imported Session object, or None if session exists and overwrite=False
        """
        session_data = data["session"]
        messages_data = data["messages"]

        # Check if session exists
        existing = self.get_session(session_data["id"])
        if existing and not overwrite:
            return None

        # Delete existing session if overwriting
        if existing:
            self.delete_session(session_data["id"])

        # Create session
        metadata = json.loads(session_data["metadata"]) if session_data.get("metadata") else None
        session = self.create_session(
            session_id=session_data["id"],
            name=session_data["name"],
            model=session_data["model"],
            metadata=metadata,
        )

        # Import messages
        for msg_data in messages_data:
            msg_metadata = json.loads(msg_data["metadata"]) if msg_data.get("metadata") else None
            self.add_message(
                session_id=session.id,
                role=msg_data["role"],
                content=msg_data["content"],
                model=msg_data.get("model"),
                prompt_tokens=msg_data.get("prompt_tokens"),
                completion_tokens=msg_data.get("completion_tokens"),
                reasoning_content=msg_data.get("reasoning_content"),
                metadata=msg_metadata,
            )

        return session

"""Tests for database operations."""

import os
import tempfile
import time
from pathlib import Path

import pytest

from llm_cli.database import ConversationDatabase, Session, Message


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = ConversationDatabase(db_path)
        yield db


class TestConversationDatabase:
    """Test ConversationDatabase class."""

    def test_init_creates_schema(self, temp_db):
        """Test that initialization creates the schema."""
        # Database should be created and schema initialized
        assert os.path.exists(temp_db.db_path)

    def test_create_session(self, temp_db):
        """Test creating a session."""
        session = temp_db.create_session(
            session_id="test-1",
            name="Test Session",
            model="gpt-4o",
        )

        assert session.id == "test-1"
        assert session.name == "Test Session"
        assert session.model == "gpt-4o"
        assert session.message_count == 0
        assert session.total_tokens == 0

    def test_get_session(self, temp_db):
        """Test retrieving a session."""
        temp_db.create_session(
            session_id="test-1",
            name="Test Session",
            model="gpt-4o",
        )

        session = temp_db.get_session("test-1")
        assert session is not None
        assert session.id == "test-1"
        assert session.name == "Test Session"

    def test_get_nonexistent_session(self, temp_db):
        """Test retrieving a session that doesn't exist."""
        session = temp_db.get_session("nonexistent")
        assert session is None

    def test_list_sessions(self, temp_db):
        """Test listing sessions."""
        # Create multiple sessions
        temp_db.create_session("session-1", "Session 1", "gpt-4o")
        temp_db.create_session("session-2", "Session 2", "claude-3")
        temp_db.create_session("session-3", "Session 3", "gemini")

        sessions = temp_db.list_sessions()
        assert len(sessions) == 3

        # Verify all sessions are returned
        session_ids = {s.id for s in sessions}
        assert session_ids == {"session-1", "session-2", "session-3"}

    def test_list_sessions_with_limit(self, temp_db):
        """Test listing sessions with a limit."""
        for i in range(10):
            temp_db.create_session(f"session-{i}", f"Session {i}", "gpt-4o")

        sessions = temp_db.list_sessions(limit=5)
        assert len(sessions) == 5

    def test_delete_session(self, temp_db):
        """Test deleting a session."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")

        result = temp_db.delete_session("test-1")
        assert result is True

        session = temp_db.get_session("test-1")
        assert session is None

    def test_delete_nonexistent_session(self, temp_db):
        """Test deleting a session that doesn't exist."""
        result = temp_db.delete_session("nonexistent")
        assert result is False

    def test_update_session(self, temp_db):
        """Test updating a session."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")

        result = temp_db.update_session(
            "test-1",
            name="Updated Session",
            message_count=10,
            total_tokens=500,
        )
        assert result is True

        session = temp_db.get_session("test-1")
        assert session.name == "Updated Session"
        assert session.message_count == 10
        assert session.total_tokens == 500

    def test_add_message(self, temp_db):
        """Test adding a message to a session."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")

        message = temp_db.add_message(
            session_id="test-1",
            role="user",
            content="Hello, world!",
            model="gpt-4o",
        )

        assert message.session_id == "test-1"
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.model == "gpt-4o"

        # Check that session stats were updated
        session = temp_db.get_session("test-1")
        assert session.message_count == 1

    def test_add_message_with_tokens(self, temp_db):
        """Test adding a message with token counts."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")

        message = temp_db.add_message(
            session_id="test-1",
            role="assistant",
            content="Hello!",
            model="gpt-4o",
            prompt_tokens=10,
            completion_tokens=5,
        )

        assert message.prompt_tokens == 10
        assert message.completion_tokens == 5
        assert message.total_tokens == 15

        # Check that session stats were updated
        session = temp_db.get_session("test-1")
        assert session.total_tokens == 15

    def test_get_messages(self, temp_db):
        """Test retrieving messages for a session."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")

        temp_db.add_message("test-1", "user", "Message 1", "gpt-4o")
        time.sleep(0.01)
        temp_db.add_message("test-1", "assistant", "Message 2", "gpt-4o")
        time.sleep(0.01)
        temp_db.add_message("test-1", "user", "Message 3", "gpt-4o")

        messages = temp_db.get_messages("test-1")
        assert len(messages) == 3
        assert messages[0].content == "Message 1"
        assert messages[1].content == "Message 2"
        assert messages[2].content == "Message 3"

    def test_get_messages_with_limit(self, temp_db):
        """Test retrieving messages with a limit."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")

        for i in range(10):
            temp_db.add_message("test-1", "user", f"Message {i}", "gpt-4o")
            time.sleep(0.01)

        messages = temp_db.get_messages("test-1", limit=5)
        assert len(messages) == 5
        # Should get the most recent 5 messages in chronological order
        assert messages[0].content == "Message 5"
        assert messages[4].content == "Message 9"

    def test_get_message_history_for_llm(self, temp_db):
        """Test formatting messages for LLM API."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")

        temp_db.add_message("test-1", "user", "Hello", "gpt-4o")
        temp_db.add_message("test-1", "assistant", "Hi there!", "gpt-4o")
        temp_db.add_message("test-1", "user", "How are you?", "gpt-4o")

        messages = temp_db.get_message_history_for_llm("test-1")

        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}
        assert messages[2] == {"role": "user", "content": "How are you?"}

    def test_clear_messages(self, temp_db):
        """Test clearing messages from a session."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")

        temp_db.add_message("test-1", "user", "Message 1", "gpt-4o")
        temp_db.add_message("test-1", "assistant", "Message 2", "gpt-4o")

        result = temp_db.clear_messages("test-1")
        assert result is True

        messages = temp_db.get_messages("test-1")
        assert len(messages) == 0

        # Check that session stats were reset
        session = temp_db.get_session("test-1")
        assert session.message_count == 0
        assert session.total_tokens == 0

    def test_export_session(self, temp_db):
        """Test exporting a session."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")
        temp_db.add_message("test-1", "user", "Hello", "gpt-4o")
        temp_db.add_message("test-1", "assistant", "Hi!", "gpt-4o", prompt_tokens=10, completion_tokens=5)

        data = temp_db.export_session("test-1")

        assert data is not None
        assert data["session"]["id"] == "test-1"
        assert data["session"]["name"] == "Test Session"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["total_tokens"] == 15
        assert "exported_at" in data
        assert data["version"] == 1

    def test_export_nonexistent_session(self, temp_db):
        """Test exporting a session that doesn't exist."""
        data = temp_db.export_session("nonexistent")
        assert data is None

    def test_import_session(self, temp_db):
        """Test importing a session."""
        # Create and export a session
        temp_db.create_session("test-1", "Test Session", "gpt-4o")
        temp_db.add_message("test-1", "user", "Hello", "gpt-4o")
        temp_db.add_message("test-1", "assistant", "Hi!", "gpt-4o")

        data = temp_db.export_session("test-1")

        # Delete the session
        temp_db.delete_session("test-1")

        # Import it back
        session = temp_db.import_session(data)

        assert session is not None
        assert session.id == "test-1"
        assert session.name == "Test Session"

        messages = temp_db.get_messages("test-1")
        assert len(messages) == 2

    def test_import_session_no_overwrite(self, temp_db):
        """Test that import fails if session exists and overwrite=False."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")
        data = temp_db.export_session("test-1")

        result = temp_db.import_session(data, overwrite=False)
        assert result is None

    def test_import_session_with_overwrite(self, temp_db):
        """Test that import succeeds with overwrite=True."""
        temp_db.create_session("test-1", "Original", "gpt-4o")
        temp_db.add_message("test-1", "user", "Original message", "gpt-4o")

        # Create export data with different content
        data = {
            "session": {
                "id": "test-1",
                "name": "Updated",
                "model": "claude-3",
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
                "message_count": 0,
                "total_tokens": 0,
                "metadata": None,
            },
            "messages": [
                {
                    "id": 1,
                    "session_id": "test-1",
                    "role": "user",
                    "content": "New message",
                    "model": "claude-3",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "reasoning_content": None,
                    "created_at": int(time.time()),
                    "metadata": None,
                }
            ],
            "exported_at": int(time.time()),
            "version": 1,
        }

        result = temp_db.import_session(data, overwrite=True)
        assert result is not None

        session = temp_db.get_session("test-1")
        assert session.name == "Updated"
        assert session.model == "claude-3"

        messages = temp_db.get_messages("test-1")
        assert len(messages) == 1
        assert messages[0].content == "New message"

    def test_message_with_reasoning_content(self, temp_db):
        """Test adding a message with reasoning content."""
        temp_db.create_session("test-1", "Test Session", "deepseek-r1")

        message = temp_db.add_message(
            session_id="test-1",
            role="assistant",
            content="The answer is 42.",
            model="deepseek-r1",
            reasoning_content="Let me think about this step by step...",
        )

        assert message.reasoning_content == "Let me think about this step by step..."

        # Verify it's persisted
        messages = temp_db.get_messages("test-1")
        assert messages[0].reasoning_content == "Let me think about this step by step..."

    def test_message_with_metadata(self, temp_db):
        """Test adding a message with metadata."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")

        metadata = {"images": ["/path/to/image.jpg"], "context_file": "context.txt"}

        message = temp_db.add_message(
            session_id="test-1",
            role="user",
            content="Analyze this image",
            model="gpt-4o",
            metadata=metadata,
        )

        assert message.metadata == metadata

        # Verify it's persisted
        messages = temp_db.get_messages("test-1")
        assert messages[0].metadata == metadata

    def test_session_with_metadata(self, temp_db):
        """Test creating a session with metadata."""
        metadata = {"created_by": "test", "tags": ["important", "work"]}

        session = temp_db.create_session(
            session_id="test-1",
            name="Test Session",
            model="gpt-4o",
            metadata=metadata,
        )

        assert session.metadata == metadata

        # Verify it's persisted
        session = temp_db.get_session("test-1")
        assert session.metadata == metadata

    def test_delete_session_cascades_messages(self, temp_db):
        """Test that deleting a session also deletes its messages."""
        temp_db.create_session("test-1", "Test Session", "gpt-4o")
        temp_db.add_message("test-1", "user", "Message 1", "gpt-4o")
        temp_db.add_message("test-1", "assistant", "Message 2", "gpt-4o")

        temp_db.delete_session("test-1")

        messages = temp_db.get_messages("test-1")
        assert len(messages) == 0

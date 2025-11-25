# Conversation History Design Document

## Overview
Add conversation session management to StreamLM using Turso (libSQL) for local-first storage with optional remote sync.

## Database Schema

### Tables

#### 1. `sessions`
Stores metadata about conversation sessions.

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,                    -- UUID or slug like "my-project"
    name TEXT NOT NULL,                     -- Display name
    model TEXT NOT NULL,                    -- Model used (can change per message)
    created_at INTEGER NOT NULL,            -- Unix timestamp
    updated_at INTEGER NOT NULL,            -- Unix timestamp
    message_count INTEGER DEFAULT 0,        -- Cache for quick display
    total_tokens INTEGER DEFAULT 0,         -- Total tokens used
    metadata TEXT                           -- JSON for extensibility
);

CREATE INDEX idx_sessions_updated_at ON sessions(updated_at DESC);
```

#### 2. `messages`
Stores individual messages in conversations.

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,               -- Foreign key to sessions.id
    role TEXT NOT NULL,                     -- "user", "assistant", "system"
    content TEXT NOT NULL,                  -- Message content
    model TEXT,                             -- Model used for this message
    prompt_tokens INTEGER,                  -- Tokens in prompt
    completion_tokens INTEGER,              -- Tokens in completion
    total_tokens INTEGER,                   -- Total tokens
    reasoning_content TEXT,                 -- For reasoning models (think mode)
    created_at INTEGER NOT NULL,            -- Unix timestamp
    metadata TEXT,                          -- JSON for images, context files, etc.
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_session_id ON messages(session_id, created_at);
```

#### 3. `schema_version`
Track database migrations.

```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);

INSERT INTO schema_version VALUES (1, strftime('%s', 'now'));
```

## Architecture

### New Module: `src/llm_cli/database.py`

```python
class ConversationDatabase:
    """Manages conversation history using Turso/libSQL."""

    def __init__(self, db_path: str, sync_url: Optional[str] = None, auth_token: Optional[str] = None)

    # Session Management
    def create_session(self, session_id: str, name: str, model: str) -> Session
    def get_session(self, session_id: str) -> Optional[Session]
    def list_sessions(self, limit: int = 50) -> List[Session]
    def delete_session(self, session_id: str) -> bool
    def update_session(self, session_id: str, **kwargs) -> bool

    # Message Management
    def add_message(self, session_id: str, role: str, content: str, **kwargs) -> Message
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Message]
    def get_message_history_for_llm(self, session_id: str) -> List[Dict[str, str]]
    def clear_messages(self, session_id: str) -> bool

    # Sync (for remote Turso)
    def sync(self) -> None

    # Export/Import
    def export_session(self, session_id: str, format: str = "json") -> str
    def import_session(self, data: str) -> Session
```

### Data Models

```python
@dataclass
class Session:
    id: str
    name: str
    model: str
    created_at: int
    updated_at: int
    message_count: int = 0
    total_tokens: int = 0
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Message:
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
```

## CLI Changes

### New Flags for `chat` command

```python
@app.command()
def chat(
    # ... existing parameters ...

    # Session management
    session: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Session ID to continue conversation"
    ),
    session_name: Optional[str] = typer.Option(
        None,
        "--session-name",
        help="Name for the session (only when creating)"
    ),
):
```

### New Commands

```python
@app.command()
def sessions(
    list_all: bool = typer.Option(False, "--list", "-l", help="List all sessions"),
    show: Optional[str] = typer.Option(None, "--show", help="Show session details"),
    delete: Optional[str] = typer.Option(None, "--delete", help="Delete a session"),
    export: Optional[str] = typer.Option(None, "--export", help="Export session to JSON"),
    clear: Optional[str] = typer.Option(None, "--clear", help="Clear messages in session"),
):
    """Manage conversation sessions."""
```

## User Experience

### Creating/Continuing Sessions

```bash
# First message - create new session
lm --session my-project "How do I implement authentication?"
# Output: Created session 'my-project'
# ... response ...

# Continue the conversation
lm --session my-project "Can you show me an example?"
# ... continues with context ...

# Use a different model in same session
lm --session my-project --model gpt-4 "What about security?"
```

### Managing Sessions

```bash
# List all sessions
lm sessions --list
# Output:
# ID              Name            Messages  Tokens   Updated
# my-project      My Project      5         2,340    2 hours ago
# debugging       Debug Session   12        5,120    1 day ago

# Show session details
lm sessions --show my-project
# Output: Shows full conversation history

# Export session
lm sessions --export my-project > session.json

# Clear session messages
lm sessions --clear my-project

# Delete session
lm sessions --delete my-project
```

### Auto-session Mode (Optional Future Enhancement)

```bash
# Configure auto-session in config
auto_session: true
auto_session_prefix: "auto"

# Then every conversation gets a session automatically
lm "hello"  # Creates session "auto-2025-01-24-1"
lm "how are you?"  # ERROR: No session specified
lm --session auto-2025-01-24-1 "how are you?"  # Continues
```

## Configuration

Add to `~/.streamlm/config.yaml`:

```yaml
# Database configuration
database:
  # Local database path
  path: "~/.streamlm/conversations.db"

  # Optional: Remote Turso sync
  # sync_url: "libsql://your-database.turso.io"
  # auth_token: "${TURSO_AUTH_TOKEN}"

  # Auto-sync after each message (only if remote configured)
  auto_sync: true

# Session defaults
sessions:
  # Maximum messages to keep in context (None = unlimited)
  max_context_messages: 50

  # Automatically create session if not specified
  auto_create: false

  # Default session name pattern
  default_name_pattern: "Session {date}"
```

## Implementation Steps

1. ✅ Design database schema
2. Add `libsql` dependency to pyproject.toml
3. Create `src/llm_cli/database.py` with ConversationDatabase class
4. Create database initialization and migration system
5. Add session flags to `chat` command
6. Integrate session loading into chat flow
7. Add message saving after each LLM response
8. Create `sessions` command for management
9. Add export/import functionality
10. Write comprehensive tests
11. Update documentation

## Testing Strategy

### Unit Tests
- Database operations (CRUD for sessions and messages)
- Message history formatting for LLM context
- Export/import functionality
- Migration system

### Integration Tests
- Full conversation flow with session persistence
- Multi-turn conversations
- Session management commands
- Token tracking accuracy

### Manual Testing
- Local-only mode
- Remote sync with actual Turso instance
- Large conversation handling
- Concurrent session usage

## Backwards Compatibility

- Sessions are **opt-in** - existing behavior unchanged without `--session` flag
- No breaking changes to existing CLI
- Database created on first use of session features
- Graceful degradation if database unavailable

## Future Enhancements

1. **Context Window Management**: Automatically trim old messages when approaching context limit
2. **Search**: Full-text search across all sessions
3. **Tags/Labels**: Categorize sessions
4. **Cost Tracking**: Track API costs per session
5. **Branching**: Create branches from specific points in conversation
6. **Shared Sessions**: Sync sessions across devices via Turso remote
7. **AI Summaries**: Automatically generate session summaries
8. **Templates**: Save and reuse conversation templates

## Security Considerations

1. **Local Database**: Stored in user directory with appropriate permissions
2. **No Sensitive Data**: API keys NOT stored in database
3. **Remote Sync**: Optional, user must configure
4. **Export Safety**: JSON export may contain sensitive conversation data
5. **SQL Injection**: Use parameterized queries throughout

## Performance

1. **Local-First**: No network calls for local-only usage
2. **Indexed Queries**: Proper indexes on session_id and timestamps
3. **Lazy Loading**: Only load messages when needed
4. **Efficient Context**: Limit context window to prevent bloat
5. **Async Sync**: Remote sync won't block CLI operations

## Dependencies

```toml
[project.dependencies]
libsql = ">=0.4.0"  # Turso client library
```

No additional dependencies needed - libsql is lightweight and well-maintained.

# Database Structure Documentation

## Overview

The project uses PostgreSQL as its database system. The schema is designed to support user authentication, chat sessions, file management, AI model integration, and dashboard metrics tracking. This document outlines the tables, relationships, and key functionalities implemented in the database.

## Tables

### 1. `users`
- Primary user account information
- Contains authentication details and role assignment
- Fields:
  - `id` (UUID, PK): Unique identifier
  - `username` (varchar): Unique username
  - `password` (text): Hashed password
  - `role` (varchar): User role (default: 'user')
  - `email` (varchar): User email
  - `name` (varchar): User's full name
  - `created_at`/`updated_at` (timestamp): Tracking dates

### 2. `sessions`
- Stores active user sessions
- Fields:
  - `id` (UUID, PK): Session identifier
  - `user_id` (UUID, FK): Reference to user
  - `token` (text): Session token
  - `expires_at` (timestamp): Expiration time
  - `created_at` (timestamp): Creation time

### 3. `chat_sessions`
- Represents a conversation thread
- Fields:
  - `id` (UUID, PK): Unique session identifier
  - `user_id` (UUID, FK): Owner of the session
  - `title` (varchar): Session title
  - `last_message_timestamp` (timestamp): Last activity
  - `created_at` (timestamp): Creation time
  - `is_active` (boolean): Session status
  - `use_rag` (boolean): Whether to use RAG for this session (default: true)
  - `rag_collections` (jsonb): Document collections to use for RAG

### 4. `messages`
- Stores individual messages in chat sessions
- Fields:
  - `id` (bigint, PK): Message identifier
  - `user_id` (UUID, FK): Message sender
  - `message` (text): User's message content
  - `response` (text): AI response
  - `file_path` (text): Optional reference to file
  - `timestamp` (timestamp): Message time
  - `session_id` (UUID, FK): Parent chat session
  - `model_id` (UUID, FK): AI model used
  - `retrieved_chunks` (jsonb): Information about document chunks retrieved for this message

### 5. `ai_models`
- Configuration for available AI models (including Ollama models)
- Fields:
  - `id` (UUID, PK): Model identifier
  - `model_id` (varchar): Model identifier used in chat interfaces
  - `ollama_model_id` (varchar): The ID used by Ollama (e.g., "llama3:latest")
  - `name` (varchar): Human-readable name
  - `description` (text): Model description
  - `parameters` (jsonb): Configuration parameters (size, etc.)
  - `is_active` (boolean): Availability status
  - `created_at` (timestamp): Creation time
  - `updated_at` (timestamp): Last update time

### 6. `messages`
- Stores individual messages in chat sessions
- Fields:
  - `id` (bigint, PK): Message identifier
  - `user_id` (UUID, FK): Message sender
  - `message` (text): User's message content
  - `response` (text): AI response
  - `file_path` (text): Optional reference to file
  - `timestamp` (timestamp): Message time
  - `session_id` (UUID, FK): Parent chat session
  - `model_id` (UUID, FK): AI model used
  - `retrieved_chunks` (jsonb): Information about document chunks retrieved for this message

### 7. `mcp_connections`
- Tracks connections to external MCP systems
- Fields:
  - `id` (bigint, PK): Connection identifier
  - `user_id` (UUID, FK): Connection owner
  - `mcp_host` (varchar): Host address
  - `mcp_port` (integer): Connection port
  - `status` (varchar): Connection status
  - `last_file_path` (text): Last accessed file
  - `created_at` (timestamp): Creation time

### 8. `dashboard_metrics`
- Stores aggregated metrics for dashboard
- Fields:
  - `id` (integer, PK): Metric identifier
  - `metric_name` (varchar): Metric identifier
  - `metric_value` (jsonb): Structured metric data
  - `updated_at` (timestamp): Last update time

### 9. `schema_migrations`
- Tracks applied database migrations
- Fields:
  - `id` (PK): Unique identifier
  - `version` (varchar): Migration version number
  - `applied_at` (timestamp): When the migration was applied
  - `description` (text): Description of the migration

## AI Integration Tables

### vector_stores
- Stores vector database configurations
- Fields:
  - `id` (UUID, PK): Store identifier
  - `name` (varchar): Store name
  - `store_type` (varchar): Type (e.g., "chroma")
  - `connection_string` (text): Connection details
  - `is_active` (boolean): Availability flag
  - `created_at`/`updated_at` (timestamp): Tracking dates
  - `config` (jsonb): Configuration parameters

### document_collections
- Organizes documents into searchable collections
- Fields:
  - `id` (UUID, PK): Collection identifier
  - `name` (varchar): Collection name
  - `description` (text): Collection description
  - `user_id` (UUID, FK): Collection owner
  - `vector_store_id` (UUID, FK): Associated vector store
  - `embedding_model` (varchar): Model for embeddings
  - `created_at`/`updated_at` (timestamp): Tracking dates
  - `is_active` (boolean): Availability flag
  - `metadata` (jsonb): Additional metadata

### documents
- Manages document storage and metadata
- Fields:
  - `id` (SERIAL, PK): Document identifier
  - `user_id` (UUID, FK): Document owner
  - `filename` (varchar, NOT NULL): Filename for storage
  - `original_name` (varchar, NOT NULL): Original filename
  - `file_path` (text, NOT NULL): Storage path
  - `file_type` (varchar, NOT NULL): Document type (e.g., "pdf", "docx")
  - `file_size` (bigint, NOT NULL): Size of the file in bytes
  - `mime_type` (varchar): MIME type of the document
  - `status` (varchar): Processing status (default: 'UPLOADED')
  - `processing_error` (text): Error message if processing failed
  - `collection_id` (UUID): Parent collection (optional)
  - `session_id` (UUID): Associated chat session (optional)
  - `created_at` (timestamp): Creation time
  - `updated_at` (timestamp): Last update time
  - Foreign key relationships:
    - `user_id` references `users(id)` ON DELETE CASCADE

### document_chunks
- Stores document segments for retrieval
- Fields:
  - `id` (UUID, PK): Chunk identifier
  - `document_id` (UUID, FK): Parent document
  - `chunk_index` (integer): Position in document
  - `content` (text): Chunk textual content
  - `vector_id` (varchar): Vector store reference identifier
  - `embedding` (vector or bytea): Embedding vector (vector type if pg_vector extension available, bytea otherwise)
  - `token_count` (integer): Estimated token count
  - `created_at` (timestamp): Creation time
  - `metadata` (jsonb): Additional chunk metadata (e.g., page number, section)

### rag_settings
- Configures RAG parameters system-wide
- Fields:
  - `id` (serial, PK): Setting identifier
  - `embedding_model` (varchar): Default embedding model
  - `chunk_size` (integer): Default chunk size
  - `chunk_overlap` (integer): Chunk overlap size
  - `similarity_top_k` (integer): Number of chunks to retrieve
  - `search_type` (varchar): Search algorithm type
  - `created_at`/`updated_at` (timestamp): Tracking dates
  - `config` (jsonb): Additional configuration parameters

### ollama_settings
- Stores Ollama AI server connection details and default model for AI chat integration
- Fields:
  - `id` (serial): Primary key
  - `host` (varchar): Ollama server hostname or IP
  - `port` (integer): Ollama server port
  - `default_model` (varchar): Default AI model name. This is used across the application including MCP AI Agent when no specific model is provided.
  - `created_at` (timestamp): Record creation timestamp
  - `updated_at` (timestamp): Record update timestamp

### AI Model Selection Priority
The application uses the following priority order when selecting an AI model for various operations:
1. **User explicitly selected model**: Model selected in the UI dropdown and stored in localStorage
2. **Default model from database**: The `default_model` field from the `ollama_settings` table
3. **Environment variable**: The `DEFAULT_OLLAMA_MODEL` environment variable if set
4. **Fallback model**: A hardcoded fallback (currently 'llama3')

This priority system ensures that users get a consistent experience across different features including chat, RAG, and MCP Agent functions.

### ai_models
- Stores information about available AI models that can be used in the application
- Fields:
  - `id` (UUID): Primary key
  - `model_id` (varchar): Model identifier used in chat interfaces
  - `ollama_model_id` (varchar): The ID used by Ollama (e.g., "llama3:latest")
  - `name` (varchar): Human-readable name
  - `description` (text): Model description
  - `parameters` (jsonb): Configuration parameters (size, etc.)
  - `is_active` (boolean): Availability status - only active models are shown in selection dropdowns
  - `created_at` (timestamp): Creation time
  - `updated_at` (timestamp): Last update time

### schema_migrations
- Tracks applied database migrations
- Fields:
  - `id` (PK): Unique identifier
  - `version` (varchar): Migration version number
  - `applied_at` (timestamp): When the migration was applied
  - `description` (text): Description of the migration

#### Protocol for Database Changes
- All new tables and schema changes must be documented in this file.
- For every schema change, provide a SQL migration script in `src/scripts/sql/`.
- Document new relationships, triggers, and indexes as needed.

## Key Relationships

1. A user can have multiple:
   - Chat sessions (1:N)
   - Messages (1:N)
   - Documents (1:N)
   - MCP connections (1:N)
   - Sessions (1:N)

2. A chat session:
   - Belongs to one user (N:1)
   - Contains multiple messages (1:N)

3. A message:
   - Belongs to one user (N:1)
   - Belongs to one chat session (N:1)
   - Can reference one AI model (N:1)

4. A document:
   - Belongs to one user (N:1)
   - Can belong to one collection (N:1, optional)
   - Can be associated with one chat session (N:1, optional)
   - Has multiple document chunks (1:N)

## Functions & Triggers

1. `update_dashboard_metrics()`: Updates dashboard statistics based on current data
2. `trigger_update_dashboard_metrics()`: Wrapper to call the update function from triggers
3. `update_chat_session_timestamp()`: Updates session timestamp when a new message is added

Triggers are set up to:
- Update dashboard metrics when users, messages, or PDFs change
- Update chat session timestamps when new messages are added

## Indexes

- `idx_chat_sessions_timestamp`: Optimizes sorting chat sessions by last activity
- `idx_chat_sessions_user_id`: Speeds up finding a user's chat sessions
- `idx_messages_session_id`: Accelerates retrieving messages for a specific session
- `idx_ollama_settings_updated_at`: Index on ollama_settings.updated_at
- `idx_ai_models_ollama_model_id`: Fast lookup of AI models by their Ollama ID
- `idx_ai_models_is_active`: Quick filtering of active models
- `idx_ai_models_name`: Optimizes searching/sorting models by name
- `idx_vector_stores_is_active`: Quick filtering of active vector stores
- `idx_vector_stores_name`: Optimizes searching/sorting vector stores by name
- `idx_document_collections_user_id`: Speeds up finding a user's collections
- `idx_document_collections_is_active`: Quick filtering of active collections
- `idx_document_collections_vector_store_id`: Accelerates finding collections for a vector store
- `idx_documents_user_id`: Speeds up finding a user's documents
- `idx_documents_collection_id`: Accelerates finding documents in a collection
- `idx_documents_session_id`: Accelerates finding documents associated with a chat session
- `idx_documents_status`: Filters documents by processing status
- `idx_documents_created_at`: Optimizes sorting documents by creation date
- `idx_document_chunks_document_id`: Accelerates retrieving chunks for a document
- `idx_document_chunks_vector_id`: Fast lookup of chunks by vector ID

## Design Considerations

1. **UUID for Primary Keys**: Most entities use UUID for primary keys, providing better security and distribution
2. **Cascading Deletes**: When a parent record is deleted, related records are automatically removed
3. **JSON Storage**: Uses JSONB for flexible storage of metrics and model parameters
4. **Timestamp Tracking**: All entities include creation timestamps, many track updates
5. **Database-level Automation**: Triggers ensure dashboard metrics stay current automatically

## Schema Evolution

The schema represents a transition from the initial SQLite design to a more robust PostgreSQL implementation with:
- Enhanced data types (UUID, JSONB)
- Improved automation through triggers
- Better performance through indexing
- Support for more complex relationships
- Integration with Ollama AI services

### Recent Schema Updates

#### AI Models Schema Enhancement (2023-08-20)
- Added `
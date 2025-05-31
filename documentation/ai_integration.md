# AI Integration Documentation

## Overview
The Platform Dashboard implements a sophisticated AI integration using Ollama, featuring real-time streaming responses, Retrieval-Augmented Generation (RAG), and a hybrid approach to document processing. The system provides a seamless chat experience with context-aware responses and source citations.

## Core Components

### Backend Services
1. **Ollama Service** (`src/services/ollamaService.js`):
   - Manages communication with Ollama API
   - Handles streaming responses
   - Provides embedding generation
   - Supports multiple models

2. **RAG Service** (`src/services/ragService.js`):
   - Orchestrates document retrieval and context augmentation
   - Manages vector database interactions
   - Provides context-aware responses
   - Handles source citations

3. **Vector Store Service** (`src/services/vectorStoreService.js`):
   - Manages ChromaDB interactions
   - Handles document embeddings storage
   - Provides semantic search capabilities

### Frontend Components
1. **Chat Interface** (`client/src/pages/Chatbot.tsx`):
   - Real-time message streaming
   - RAG toggle functionality
   - Source citation display
   - Model selection

2. **Message Components**:
   - `ChatMessage.tsx`: Handles message display and markdown rendering
   - `MessageList.tsx`: Manages message history and streaming states
   - `ChatInput.tsx`: User input with RAG controls

## Streaming Implementation

### Frontend Streaming
The chat interface implements streaming using the following approach:

1. **Message Creation**:
   ```typescript
   const aiMessage = {
     id: aiMessageId,
     role: 'assistant',
     content: '',
     timestamp: new Date(),
     isStreaming: true,
     useRag: shouldUseRag
   };
   ```

2. **Stream Processing**:
   - Uses a WebSocket or SSE connection for real-time updates
   - Maintains a content reference for accumulating streamed text
   - Updates UI in real-time as chunks arrive
   - Handles completion and error states

3. **State Management**:
   - Tracks streaming state with `isStreaming` flag
   - Uses `streamedContentRef` for content accumulation
   - Manages abort functionality for stream cancellation

### Backend Streaming
The Ollama service implements streaming through:

1. **Stream Setup**:
   ```javascript
   const response = await client.post('/api/chat', {
     model,
     messages: formattedMessages,
     stream: true
   }, {
     responseType: 'stream'
   });
   ```

2. **Chunk Processing**:
   - Parses incoming chunks in real-time
   - Validates chunk structure
   - Formats content for frontend consumption
   - Handles stream completion

3. **Error Handling**:
   - Manages connection issues
   - Handles timeout scenarios
   - Provides fallback mechanisms

## RAG Implementation

### Document Processing
1. **Text Extraction**:
   - Supports multiple document formats (PDF, DOCX, etc.)
   - Implements chunking with overlap
   - Maintains document metadata

2. **Embedding Generation**:
   - Uses Ollama's `nomic-embed-text` model
   - Processes chunks in batches
   - Stores embeddings in ChromaDB

3. **Vector Storage**:
   - ChromaDB for vector storage
   - Efficient similarity search
   - Metadata management

### Query Processing
1. **Context Retrieval**:
   ```javascript
   const ragResult = await retrieveContext(message, {
     model: embeddingModel,
     sessionId,
     topK: 5
   });
   ```

2. **Response Generation**:
   - Combines retrieved context with user query
   - Uses system prompts for response formatting
   - Includes source citations

3. **Source Management**:
   - Tracks source documents
   - Provides relevance scores
   - Formats citations for display

### RAG Optimization Techniques

1. **Chunking Strategies**:
   - Recursive text splitting
   - Overlap for context preservation
   - Semantic boundary detection

2. **Retrieval Enhancement**:
   - Hybrid search (semantic + keyword)
   - Re-ranking of results
   - Context window optimization

3. **Response Quality**:
   - Source validation
   - Confidence scoring
   - Fallback mechanisms

## Security and Performance

### Security Measures
1. **Authentication**:
   - Session-based auth for API access
   - Role-based access control
   - Secure credential handling

2. **Data Protection**:
   - Encrypted storage
   - Secure transmission
   - Access logging

### Performance Optimization
1. **Caching**:
   - Embedding cache
   - Response cache
   - Model loading optimization

2. **Resource Management**:
   - Connection pooling
   - Batch processing
   - Memory optimization

## Future Enhancements

1. **Advanced RAG Features**:
   - Multi-document reasoning
   - Cross-reference validation
   - Dynamic context selection

2. **Model Improvements**:
   - Custom model fine-tuning
   - Model performance metrics
   - Automated model selection

3. **User Experience**:
   - Advanced source visualization
   - Interactive context exploration
   - Feedback integration

## Troubleshooting

### Common Issues
1. **Streaming Issues**:
   - Check WebSocket connection
   - Verify Ollama server status
   - Monitor memory usage

2. **RAG Problems**:
   - Verify ChromaDB connection
   - Check embedding generation
   - Validate document processing

### Debugging Steps
1. Check server logs for errors
2. Verify model availability
3. Test document processing pipeline
4. Monitor resource usage

## Configuration

### Environment Setup
```ini
[ollama]
host=localhost
port=11434
timeout=30000

[chromadb]
host=localhost
port=8000
collection=documents

[rag]
chunk_size=1000
chunk_overlap=200
top_k=5
```

### Model Configuration
```javascript
const defaultConfig = {
  embeddingModel: 'nomic-embed-text',
  chatModel: 'qwen2',
  temperature: 0.7,
  maxTokens: 2000
};
```

## Database Schema
The AI integration utilizes the following tables (defined in `DatabaseStructure.md`):
- `ollama_settings`: Stores connection settings (host, port) for the Ollama server and the default model name. Expects a single row.
- `ai_models`: Stores information about available AI models synced from the Ollama server, including their Ollama ID (`ollama_model_id`), display name, description, and an `is_active` flag to control availability in the chat UI.

## API Endpoints (Backend)

### Settings Management (Admin Only)
- `GET /api/ollama/settings`: Retrieve current Ollama settings from the `ollama_settings` table.
- `PUT /api/ollama/settings`: Update Ollama server settings in the `ollama_settings` table.
- `POST /api/ollama/test-connection`: Test connectivity to the configured Ollama server (e.g., by calling `/api/version`).

### Model Management (Admin Only, except GET active)
- `GET /api/ollama/models/available`: Fetch models directly from the configured Ollama server's `/api/tags` endpoint.
- `POST /api/ollama/models/sync`: Fetch models from Ollama server (`/api/tags`) and update the `ai_models` table in the database.
- `GET /api/ollama/models/db`: Get all models currently stored in the `ai_models` database table (for admin UI).
- `PUT /api/ollama/models/:id/status`: Toggle the `is_active` status of a model in the `ai_models` table.
- `GET /api/ollama/models/active`: Get active AI models (`is_active = true`) from the `ai_models` table (for chat UI model selector).

### Chat Interaction (Authenticated Users)
- `POST /api/ollama/chat`: Takes a user message, conversation history, and selected model ID (from `ai_models` table). Formats the request for Ollama's `/api/chat` endpoint, calls it using the configured server and selected model, and streams the response back to the client.

### API Status
- `GET /api/ollama/status`: Check if the configured Ollama API is reachable from the backend.

## Workflow

1.  **Admin Configures:** An admin user navigates to Settings -> Ollama Integration.
2.  **Enter Details:** Enters the Ollama server Host and Port.
3.  **Test Connection:** Clicks "Test Connection" to verify the backend can reach the Ollama server.
4.  **Save Settings:** Clicks "Save Settings" to store the host/port in the `ollama_settings` table.
5.  **Fetch/Sync Models:** Clicks "Fetch Available Models" (optional preview) and then "Sync Models". The backend calls Ollama's `/api/tags`, gets the list of models (e.g., `llama3:latest`), and populates/updates the `ai_models` table in the database.
6.  **Enable Models:** Admin enables desired models using the toggle switches in the UI (updates `is_active` flag in `ai_models`).
7.  **User Chats:** A regular user goes to the Chat page.
8.  **Select Model:** The user selects an *active* model from the dropdown (populated by `/api/ollama/models/active`).
9.  **Send Message:** User types a message and sends it.
10. **Backend Processes:** The frontend sends the message, chat history, and selected model DB ID to `/api/ollama/chat`.
11. **Ollama Interaction:** The backend retrieves the actual Ollama model name, formats the request for `/api/chat`, calls the Ollama server, and streams the response back.
12. **Display Response:** The frontend displays the streamed response in the chat UI.

## Endpoint Choice: `/api/chat` vs `/api/generate`

We are using Ollama's **`/api/chat`** endpoint for the chatbot interaction.
- **Reasoning:** It natively supports conversational history through the `messages` array, simplifying backend logic compared to the deprecated `context` parameter in `/api/generate`.
- **Future-Proofing:** The `/api/chat` endpoint supports `tools` (function calling), which provides a better foundation for future RAG implementations or agentic features where the AI might need to interact with external knowledge sources or tools.

## Usage Guidelines

### Setting Up Ollama
1.  Install Ollama ([link](https://ollama.ai/download)).
2.  Pull desired models (e.g., `ollama pull llama3`).
3.  Ensure the Ollama server is running.
4.  Configure the connection in the Platform Dashboard admin settings.
5.  Test the connection and sync models.
6.  Enable the models you want to make available for chat.

### Using AI Chat
1.  Navigate to the Chat interface.
2.  Select an enabled AI model from the dropdown.
3.  Type your prompt and submit. The conversation history will be automatically sent with subsequent messages in the same session.

## Performance Considerations
- Model response time depends on the model size, hardware (CPU/GPU), and prompt complexity.
- The first request to a model might be slower as Ollama loads it into memory. Use `ollama ps` to see loaded models. [See Ollama FAQ](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-tell-if-my-model-was-loaded-onto-the-gpu)
- Streaming responses improve perceived performance for the user.

## Security Notes
- Ollama API does not have built-in authentication. Access should be controlled at the network level (firewall) or via a reverse proxy if exposed.
- Ensure the application backend properly authenticates and authorizes requests to its own Ollama-related endpoints (e.g., admin-only for settings).
- Consider data privacy: prompts and responses are sent to the Ollama server.

## Future Enhancements
- Implementing response streaming in the backend and frontend.
- Allowing configuration of model parameters (temperature, context window size `num_ctx`, etc.) per request or globally. [See Ollama API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- User-specific model preferences.
- Integrating RAG capabilities using the `/api/chat` endpoint's structure. 
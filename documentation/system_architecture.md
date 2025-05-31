# AI Chatbot System Architecture Documentation

## Overview

This document provides a detailed explanation of the AI chatbot system architecture, which allows users to interact with AI models through a web interface. The system uses Ollama as the AI backend, with a React frontend and Node.js/Express backend.

## System Components

### Backend Components

#### 1. `ollama.js` - API Routes

This file defines the Express router for Ollama-related endpoints, handling:

- Ollama settings management (GET/PUT /api/ollama/settings)
- Connection testing (POST /api/ollama/test-connection)
- Model management (GET /api/ollama/models/*)
- Chat interactions (POST /api/ollama/chat)
- RAG (Retrieval Augmented Generation) functionality for document-based Q&A

Key responsibilities:
- Routes HTTP requests to appropriate handler functions
- Implements middleware for authentication
- Structures API responses in a consistent format
- Handles streaming responses for real-time AI interactions

#### 2. `ollamaService.js` - Ollama Communication Service

This service handles direct communication with the Ollama AI server:

- Manages Ollama connection settings (host, port, default model)
- Creates and maintains HTTP clients for Ollama API
- Tests connections to verify Ollama server availability
- Fetches, syncs, and manages AI models from Ollama
- Processes chat requests, including streaming responses
- Handles embeddings generation for RAG functionality

Key capabilities:
- Connection pool management
- Error handling and retry logic
- Streaming response handling
- Database integration for model persistence
- Advanced chat formatting and processing

#### 3. `chatbot.js` - Chat Management Routes

This file implements chat session management endpoints:

- Session creation, retrieval, updating, and deletion
- Message management and storage
- File uploads and document processing
- Chat history pagination

Key features:
- User authentication and session validation
- File upload handling and processing
- Message persistence in the database
- Session management and organization

### Frontend Services

#### 4. `ollamaService.ts` - Frontend Ollama Service

This TypeScript service provides frontend access to Ollama functionality:

- Defines TypeScript interfaces for Ollama data types
- Implements API calls to backend Ollama endpoints
- Handles settings management, model retrieval, and status checks
- Provides methods for testing connections and syncing models

Key methods:
- `getOllamaSettings`, `saveOllamaSettings`
- `testOllamaConnection`
- `getAvailableOllamaModels`, `syncOllamaModels`
- `toggleOllamaModelStatus`

#### 5. `chatbotService.ts` - Frontend Chat Service

This service handles chat interactions from the frontend:

- Session management (create, get, update, delete)
- Message sending, including file attachments
- Session file management
- Document status tracking

Key methods:
- `createSession`, `getSessions`, `getSession`
- `sendMessage`, `sendMessageWithFile`
- `getSessionFiles`, `getDocumentStatus`

### Frontend UI Components

#### 6. `Chatbot.tsx` - Main Chat Interface

This React component implements the main chat interface:

- Manages chat state (sessions, messages, models)
- Handles user interactions (sending messages, selecting models)
- Implements real-time message streaming from AI
- Manages file uploads and document processing
- Integrates with RAG functionality for document-based chat
- Supports MCP (Model Context Protocol) integration for agent-based interactions

Key features:
- Real-time message updates with streaming
- File upload and document processing UI
- Session management and navigation
- RAG mode for document-based Q&A
- MCP integration for advanced agent capabilities

#### 7. `ChatMessage.tsx` - Message Display Component

This component renders individual chat messages:

- Formats and displays user and AI messages
- Implements markdown rendering with syntax highlighting
- Handles file attachment display and status indicators
- Shows document processing status
- Displays RAG sources when available

Key features:
- Syntax highlighting for code blocks
- Copy-to-clipboard functionality
- File attachment visualization
- Loading animations for streaming messages
- Collapsible sources panel for RAG responses

#### 8. `MessageList.tsx` - Message Container Component

This component manages the display of multiple messages:

- Renders the list of messages in a conversation
- Implements infinite scrolling for message history
- Groups messages by sender for better readability
- Displays loading indicators for message streaming
- Shows MCP agent commands and results

Key features:
- Auto-scrolling to latest messages
- Message grouping and formatting
- Infinite scrolling with loading indicators
- Empty state handling
- MCP agent command visualization

#### 9. `OllamaSettings.tsx` - Settings Management UI

This component provides a user interface for Ollama settings:

- Displays and allows editing of Ollama connection settings
- Provides model management capabilities (sync, activate, deactivate)
- Shows connection status and test functionality
- Allows selection of default model for chat
- Displays model details and statistics

Key features:
- Connection testing with visual feedback
- Model synchronization from Ollama server
- Model activation/deactivation controls
- Default model selection
- Error handling and notification system

## Data Flow

### Chat Interaction Flow

1. **User Input**:
   - User enters a message in the `ChatInput` component within `Chatbot.tsx`
   - Message is captured in state and UI is updated to show the message

2. **API Request**:
   - Frontend calls `chatbotService.sendMessage()` (or `sendMessageWithFile()` if uploading a file)
   - If using direct AI completion, it also calls `aiChatService.streamChatCompletion()`
   - If using RAG, it calls `ragChatService.sendRagChatMessage()`

3. **Backend Processing**:
   - Request is received by `chatbot.js` endpoints (`/message` or `/message-with-file`)
   - Message is stored in the database
   - If requesting AI completion, forwards request to Ollama via `ollama.js` endpoint (`/chat`)

4. **Ollama Service Processing**:
   - `ollamaService.js` handles the chat request
   - Creates a properly formatted request to the Ollama API
   - Manages streaming or non-streaming interactions
   - Processes response from Ollama

5. **Response Handling**:
   - For streaming responses, chunks are sent back to the client in real-time
   - Each chunk updates the UI via state in `Chatbot.tsx`
   - The complete message is saved to the database once streaming is complete

6. **UI Update**:
   - Messages are rendered through `MessageList.tsx` and `ChatMessage.tsx`
   - Streaming indicators show progress
   - Code blocks, markdown, and attachments are properly formatted

### Document/RAG Flow

1. **File Upload**:
   - User uploads a file through the chat interface
   - File is sent using `chatbotService.sendMessageWithFile()`
   - Backend stores the file and creates a document record

2. **Document Processing**:
   - Backend processes the document (extract text, generate embeddings)
   - Status is tracked and can be polled via WebSockets or HTTP

3. **RAG Query**:
   - When RAG is enabled, user questions are sent with `ragChatService.sendRagChatMessage()`
   - Backend retrieves relevant document segments using vector search
   - AI response is generated with document context

4. **Result Display**:
   - RAG responses include cited sources
   - `ChatMessage.tsx` displays these sources in a collapsible panel

## Configuration

### Ollama Settings

The system requires configuration of Ollama connection settings:

- Host: The server hosting Ollama (default: localhost)
- Port: The port Ollama is running on (default: 11434)
- Default model: The preferred model to use for chat

These settings are managed through the `OllamaSettings.tsx` UI and stored in the database via the `ollamaService.js` backend.

### Model Management

AI models are managed through:

1. **Discovery**: Fetching available models from Ollama server
2. **Synchronization**: Adding/updating models in the local database
3. **Activation**: Enabling or disabling specific models for use
4. **Selection**: Choosing which model to use for a specific chat

## Advanced Features

### 1. Real-time Streaming

The system implements real-time streaming of AI responses:

- Uses Server-Sent Events (SSE) for backend-to-frontend streaming
- Updates UI incrementally as tokens are generated
- Provides abort capabilities to stop generation
- Maintains proper state management during streaming

### 2. RAG (Retrieval Augmented Generation)

The system implements document-based Q&A through RAG:

- Document uploading and processing
- Vector embedding generation
- Semantic search for relevant context
- Source citation in responses
- UI for viewing cited sources

### 3. MCP (Model Context Protocol) Integration

For advanced agent capabilities, the system integrates with MCP:

- Connects to external MCP servers
- Executes tools and commands through agents
- Displays command results in the chat interface
- Maintains session context across interactions

## Troubleshooting

Common issues and solutions:

1. **Connection Issues**:
   - Verify Ollama server is running
   - Check host/port settings
   - Ensure network connectivity between components

2. **Model Problems**:
   - Sync models if not appearing
   - Check model activation status
   - Verify model is properly installed on Ollama server

3. **Performance Issues**:
   - Large files may take time to process
   - Complex RAG queries require more processing time
   - Streaming reduces perceived latency but total generation time remains similar

## Conclusion

This AI chatbot system provides a comprehensive solution for interacting with Ollama-powered AI models through a web interface. Its modular architecture allows for flexibility and extensibility, while the tight integration between frontend and backend components ensures a smooth user experience with advanced features like streaming responses, document Q&A, and agent capabilities. 
# Five Most Important Questions for Developers

## 1. How does the authentication system work and how do I integrate with it?

The platform uses a session-based authentication system with role-based access control:

- **Authentication Flow**:
  - Users submit credentials to `/api/auth/login`
  - Backend validates credentials and creates a session
  - Session token is stored in HTTP-only cookies
  - Frontend stores authentication state in React context
  - Protected routes check authentication status via `PrivateRoute` and `AdminRoute` components

- **User Roles**:
  - Admin: Full access to all features including user management
  - User: Standard access to chatbot and other features

- **Integration Points**:
  - Frontend: Use the `AuthContext` to access user state and authentication functions
  - Backend: Use the `requireAuth` middleware to protect API routes
  - Admin-only features: Use the `requireAdmin` middleware or `AdminRoute` component

- **Default Credentials**:
  - Username: admin
  - Password: admin

## 2. How does the MCP (Model Context Protocol) integration work?

The MCP integration allows AI models to interact with system resources through a standardized protocol:

- **Architecture**:
  - MCP Server: Standalone service that provides tools for system interaction
  - MCP Client: Built into the platform to connect to MCP servers
  - MCP Agent: AI-powered agent that can use MCP tools to perform tasks

- **Key Components**:
  - `MCPContext`: Manages connections to MCP servers
  - `MCPAgentContext`: Processes user requests and generates commands
  - `mcpService.js`: Backend service for MCP server connections
  - `MCPServerSelector`: UI for selecting and connecting to MCP servers

- **Connection Flow**:
  1. User enables MCP in the chatbot interface
  2. Platform connects to MCP server via Server-Sent Events (SSE)
  3. MCP server assigns a unique client ID
  4. Platform can now invoke tools on the MCP server

- **Tool Invocation**:
  - Tools are discovered from the MCP server
  - Each tool has a name, description, and parameters
  - Tools can be invoked via the `executeCommand` function in `MCPContext`

## 3. How is the chatbot functionality implemented and how can I extend it?

The chatbot is a core feature of the platform that connects to AI models:

- **Architecture**:
  - Frontend: React components in `client/src/components/chat`
  - Backend: API routes in `src/routes/chatbot.js`
  - AI Integration: Ollama service in `src/services/ollamaService.js`

- **Key Components**:
  - `Chatbot.tsx`: Main page component
  - `ChatMessage.tsx`: Renders individual messages with markdown/code support
  - `ChatInput.tsx`: Handles user input and submission
  - `ModelSelector.tsx`: Allows selection of different AI models

- **Data Flow**:
  1. User sends message via `ChatInput`
  2. Message is sent to backend via `chatbotService.sendMessage()`
  3. Backend processes message and forwards to Ollama
  4. Response is streamed back to frontend via SSE
  5. UI updates in real-time as tokens are received

- **Extension Points**:
  - Add new AI models in the `ai_models` database table
  - Implement custom message processors in the backend
  - Create new UI components for specialized message types
  - Add RAG capabilities by connecting document collections

## 4. How does the document processing and RAG (Retrieval Augmented Generation) system work?

The platform includes a document processing pipeline for RAG:

- **Architecture**:
  - Document Upload: Handled by `src/routes/documents.js`
  - Text Extraction: Uses Python's `pdfplumber` for PDFs and `mammoth` for DOCX
  - Vector Storage: Uses ChromaDB or falls back to file-based storage
  - RAG Integration: Enhances AI responses with document context

- **Processing Flow**:
  1. User uploads document via the UI
  2. Backend extracts text and generates embeddings
  3. Embeddings are stored in vector database
  4. When RAG is enabled, relevant document chunks are retrieved for each query
  5. AI model receives both the query and document context

- **Key Components**:
  - `vectorStoreService.js`: Manages document embeddings and retrieval
  - `documentProcessingService.js`: Handles document processing pipeline
  - `ragChatService.js`: Integrates RAG with chat functionality

- **Configuration**:
  - Vector store settings in `config.ini` under `[paths]` section
  - Python interpreter path for enhanced text extraction

## 5. How is the application configured and deployed?

The platform uses a centralized configuration system:

- **Configuration**:
  - Main config file: `conf/config.ini`
  - Configuration loaded by `src/utils/config.js`
  - Frontend receives configuration via `/api/config` endpoint

- **Key Configuration Sections**:
  - `[server]`: HTTP server settings
  - `[database]`: PostgreSQL connection details
  - `[paths]`: File storage locations
  - `[security]`: Authentication settings
  - `[python]`: Python interpreter path for document processing

- **Deployment**:
  - Development: `npm run dev` (backend) and `npm run dev:client` (frontend)
  - Production: `npm run deploy` (builds frontend and starts server)
  - The Express server serves both the API and static frontend files

- **Database**:
  - PostgreSQL database with tables for users, sessions, messages, etc.
  - Schema defined in `src/scripts/sql/copilotdbcreationscript.sql`
  - Migrations tracked in `schema_migrations` table

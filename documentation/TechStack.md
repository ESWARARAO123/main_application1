# Technology Stack Documentation

This document provides a comprehensive overview of the technology stack used in the Product Demo application, covering both frontend and backend components, as well as infrastructure and development tools.

## Table of Contents

1. [Backend Technologies](#backend-technologies)
2. [Frontend Technologies](#frontend-technologies)
3. [Database](#database)
4. [AI and Machine Learning](#ai-and-machine-learning)
5. [Document Processing](#document-processing)
6. [Python Integration](#python-integration)
7. [MCP (Model Context Protocol) System](#mcp-model-context-protocol-system)
8. [Infrastructure](#infrastructure)
9. [Development Tools](#development-tools)
10. [Security Components](#security-components)
11. [Deployment and DevOps](#deployment-and-devops)

## Backend Technologies

### Core Framework and Runtime
- **Node.js**: JavaScript runtime for server-side execution
- **Express.js**: Web application framework for Node.js
- **REST API**: Architecture for API endpoints
- **WebSocket (ws)**: Real-time bidirectional communication server

### Authentication and Session Management
- **express-session**: Session management middleware with MemoryStore
- **bcrypt**: Password hashing library
- **cookie-parser**: Cookie parsing middleware
- **CORS**: Cross-origin resource sharing configuration

### File Handling
- **multer**: Middleware for handling multipart/form-data (file uploads)
- **fs (File System)**: Node.js module for file operations
- **path**: Node.js module for path manipulations

### Configuration
- **ini**: Library for parsing .ini configuration files
- **dotenv**: Environment variable management
- **yargs**: Command-line argument parsing with configuration support

### Utilities
- **uuid**: Library for generating unique identifiers
- **winston**: Structured logging library
- **axios**: HTTP client for external API requests
- **eventsource**: Server-sent events support

### Remote System Management
- **node-ssh**: SSH client for remote server management
- **node-windows**: Windows service management (for production deployment)

## Frontend Technologies

### Core Framework and Libraries
- **React 18**: JavaScript library for building user interfaces with latest features
- **React Router DOM**: Client-side routing library for React applications
- **TypeScript**: Typed superset of JavaScript for enhanced development experience

### State Management
- **React Context API**: For global state management (Auth, Theme, WebSocket, MCP, etc.)
- **Custom hooks**: For component-specific state management and reusable logic
- **Context Providers**: Multiple context providers for different application domains

### UI Components and Styling
- **Chakra UI**: Comprehensive component library (@chakra-ui/react, @chakra-ui/icons, @chakra-ui/table, @chakra-ui/card)
- **Tailwind CSS**: Utility-first CSS framework with custom theme configuration
- **Emotion**: CSS-in-JS styling (@emotion/react, @emotion/styled)
- **Framer Motion**: Animation library for smooth UI transitions
- **CSS Variables**: Theme-based color system with CSS custom properties
- **Responsive Design**: Mobile-first responsive UI components

### Additional UI Libraries
- **Headless UI**: Unstyled accessible UI components (@headlessui/react)
- **Heroicons**: Icon library (@heroicons/react)
- **React Icons**: Additional icon sets
- **React Markdown**: Markdown rendering with GitHub Flavored Markdown support
- **React Syntax Highlighter**: Code syntax highlighting with multiple themes

### API Communication
- **Axios**: Primary HTTP client for API requests
- **Custom API service**: Wrapper around HTTP clients for consistent API calls
- **WebSocket API**: Real-time bidirectional communication with reconnection logic
- **Server-Sent Events**: For streaming responses and real-time updates
- **EventSource**: Browser API for SSE with polyfill support

## Database

### Primary Database
- **PostgreSQL**: Primary relational database for structured data storage
- **Better-SQLite3**: Legacy SQLite support for fallback scenarios

### Database Access
- **pg (node-postgres)**: PostgreSQL client for Node.js with connection pooling
- **Connection pooling**: Efficient database connection management with configurable pool size
- **Database abstraction layer**: Custom wrapper providing SQLite-compatible interface for PostgreSQL

### Schema Management
- **SQL migrations**: Automated database schema management with version tracking
- **Migration system**: JavaScript-based migration files with up/down support
- **Schema versioning**: Tracking applied migrations in schema_migrations table
- **Relational model**: Normalized data relationships across multiple tables

### Vector Database
- **ChromaDB**: Vector database for embeddings and similarity search
- **Docker deployment**: Containerized ChromaDB instance for isolation
- **Embedding storage**: High-dimensional vector storage for RAG functionality

## AI and Machine Learning

### LangChain Integration
- **LangChain**: Core AI application framework (^0.3.26)
- **@langchain/community**: Community integrations and tools
- **@langchain/openai**: OpenAI API integration for GPT models
- **Document loaders**: Various document processing capabilities

### Local AI Models
- **Ollama**: Local AI model serving and management (^0.5.15)
- **nomic-embed-text**: Text embedding model for local deployment
- **Model management**: Dynamic model loading and configuration

### External AI Services
- **VoyageAI**: External embedding service for enhanced text embeddings
- **OpenAI integration**: GPT model access through LangChain

### RAG (Retrieval-Augmented Generation)
- **Custom RAG implementation**: Document-based question answering system
- **Intelligent chunking**: Advanced document segmentation strategies
- **Similarity search**: Vector-based document retrieval
- **Context injection**: Dynamic context insertion into AI prompts
- **Multi-modal processing**: Support for various document types

## Document Processing

### Node.js Text Extraction
- **pdf-parse**: Basic PDF text extraction for Node.js
- **mammoth**: DOCX document processing and text extraction
- **LangChain document loaders**: Alternative document processing methods

### Document Storage
- **File system**: Local storage for uploaded documents
- **Metadata storage**: Document metadata in PostgreSQL database
- **Organized directory structure**: Systematic file organization by user/session

### Document Processing Pipeline
- **Asynchronous processing**: Background document processing with queue management
- **Progress tracking**: Real-time monitoring of document processing status
- **WebSocket notifications**: Live updates during document processing
- **Intelligent chunking**: Advanced document segmentation for optimal RAG performance
- **Fallback mechanisms**: Multiple extraction methods with graceful degradation
- **Error handling**: Robust error recovery and user feedback

## Python Integration

### Python Environment
- **Python 3.9**: Dedicated Python version for enhanced PDF processing
- **Virtual Environment**: Isolated Python environment in `python/.venv/`
- **Cross-platform support**: Windows and Linux compatibility

### Python Dependencies
- **pdfplumber (0.9.0)**: Advanced PDF text extraction with table support
- **Pillow**: Image processing library for PDF image handling
- **Wand**: ImageMagick binding for complex document processing
- **cryptography**: Security utilities for document processing

### Python Scripts
- **extract_text.py**: Basic PDF text extraction with page markers
- **extract_text_with_tables.py**: Advanced table-aware PDF extraction with Markdown output
- **Automated setup**: `installvenv.sh` script for environment setup
- **Testing utilities**: `test_extraction.sh` for verification

### Node.js-Python Integration
- **Child process execution**: Spawning Python processes from Node.js
- **JSON communication**: Structured data exchange between Node.js and Python
- **Error handling**: Robust error management across language boundaries
- **Timeout management**: Process timeout and resource management
- **Configuration integration**: Python interpreter path from config.ini

### Advanced PDF Processing Features
- **Table detection**: Intelligent table extraction and Markdown conversion
- **Layout preservation**: Maintaining document structure with page markers
- **Caption detection**: Smart detection of table captions and numbering
- **Multi-format output**: JSON structured output for seamless integration

## MCP (Model Context Protocol) System

### Terminal MCP Orchestrator
- **Python orchestrator**: `python/terminal-mcp-orchestrator/` for command execution
- **MCP client**: Python-based MCP protocol implementation
- **Tool execution**: Remote tool execution via MCP protocol
- **Request/response handling**: Structured communication with MCP servers

### MCP Dependencies
- **requests**: HTTP client for MCP communication
- **sseclient-py**: Server-sent events client for streaming responses
- **pytest**: Testing framework for MCP functionality

### MCP Server Management
- **SSH integration**: Remote MCP server installation and management
- **Configuration storage**: MCP server configurations in PostgreSQL
- **Status monitoring**: Real-time MCP server health monitoring
- **Port management**: Dynamic port discovery and configuration

### MCP Protocol Implementation
- **WebSocket communication**: Real-time MCP protocol over WebSocket
- **Tool discovery**: Dynamic tool enumeration from MCP servers
- **Command approval workflow**: User confirmation for command execution
- **Result streaming**: Real-time command output streaming

### MCP UI Components
- **Installation wizard**: Multi-step MCP server setup process
- **Server management**: Start, stop, restart, status operations
- **Remote filesystem browser**: SSH-based file system exploration
- **Command interface**: Interactive command execution with approval workflow

## Infrastructure

### Server
- **Express.js server**: For handling HTTP requests
- **CORS support**: For cross-origin resource sharing

### File Storage
- **Local file system**: For document and embedding storage
- **Configurable paths**: Via config.ini
- **Docker volumes**: For persistent storage of ChromaDB data

### Containerization
- **Docker**: For running ChromaDB in an isolated container
- **docker-compose**: For defining and running the multi-container setup
- **Container networking**: For communication between application and ChromaDB

### Environment Configuration
- **config.ini**: For application configuration
- **Environment-specific settings**: Development vs. production
- **Docker environment variables**: For container configuration

### WebSocket Infrastructure
- **Connection Management**: Singleton pattern with connection limits (max 3 per user)
- **Authentication**: Cookie-based WebSocket authentication
- **Broadcast System**: User-specific and global message broadcasting
- **State Tracking**: Connection state management across page refreshes
- **Error Recovery**: Automatic reconnection with exponential backoff
- **Message Queuing**: Storage of messages for disconnected users
- **Heartbeat Mechanism**: Connection health monitoring
- **Deduplication**: Prevention of duplicate message delivery

### MCP Infrastructure
- **Database Tables**: SSH and MCP server configurations
- **Service Layer**: Connection management and tool execution
- **Security Layer**: Command validation and approval workflow
- **UI Components**: Settings, installation wizard, remote filesystem browser, and command interface
- **Status Monitoring**: Real-time connection and execution status
- **SSH Integration**: Secure remote server management
- **Installation Process**: Multi-step wizard with directory selection
- **Port Detection**: Dynamic port discovery on target machines
- **Command Management**: Start, stop, restart, status, and uninstall operations

## Development Tools

### Build System
- **React Scripts**: Create React App build system with TypeScript support
- **cross-env**: Cross-platform environment variable management
- **PostCSS**: CSS processing and transformation
- **Autoprefixer**: Automatic CSS vendor prefixing

### Code Quality and Linting
- **ESLint**: JavaScript/TypeScript linting with custom rules
- **@typescript-eslint**: TypeScript-specific ESLint rules and parser
- **Prettier integration**: Code formatting (via IDE extensions)

### Development Workflow
- **nodemon**: Automatic server restarts during development
- **concurrently**: Running multiple development processes simultaneously
- **npm scripts**: Comprehensive script collection for development tasks
- **Hot reloading**: React development server with hot module replacement

### Testing Framework
- **Jest**: JavaScript testing framework for unit and integration tests
- **React Testing Library**: Testing utilities for React components
- **pytest**: Python testing framework for Python components

### Development Environment
- **VS Code support**: Optimized for Visual Studio Code development
- **TypeScript IntelliSense**: Enhanced code completion and error detection
- **Debugging support**: Integrated debugging for both Node.js and React

### Version Control
- **Git**: Source code management with branching strategies
- **GitHub**: Repository hosting, collaboration, and CI/CD integration

## Security Components

### Authentication
- **Session-based authentication**: Using express-session
- **Password hashing**: Using bcrypt

### API Security
- **Input validation**: For preventing injection attacks
- **CORS configuration**: For controlling API access

### File Security
- **File type validation**: For preventing malicious uploads
- **File size limits**: For preventing DoS attacks

### WebSocket Security
- **Connection Authentication**: Cookie-based validation
- **Connection Limits**: Maximum 3 connections per user
- **Message Validation**: Input validation for WebSocket messages
- **State Management**: Secure connection state tracking
- **Heartbeat System**: Connection health monitoring

### MCP Security
- **Credential Encryption**: AES-256-GCM for SSH passwords
- **Command Validation**: Server-side command verification
- **Approval Workflow**: User confirmation for command execution
- **Access Control**: Role-based access to MCP features
- **Audit Logging**: Command execution and error tracking
- **Input Sanitization**: Prevention of command injection
- **Secure Directory Handling**: Path validation and sanitization
- **Port Specification**: Proper handling of SSH port configurations
- **Connection Verification**: Pre-execution connection testing

## Deployment and DevOps

### Production Deployment
- **Docker**: For containerized deployment of ChromaDB and other services
- **docker-compose**: For orchestrating multi-container applications
- **Volume mapping**: For persistent data storage in containers
- **Container health checks**: For monitoring container status

### Process Management
- **PM2**: For Node.js process management in production (optional)

### Monitoring
- **Console logging**: For basic application monitoring
- **Custom logger**: For structured logging
- **WebSocket status**: Real-time connection monitoring
- **MCP status tracking**: Server connection and execution monitoring
- **Document processing tracking**: Progress and status monitoring

---

## Technology Stack Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Browser                         │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                      Frontend (React)                       │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ React Router│   │ Context API  │   │ UI Components    │  │
│  └─────────────┘   └──────────────┘   └──────────────────┘  │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ TypeScript  │   │ CSS Modules  │   │ API Service      │  │
│  └─────────────┘   └──────────────┘   └──────────────────┘  │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ WebSocket   │   │ MCP UI       │   │ Document UI      │  │
│  └─────────────┘   └──────────────┘   └──────────────────┘  │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                     Backend (Node.js)                       │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ Express.js  │   │ Authentication│  │ File Handling    │  │
│  └─────────────┘   └──────────────┘   └──────────────────┘  │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ REST API    │   │ Document     │   │ RAG Implementation│ │
│  │             │   │ Processing   │   │                  │  │
│  └─────────────┘   └──────────────┘   └─────────┬────────┘  │
│                                                 │           │
│  ┌─────────────┐   ┌──────────────┐   ┌────────┴────────┐  │
│  │ WebSocket   │   │ MCP Service  │   │ SSH Service     │  │
│  │ Server      │   │              │   │                 │  │
│  └─────────────┘   └──────────────┘   └─────────────────┘  │
└───┬───────────────────────────────────────────┬─┼───────────┘
    │                                           │ │
    │                                           │ │
┌───▼───────────────────────┐   ┌───────────────▼─┘           ┐
│      PostgreSQL Database  │   │  ┌─────────────────────────┐│
│                           │   │  │  Docker Container       ││
│  ┌─────────────────────┐  │   │  │  ┌───────────────────┐  ││
│  │ User Data           │  │   │  │  │    ChromaDB       │  ││
│  └─────────────────────┘  │   │  │  │    Vector DB      │  ││
│                           │   │  │  └───────────────────┘  ││
│  ┌─────────────────────┐  │   │  └─────────────────────────┘│
│  │ Document Metadata   │  │   │                             │
│  └─────────────────────┘  │   │  ┌─────────────────────┐    │
│                           │   │  │ Uploaded Files      │    │
│  ┌─────────────────────┐  │   │  └─────────────────────┘    │
│  │ Chat History        │  │   │                             │
│  └─────────────────────┘  │   │  ┌─────────────────────┐    │
│                           │   │  │ Embeddings          │    │
│  ┌─────────────────────┐  │   │  └─────────────────────┘    │
│  │ MCP Configurations  │  │   │                             │
│  └─────────────────────┘  │   │  ┌─────────────────────┐    │
│                           │   │  │ Remote MCP Servers  │    │
│  ┌─────────────────────┐  │   │  └─────────────────────┘    │
│  │ SSH Configurations  │  │   │                             │
│  └─────────────────────┘  │   └─────────────────────────────┘
└───────────────────────────┘
```

## Dependency Management

The application uses npm for dependency management across multiple package.json files. Key dependencies include:

### Backend Dependencies (Root package.json)
- **express**: Web framework for Node.js
- **pg**: PostgreSQL client with connection pooling
- **bcrypt**: Password hashing and security
- **multer**: File upload handling middleware
- **ws**: WebSocket server implementation
- **node-ssh**: SSH client for remote server management
- **winston**: Structured logging library
- **axios**: HTTP client for external API requests
- **uuid**: Unique identifier generation
- **ini**: Configuration file parsing
- **yargs**: Command-line argument parsing

### AI and Document Processing
- **langchain**: AI application framework
- **@langchain/community**: Community integrations
- **@langchain/openai**: OpenAI API integration
- **ollama**: Local AI model serving
- **chromadb**: Vector database client
- **pdf-parse**: Basic PDF text extraction
- **mammoth**: DOCX document processing
- **voyageai**: External embedding service

### Frontend Dependencies (Client package.json)
- **react**: UI library (v18)
- **react-dom**: React DOM rendering
- **react-router-dom**: Client-side routing
- **typescript**: Type checking and development
- **@chakra-ui/react**: Component library
- **@chakra-ui/icons**: Icon components
- **tailwindcss**: Utility-first CSS framework
- **framer-motion**: Animation library
- **@emotion/react**: CSS-in-JS styling
- **react-markdown**: Markdown rendering
- **react-syntax-highlighter**: Code highlighting
- **axios**: HTTP client for API requests

### Development Dependencies
- **nodemon**: Development server auto-restart
- **concurrently**: Multiple process management
- **jest**: Testing framework
- **@typescript-eslint**: TypeScript linting
- **cross-env**: Cross-platform environment variables
- **react-scripts**: Create React App build system

### Python Dependencies (requirements.txt)
- **pdfplumber**: Advanced PDF processing
- **Pillow**: Image processing
- **Wand**: ImageMagick binding
- **cryptography**: Security utilities
- **requests**: HTTP client for MCP
- **sseclient-py**: Server-sent events client

For a complete list of dependencies, refer to the `package.json` files in the project root and client directory, as well as the Python `requirements.txt` files.

---

## Technology Stack Summary

This Product Demo application represents a modern, comprehensive AI-powered platform that combines:

### **Core Architecture**
- **Full-Stack TypeScript/JavaScript**: Unified language across frontend and backend
- **Microservices Design**: Modular services for different functionalities
- **Real-time Communication**: WebSocket integration for live updates
- **Multi-language Integration**: Node.js + Python hybrid for specialized processing

### **Key Capabilities**
- **AI-Powered Chat**: LangChain integration with local and external AI models
- **Advanced Document Processing**: Python-enhanced PDF extraction with table support
- **RAG Implementation**: Vector database integration for enhanced AI responses
- **MCP Protocol**: Custom protocol for AI model context management
- **Remote System Management**: SSH-based server management and tool execution

### **Modern UI/UX**
- **React 18**: Latest React features with TypeScript
- **Chakra UI + Tailwind**: Comprehensive component library with utility-first CSS
- **Real-time Updates**: WebSocket-based live UI updates
- **Responsive Design**: Mobile-first responsive interface

### **Enterprise Features**
- **PostgreSQL Database**: Robust relational database with migration system
- **Security**: Comprehensive authentication, encryption, and validation
- **Scalability**: Connection pooling, process management, and containerization
- **Monitoring**: Real-time status tracking and structured logging

This technology stack is designed to handle complex AI workflows, document processing, and provide a rich user experience through modern web technologies and real-time communication features.

---

This document should be updated as the technology stack evolves. Last updated: December 2024.

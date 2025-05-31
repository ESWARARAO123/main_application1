# Chatbot Component Structure

This document outlines the new modular component structure for the Chatbot feature. The goal is to break down the large monolithic Chatbot.tsx file into smaller, reusable components.

## Directory Structure

```
client/src/
├── components/
│   ├── chat/
│   │   ├── ChatMessage.tsx       # Individual message rendering with markdown/code support
│   │   ├── ChatInput.tsx         # Message input and submission component
│   │   ├── MessageList.tsx       # Display for all messages with grouping
│   │   ├── ModelSelector.tsx     # AI model selection dropdown
│   │   ├── ChatSidebar.tsx       # Session management sidebar
│   │   └── ChatHeader.tsx        # Header with title editing and actions (future)
├── pages/
│   └── Chatbot.tsx               # Main chatbot page (simplified)
├── services/
│   ├── chatbotService.ts         # Session CRUD operations
│   ├── aiChatService.ts          # Ollama AI integration
│   └── ollamaService.ts          # Ollama settings and configuration
```

## Component Responsibilities

### ChatMessage
- Render a single chat message
- Support both user and AI message styling
- Render markdown content for AI messages
- Syntax highlighting for code blocks
- Display message timestamp

### ChatInput
- Handle user input with a text field
- Process message submission
- Support keyboard shortcuts (Enter to send)
- Show loading state during AI response
- Maintain focus for smooth UX

### MessageList
- Display and group messages by sender
- Infinite scrolling for message history
- Auto-scroll to new messages
- Show loading indicators during AI response
- Handle empty state with a welcome message

### ModelSelector
- Fetch and display available AI models
- Allow selecting different models
- Show model tags/capabilities (code, general, etc.)
- Remember user's last selected model
- Display model-specific information

### ChatSidebar
- List chat sessions grouped by date
- Create/select/delete chat sessions
- Show session timestamps
- Expand/collapse session groups
- Mobile-responsive design

### Chatbot Page (Simplified)
- Orchestrate component interactions
- Manage global state (active session, messages, etc.)
- Handle API calls through service layer
- Coordinate UI layout and responsiveness

## State Management

- **Local Component State**: Each component manages its internal UI state
- **Parent-Child Props**: Pass data and callbacks between components
- **localStorage**: Remember user preferences (selected model, sidebar state)
- **API Services**: Handle all backend communication through dedicated services

## API Integration

The implementation will use dedicated service modules:

1. **chatbotService.ts**: Handle session management (create, read, update, delete)
2. **aiChatService.ts**: Interact with the Ollama AI API for chat completions
3. **ollamaService.ts**: Configure and manage Ollama settings and models

## Future Enhancements

- Streaming AI responses for better UX
- Message reactions and feedback
- File attachments in conversations
- Session search and filtering
- Conversation export/import
- User feedback mechanisms for responses
- Voice input/output capabilities 
# Tool Context AI Agent Implementation Tracker

## Overview

We're implementing a context-reading AI agent that can:
1. Access tool context information when needed during conversations
2. Read context when the user asks it to or when the AI determines it needs to
3. Call appropriate tools based on the context it has read
4. Store and retrieve context information from a persistent storage system

## Context Storage Approach

### Requirements for Context Storage
- Context should be user-independent (shared across all users)
- Tool documentation and usage patterns should be centrally managed
- Context should be easily updatable without changing system prompts
- Context should be structured and queryable

### Storage Options Analysis

#### 1. Database Storage (Recommended)
- **Pros**:
  - Structured storage with query capabilities
  - Separation from user data (different tables/collections)
  - Centralized management
  - Scalable as tool collection grows
- **Cons**:
  - Requires database schema design
  - Additional database operations

#### 2. File-based Storage
- **Pros**:
  - Simple implementation
  - Easy to edit manually
  - No database dependencies
- **Cons**:
  - Limited query capabilities
  - Harder to scale
  - No built-in validation

#### 3. Combined Approach
- Store basic context in database
- Keep detailed documentation in markdown files
- Reference files from database entries

### Recommended Approach
Use a dedicated database table/collection for tool context that is separate from user data:

```
Table: tool_contexts
- id: unique identifier
- tool_name: name of the tool
- description: general description
- parameters: JSON structure of parameters
- examples: JSON array of usage examples
- category: tool category for organization
- updated_at: timestamp for tracking updates
```

## User-Specific Rules

In addition to the shared tool context, implement user-specific rules:
- Create an "AI Rules" section in settings
- Allow users to define custom rules for their AI interactions
- Store these rules in the user's profile
- Include these rules when reading context

```
Table: user_ai_rules
- id: unique identifier
- user_id: reference to user
- rule_type: type of rule (e.g., "tool_usage", "conversation_style")
- rule_content: the actual rule content
- priority: order of application
- created_at: creation timestamp
- updated_at: last update timestamp
```

## Implementation Steps

### 1. Database Schema Setup
- [x] Create `user_ai_rules` table/collection
- [ ] Create `tool_contexts` table/collection
- [x] Create database access methods

### 2. Context Management API
- [x] Implement API endpoints for managing user AI rules
- [ ] Implement API endpoints for managing tool contexts
- [ ] Create admin interface for managing tool contexts

### 3. Context Reading Tool
- [x] Implement `read_context` tool in backend
- [ ] Create different query types (by tool name, category, search)
- [ ] Implement context merging (shared + user-specific)

### 4. AI Integration
- [x] Update system prompt to include instructions for using `read_context`
- [ ] Implement context detection in AI responses
- [ ] Create middleware to process context requests

### 5. Frontend Integration
- [x] Create "AI Rules" settings component
- [x] Implement UI for managing user-specific rules
- [ ] Update chat interface to handle context-related messages

### 6. Initial Context Population
- [ ] Document all available tools
- [ ] Create usage patterns and examples
- [ ] Populate the database with initial context data

## Current Progress (May 22, 2024)

### Completed
1. **Database Schema**:
   - Created `user_ai_rules` table with SQL migration
   - Added appropriate indexes and constraints

2. **Backend API**:
   - Implemented CRUD endpoints for user AI rules
   - Created context reading endpoint

3. **Frontend Components**:
   - Created AI Rules settings component
   - Implemented theme-aware styling
   - Added rule saving functionality
   - Created example AI rules for testing

4. **Context Agent Implementation**:
   - Created context agent system prompt (contextAgentPrompt.ts)
   - Implemented read_context tool functionality
   - Added client-side fallback for context reading
   - Created tool handler for AI chat flow (aiToolHandler.ts)

### In Progress
1. **AI Integration**:
   - Created system prompt enhancement utility
   - Implemented read_context tool execution
   - Added context-aware prompt generation

### Next Steps: Context Agent Testing and Refinement

Now that we've implemented the core context agent functionality, we need to test and refine it:

1. **Testing Context Agent**:
   - Test AI rules saving and retrieval
   - Verify context reading functionality with visual feedback
   - Test integration with AI chat flow
   - Ensure the context agent works in MCP mode

### Implementation Details (May 22, 2024)

We've implemented the following components for the context agent:

1. **System Prompt Enhancement**:
   - Created `mcpSystemPrompt.ts` with explicit instructions for using the read_context tool
   - Added instructions to check context at the beginning of every conversation
   - Provided examples of how to use the tool and interpret results

2. **Tool Execution Framework**:
   - Created `toolParser.ts` to extract tool calls from AI responses
   - Implemented `useToolExecution` hook to handle tool execution with visual feedback
   - Added tool execution to the MCP chat flow

3. **Visual Feedback**:
   - Created `ContextReadingIndicator` component to show when context is being read
   - Added the indicator to the chat UI
   - Implemented callback mechanism to update UI during tool execution

4. **MCP Integration**:
   - Enhanced MCP chat flow to include context agent system prompt
   - Added tool call detection and processing
   - Implemented fallback mechanisms for context reading

## How the Context Agent Works

The context agent is an integrated feature of the MCP mode that makes the AI invoke a tool to read user context and feed it back. Here's how it works:

1. **System Prompt Enhancement**:
   - The base system prompt is enhanced with context agent instructions
   - These instructions tell the AI when and how to use the read_context tool
   - The AI is instructed to check context for user preferences

2. **Context Reading Flow**:
   - When the AI determines it needs context (based on system prompt instructions)
   - It invokes the read_context tool
   - The tool retrieves the user's saved context rules from the database
   - The AI incorporates these rules into its response

3. **Tool Execution**:
   - The read_context tool is handled by the aiToolHandler
   - It calls the executeReadContextTool function from contextAgentService
   - This function retrieves the user's context using the aiContextService
   - The context is returned to the AI for processing

4. **Fallback Mechanism**:
   - If the backend context agent is not available
   - The system falls back to client-side implementation
   - This ensures the context agent works even without backend support

5. **MCP Mode Integration**:
   - The context agent is activated when MCP mode is enabled
   - It works alongside other MCP tools
   - It provides context-aware responses based on user preferences

2. **AI Integration**:
   - Modify the AI system prompt to include the retrieved context
   - Implement context-aware response generation
   - Add instructions for the AI to respect user preferences

3. **MCP Mode Integration**:
   - Update the MCP toggle to activate the context agent
   - Add context agent status indicators in the UI
   - Implement context refresh mechanism

4. **Context Utilization**:
   - Create a mechanism for the AI to reference specific parts of the context
   - Implement priority handling for conflicting context rules
   - Add context validation to ensure proper formatting

5. **Testing and Refinement**:
   - Test context retrieval with various user rules
   - Verify that the AI respects the context in its responses
   - Refine the context processing based on user feedback

## Context Reading Flow

1. **Automatic Context Reading**:
   ```
   User: "Can you help me list all files in my project?"

   AI: (thinks) "I need to list files, let me check available tools"

   AI: (uses read_context) "read_context(contextType: 'search', query: 'list files')"

   System: (returns relevant tools and usage patterns)

   AI: "I can help you list files. I'll use the readDirectory tool:

   <command>
   tool: readDirectory
   parameters: {
     "dirPath": "."
   }
   description: This will list all files in the current directory.
   </command>

   Would you like me to run this command?"
   ```

2. **User-Requested Context Reading**:
   ```
   User: "What tools do you have for working with files?"

   AI: (uses read_context) "read_context(contextType: 'category', contextName: 'file_operations')"

   System: (returns file operation tools)

   AI: "I have several tools for working with files:

   1. readDirectory - Lists files in a directory
   2. readFile - Reads the content of a file
   3. createFile - Creates a new file with specified content
   4. editFile - Modifies an existing file
   5. deleteFile - Removes a file

   Would you like more details about any of these tools?"
   ```

## Database Schema Details

### tool_contexts Table

```sql
CREATE TABLE tool_contexts (
    id SERIAL PRIMARY KEY,
    tool_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    parameters JSONB NOT NULL,
    examples JSONB NOT NULL,
    category VARCHAR(100) NOT NULL,
    usage_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### user_ai_rules Table

```sql
CREATE TABLE user_ai_rules (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    rule_type VARCHAR(50) NOT NULL,
    rule_content TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## API Endpoints

### Tool Context Management

- `GET /api/context/tools` - List all tool contexts
- `GET /api/context/tools/:toolName` - Get specific tool context
- `GET /api/context/categories/:category` - Get tools by category
- `GET /api/context/search?q=query` - Search tool contexts
- `POST /api/context/tools` - Create new tool context (admin only)
- `PUT /api/context/tools/:toolName` - Update tool context (admin only)
- `DELETE /api/context/tools/:toolName` - Delete tool context (admin only)

### User AI Rules Management

- `GET /api/users/:userId/ai-rules` - Get user's AI rules
- `POST /api/users/:userId/ai-rules` - Create new AI rule
- `PUT /api/users/:userId/ai-rules/:ruleId` - Update AI rule
- `DELETE /api/users/:userId/ai-rules/:ruleId` - Delete AI rule

## Next Steps

1. Create database schema for tool contexts and user AI rules
2. Implement backend API for context management
3. Create the `read_context` tool implementation
4. Update AI system prompt to include context reading instructions
5. Implement frontend components for AI rules management
6. Populate initial tool context data

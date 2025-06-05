# AI Rules Implementation Plan

## Overview

The AI Rules component will allow users to define custom rules for how the AI agent behaves, including how it accesses and uses tool context. This feature will be integrated into the settings section of the application.

## User Interface Design

### AI Rules Settings Page

Create a new section in the user settings called "AI Rules" with the following components:

1. **Rule Categories**:
   - Tool Usage Rules
   - Conversation Style Rules
   - Context Access Rules
   - Command Approval Rules

2. **Rule Management Interface**:
   - List of existing rules with edit/delete options
   - "Add New Rule" button
   - Rule priority adjustment (drag-and-drop reordering)
   - Rule activation toggle

3. **Rule Creation Form**:
   - Rule type selector
   - Rule content editor (with templates based on type)
   - Priority setting
   - Save/Cancel buttons

### Rule Types and Templates

#### 1. Tool Usage Rules
Rules that define how and when specific tools should be used.

**Template Example**:
```json
{
  "rule_type": "tool_usage",
  "tool_name": "readDirectory",
  "condition": "User asks about listing files",
  "action": "Suggest using this tool with default parameters",
  "priority": 5
}
```

#### 2. Conversation Style Rules
Rules that define the AI's conversation style and response format.

**Template Example**:
```json
{
  "rule_type": "conversation_style",
  "context": "When discussing technical topics",
  "style": "Detailed and technical with code examples",
  "priority": 3
}
```

#### 3. Context Access Rules
Rules that define when and how the AI should access context information.

**Template Example**:
```json
{
  "rule_type": "context_access",
  "trigger": "When user mentions file operations",
  "action": "Read file operation tool contexts",
  "priority": 4
}
```

#### 4. Command Approval Rules
Rules that define when commands should require explicit approval.

**Template Example**:
```json
{
  "rule_type": "command_approval",
  "tool_category": "file_modification",
  "requirement": "Always require explicit approval",
  "priority": 10
}
```

## Backend Implementation

### Database Schema

Extend the previously defined `user_ai_rules` table with additional fields:

```sql
CREATE TABLE user_ai_rules (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    rule_type VARCHAR(50) NOT NULL,
    rule_name VARCHAR(100) NOT NULL,
    rule_content JSONB NOT NULL,
    priority INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### API Endpoints

#### User AI Rules Management

1. **List Rules**
   - `GET /api/users/:userId/ai-rules`
   - Returns all AI rules for a specific user
   - Sorted by priority

2. **Get Rule**
   - `GET /api/users/:userId/ai-rules/:ruleId`
   - Returns a specific rule by ID

3. **Create Rule**
   - `POST /api/users/:userId/ai-rules`
   - Creates a new AI rule
   - Validates rule structure based on type

4. **Update Rule**
   - `PUT /api/users/:userId/ai-rules/:ruleId`
   - Updates an existing rule
   - Validates rule structure based on type

5. **Delete Rule**
   - `DELETE /api/users/:userId/ai-rules/:ruleId`
   - Removes a rule

6. **Reorder Rules**
   - `PUT /api/users/:userId/ai-rules/reorder`
   - Updates priorities of multiple rules at once

7. **Toggle Rule**
   - `PATCH /api/users/:userId/ai-rules/:ruleId/toggle`
   - Activates or deactivates a rule

### Rule Validation Service

Create a service to validate rules based on their type:

```javascript
// Example rule validation service
const validateRule = (rule) => {
  const { rule_type, rule_content } = rule;
  
  switch (rule_type) {
    case 'tool_usage':
      return validateToolUsageRule(rule_content);
    case 'conversation_style':
      return validateConversationStyleRule(rule_content);
    case 'context_access':
      return validateContextAccessRule(rule_content);
    case 'command_approval':
      return validateCommandApprovalRule(rule_content);
    default:
      return { valid: false, error: 'Unknown rule type' };
  }
};
```

## AI Integration

### Context Reading with User Rules

When the AI uses the `read_context` tool, include user-specific rules in the context:

```javascript
// Example context reading with user rules
app.post('/api/mcp/agent/read-context', async (req, res) => {
  try {
    const { contextType, contextName, query, userId } = req.body;
    
    // Get base context
    let contextData = await getContextData(contextType, contextName, query);
    
    // If userId is provided, get and apply user rules
    if (userId) {
      const userRules = await getUserAIRules(userId);
      
      // Filter relevant rules
      const relevantRules = userRules.filter(rule => 
        isRuleRelevantToContext(rule, contextType, contextName, query)
      );
      
      // Apply rules to context
      contextData = applyRulesToContext(contextData, relevantRules);
    }
    
    return res.json({ context: contextData });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

### System Prompt Enhancement

Update the AI system prompt to include awareness of user-specific rules:

```javascript
const systemPrompt = `
You are an AI assistant with access to a context memory system.
When you need information about available tools or how to use them, you can use the read_context tool.

The user has defined specific AI rules that you should follow. These rules will be included
when you read context using the read_context tool.

Always check the context memory before suggesting commands to ensure you're using the correct syntax and parameters.
Also, be aware of any user-specific rules that may apply to the current conversation.
`;
```

## Frontend Implementation

### React Components

#### 1. AI Rules Settings Page

```jsx
// AIRulesSettings.tsx
import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { fetchUserRules, createRule, updateRule, deleteRule, reorderRules } from '../services/aiRulesService';

const AIRulesSettings = () => {
  const { currentUser } = useAuth();
  const [rules, setRules] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const loadRules = async () => {
      if (currentUser) {
        setLoading(true);
        const userRules = await fetchUserRules(currentUser.id);
        setRules(userRules);
        setLoading(false);
      }
    };
    
    loadRules();
  }, [currentUser]);
  
  // Rule management functions
  const handleAddRule = async (newRule) => {
    const createdRule = await createRule(currentUser.id, newRule);
    setRules([...rules, createdRule]);
  };
  
  const handleUpdateRule = async (ruleId, updatedRule) => {
    const updated = await updateRule(currentUser.id, ruleId, updatedRule);
    setRules(rules.map(rule => rule.id === ruleId ? updated : rule));
  };
  
  const handleDeleteRule = async (ruleId) => {
    await deleteRule(currentUser.id, ruleId);
    setRules(rules.filter(rule => rule.id !== ruleId));
  };
  
  const handleReorderRules = async (reorderedRules) => {
    setRules(reorderedRules);
    await reorderRules(currentUser.id, reorderedRules.map((rule, index) => ({
      id: rule.id,
      priority: index
    })));
  };
  
  return (
    <div className="ai-rules-settings">
      <h2>AI Rules</h2>
      
      {loading ? (
        <div>Loading rules...</div>
      ) : (
        <>
          <div className="rule-categories">
            {/* Rule category tabs */}
          </div>
          
          <div className="rules-list">
            {/* List of rules with edit/delete options */}
          </div>
          
          <button onClick={() => handleAddRule()}>Add New Rule</button>
        </>
      )}
    </div>
  );
};

export default AIRulesSettings;
```

#### 2. Rule Editor Component

```jsx
// RuleEditor.tsx
import React, { useState } from 'react';

const RuleEditor = ({ rule, onSave, onCancel }) => {
  const [ruleType, setRuleType] = useState(rule?.rule_type || 'tool_usage');
  const [ruleName, setRuleName] = useState(rule?.rule_name || '');
  const [ruleContent, setRuleContent] = useState(rule?.rule_content || {});
  const [priority, setPriority] = useState(rule?.priority || 0);
  
  // Get template based on rule type
  const getTemplate = (type) => {
    switch (type) {
      case 'tool_usage':
        return {
          tool_name: '',
          condition: '',
          action: ''
        };
      case 'conversation_style':
        return {
          context: '',
          style: ''
        };
      case 'context_access':
        return {
          trigger: '',
          action: ''
        };
      case 'command_approval':
        return {
          tool_category: '',
          requirement: ''
        };
      default:
        return {};
    }
  };
  
  // Handle rule type change
  const handleTypeChange = (type) => {
    setRuleType(type);
    setRuleContent(getTemplate(type));
  };
  
  // Handle save
  const handleSave = () => {
    onSave({
      rule_type: ruleType,
      rule_name: ruleName,
      rule_content: ruleContent,
      priority
    });
  };
  
  return (
    <div className="rule-editor">
      <h3>{rule ? 'Edit Rule' : 'Create New Rule'}</h3>
      
      <div className="form-group">
        <label>Rule Name</label>
        <input 
          type="text" 
          value={ruleName} 
          onChange={(e) => setRuleName(e.target.value)} 
        />
      </div>
      
      <div className="form-group">
        <label>Rule Type</label>
        <select 
          value={ruleType} 
          onChange={(e) => handleTypeChange(e.target.value)}
        >
          <option value="tool_usage">Tool Usage</option>
          <option value="conversation_style">Conversation Style</option>
          <option value="context_access">Context Access</option>
          <option value="command_approval">Command Approval</option>
        </select>
      </div>
      
      {/* Dynamic form fields based on rule type */}
      {ruleType === 'tool_usage' && (
        <>
          <div className="form-group">
            <label>Tool Name</label>
            <input 
              type="text" 
              value={ruleContent.tool_name || ''} 
              onChange={(e) => setRuleContent({...ruleContent, tool_name: e.target.value})} 
            />
          </div>
          {/* Additional fields for tool_usage */}
        </>
      )}
      
      {/* Similar sections for other rule types */}
      
      <div className="form-group">
        <label>Priority</label>
        <input 
          type="number" 
          value={priority} 
          onChange={(e) => setPriority(parseInt(e.target.value))} 
        />
      </div>
      
      <div className="form-actions">
        <button onClick={handleSave}>Save</button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
};

export default RuleEditor;
```

## Implementation Steps

1. **Database Setup**:
   - Create the `user_ai_rules` table
   - Set up indexes for efficient queries

2. **Backend API**:
   - Implement CRUD endpoints for AI rules
   - Create rule validation service
   - Integrate with context reading system

3. **Frontend Components**:
   - Create AI Rules settings page
   - Implement rule editor component
   - Add rule list with management options

4. **AI Integration**:
   - Update system prompt
   - Modify context reading to include user rules
   - Implement rule application logic

5. **Testing**:
   - Test rule creation and management
   - Verify rule application in AI conversations
   - Test integration with context reading

## Next Steps

1. Design database schema for user AI rules
2. Implement backend API endpoints for rule management
3. Create frontend components for the AI Rules settings page
4. Integrate rule application with the context reading system
5. Test the complete workflow with various rule types

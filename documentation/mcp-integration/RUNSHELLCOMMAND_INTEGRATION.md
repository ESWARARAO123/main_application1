# RunShellCommand Tool Integration

## Overview

The `runshellcommand` tool enables AI assistants to execute shell commands through your existing MCP (Model Context Protocol) infrastructure. This integration allows users to interact with their systems via natural language, with the AI translating requests into appropriate shell commands and executing them securely through the MCP orchestrator.

## Architecture

### Components

1. **Backend Service** (`src/services/shellCommandService.js`)
   - Manages shell command execution via Python MCP orchestrator
   - Integrates with existing MCP server configurations from database
   - Uses configured Python virtual environment

2. **API Routes** (`src/routes/ai.js`)
   - `POST /api/ai/tools/runshellcommand` - Execute shell commands
   - `GET /api/ai/tools/runshellcommand/test` - Test MCP connection
   - `GET /api/ai/tools/runshellcommand/tools` - Get available MCP tools
   - `GET /api/ai/tools/runshellcommand/servers` - Get user's MCP servers

3. **Frontend Integration**
   - `client/src/services/aiToolsService.ts` - AI tool definition
   - `client/src/services/aiToolHandler.ts` - Tool execution handler
   - `client/src/services/shellCommandService.ts` - Frontend service

### Data Flow

```
User Request → AI Assistant → runshellcommand Tool → Backend API → 
Python Orchestrator → MCP Server → Shell Execution → Response Chain
```

## Features

### ✅ **Dynamic MCP Server Management**
- Uses user-configured MCP servers from database
- No hardcoded server configurations
- Supports multiple MCP servers per user
- Automatic default server selection

### ✅ **Secure Execution**
- User authentication required
- Server configurations isolated per user
- Configurable timeouts
- Error handling and logging

### ✅ **Context Preservation**
- Command history maintained in chat
- Results formatted for AI understanding
- Server information included in responses
- Timestamps for execution tracking

### ✅ **Python Environment Integration**
- Uses configured Python virtual environment
- Leverages existing Python orchestrator
- No additional dependencies required

## Setup and Configuration

### Prerequisites

1. **MCP Server Running**
   ```bash
   # Ensure your MCP server is running and accessible
   # Example: http://172.16.16.54:8080
   ```

2. **MCP Server Configuration**
   - Configure MCP server details in the application UI
   - Go to MCP settings and add your server
   - Set one server as default for your user

3. **Python Virtual Environment**
   ```bash
   # Ensure Python venv is configured in conf/config.ini
   [python]
   interpreter = python\.venv\Scripts\python.exe
   ```

### Testing the Integration

Run the test script to verify everything is working:

```bash
node test_shell_command_integration.js
```

## Usage Examples

### Chat Interface

Users can interact with the AI using natural language:

**User:** "I want to see the list of folders available in this folder"

**AI Response:** The AI will:
1. Recognize this as a shell command request
2. Generate appropriate command: `ls -la`
3. Execute via MCP orchestrator
4. Return formatted results with context

**User:** "What's the current disk usage?"

**AI Response:** Executes `df -h` and provides formatted output.

### Direct API Usage

```javascript
// Execute a shell command
const result = await fetch('/api/ai/tools/runshellcommand', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    command: 'ls -la',
    timeout: 30
  })
});

const response = await result.json();
console.log(response.result);
```

### Frontend Service Usage

```typescript
import { shellCommandService } from './services/shellCommandService';

// Execute command
const result = await shellCommandService.executeCommand('pwd');

// Test connection
const status = await shellCommandService.testConnection();

// Get user's MCP servers
const servers = await shellCommandService.getUserMCPServers();
```

## Command Examples

The AI can handle various types of requests:

| User Request | Generated Command | Description |
|-------------|------------------|-------------|
| "List files in current directory" | `ls -la` | Show detailed file listing |
| "Show current directory" | `pwd` | Print working directory |
| "Check disk space" | `df -h` | Display disk usage |
| "Show memory usage" | `free -h` | Display memory statistics |
| "List running processes" | `ps aux` | Show active processes |
| "Check system info" | `uname -a` | Display system information |
| "Show current time" | `date` | Display current date/time |
| "Who is logged in" | `who` | Show logged in users |

## Response Format

### Successful Execution
```json
{
  "success": true,
  "command": "ls -la",
  "result": {
    "output": "total 24\ndrwxr-xr-x 6 user user 4096 Dec 10 10:30 .\n..."
  },
  "serverConfig": {
    "id": "server-123",
    "name": "Production Server",
    "host": "172.16.16.54",
    "port": "8080"
  },
  "timestamp": "2024-12-10T10:30:00.000Z"
}
```

### Failed Execution
```json
{
  "success": false,
  "command": "invalid-command",
  "error": "Command not found",
  "stderr": "bash: invalid-command: command not found",
  "serverConfig": {...},
  "timestamp": "2024-12-10T10:30:00.000Z"
}
```

## Security Considerations

### ✅ **Authentication & Authorization**
- All API endpoints require user authentication
- Server configurations are user-isolated
- No cross-user data access

### ✅ **Command Execution Safety**
- Commands executed through MCP server (not directly on application server)
- Configurable timeouts prevent hanging processes
- Error handling prevents system exploitation

### ✅ **Input Validation**
- Command parameters validated
- Server ID validation
- Timeout bounds checking

### ⚠️ **Security Notes**
- Users can execute any command their MCP server allows
- Consider implementing command whitelisting for production
- Monitor command execution logs
- Ensure MCP server has appropriate user permissions

## Error Handling

### Common Error Scenarios

1. **No MCP Server Configured**
   ```
   Error: No default MCP server configured for user. 
   Please configure an MCP server first.
   ```

2. **MCP Server Unreachable**
   ```
   Error: Failed to connect to MCP server at http://host:port
   ```

3. **Command Execution Timeout**
   ```
   Error: Command execution timed out after 30 seconds
   ```

4. **Python Environment Issues**
   ```
   Error: Python interpreter not found or MCP orchestrator missing
   ```

## Troubleshooting

### 1. Check MCP Server Status
```bash
# Test MCP server connectivity
curl http://your-mcp-server:8080/info
```

### 2. Verify Python Environment
```bash
# Check if Python venv is working
python\.venv\Scripts\python.exe --version
```

### 3. Test MCP Orchestrator
```bash
# Test orchestrator directly
python python/terminal-mcp-orchestrator/orchestrator.py --server http://host:port --list
```

### 4. Check Application Logs
```bash
# Monitor application logs for errors
tail -f logs/app.log
```

## Development

### Adding New Command Patterns

To add new command suggestions, modify the `generateCommandSuggestion` method in `shellCommandService.ts`:

```typescript
if (request.includes('your-pattern')) {
  return 'your-command';
}
```

### Extending API Endpoints

Add new endpoints in `src/routes/ai.js`:

```javascript
router.post('/tools/runshellcommand/new-feature', requireAuth, async (req, res) => {
  // Implementation
});
```

### Custom Result Formatting

Modify the `formatResultForDisplay` method in `shellCommandService.ts` to customize how results are displayed to users.

## Integration with Existing Systems

### MCP Server Management
- Leverages existing MCP server configuration UI
- Uses existing database tables: `user_mcp_server_configurations`
- Integrates with existing MCP connection management

### Authentication System
- Uses existing session-based authentication
- Respects existing user permissions
- Integrates with existing user management

### Logging & Monitoring
- Uses existing logger configuration
- Follows existing error handling patterns
- Integrates with existing monitoring systems

## Future Enhancements

### Planned Features
- [ ] Command history and favorites
- [ ] Command templates and snippets
- [ ] Real-time command output streaming
- [ ] Multi-server command execution
- [ ] Command scheduling and automation
- [ ] Enhanced security with command whitelisting
- [ ] Interactive command sessions

### Possible Integrations
- File upload/download capabilities
- Integration with CI/CD pipelines
- Database query capabilities
- System monitoring dashboards
- Automated deployment tools

## Support

For issues or questions:

1. Check the application logs
2. Verify MCP server connectivity
3. Test Python environment setup
4. Review user MCP server configurations
5. Check network connectivity between components

## License

This integration follows the same license as the main application. 
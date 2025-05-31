# Model Context Protocol (MCP) Server Guide

## Introduction

The Model Context Protocol (MCP) is a standardized communication protocol that enables AI models and applications to interact with external systems through a set of predefined tools. This document explains how the MCP server works, the connection flow, available tools, and how to use them effectively.

## How MCP Works

### Core Concepts

1. **Server-Sent Events (SSE)**: MCP uses SSE for establishing persistent connections between clients and the server, enabling real-time updates.

2. **Client ID**: Each client must first establish an SSE connection to receive a unique `clientId`. This ID is required for all subsequent tool invocations.

3. **Tools**: The MCP server provides various tools like file operations, shell commands, and directory management. Each tool accepts specific parameters and returns structured responses.

4. **Stateful Sessions**: The MCP server maintains session state for each connected client, allowing for context-aware interactions.

## Connection Flow

The MCP connection flow follows these steps:

1. **Establish SSE Connection**: Client connects to the `/sse` endpoint
2. **Receive Client ID**: Server responds with a unique client ID
3. **Invoke Tools**: Use the client ID to execute various tools via the `/messages` endpoint
4. **Receive Results**: Results come back as structured responses, and real-time updates are sent via the SSE connection

### Example Connection Sequence

```
Client → Server: GET /sse
Server → Client: data: {"type":"connected","clientId":"1747810889393"}
Client → Server: POST /messages {"id":"msg-1","type":"invoke_tool","content":{"name":"readDirectory","parameters":{}},"clientId":"1747810889393"}
Server → Client: {"id":"msg-1","type":"tool_result","content":{...}}
```

## Available MCP Tools

Based on my experience with your MCP server at 172.16.16.54:8080, these tools are available:

### File Operations

1. **readFile**: Read the content of a file
   - Parameters: `filePath` (required), `encoding` (optional), `startLine` (optional), `endLine` (optional)

2. **createFile**: Create a new file with specified contents
   - Parameters: `filePath` (required), `content` (required)

3. **editFile**: Edit the content of an existing file
   - Parameters: `filePath` (required), `operation` (required: "append", "prepend", "replace", "insert"), `content` (required), `lineNumber` (optional), `startLine` (optional), `endLine` (optional), `encoding` (optional)

4. **deleteFile**: Delete a file from the filesystem
   - Parameters: `filePath` (required)

5. **copyFile**: Copy a file from one location to another
   - Parameters: `sourcePath` (required), `destinationPath` (required)

6. **moveFile**: Move a file from one location to another
   - Parameters: `sourcePath` (required), `destinationPath` (required)

### Directory Operations

7. **readDirectory**: List files and folders in a directory
   - Parameters: `dirPath` (optional, defaults to workspace root)

8. **createDirectory**: Create a new directory
   - Parameters: `dirPath` (required), `recursive` (optional, default: true)

9. **moveDirectory**: Move a directory from one location to another
   - Parameters: `sourcePath` (required), `destinationPath` (required)

10. **copyDirectory**: Copy a directory from one location to another
    - Parameters: `sourcePath` (required), `destinationPath` (required), `overwrite` (optional), `errorOnExist` (optional)

11. **deleteDirectory**: Delete a directory and its contents
    - Parameters: `dirPath` (required), `recursive` (optional, default: true)

12. **getDirectoryTree**: Get a hierarchical representation of a directory
    - Parameters: `dirPath` (required), `maxDepth` (optional), `includeFiles` (optional), `includeDirs` (optional), `includeSize` (optional), `extensions` (optional), `exclude` (optional)

### Command Execution

13. **runShellCommand**: Run a terminal command in the system shell
    - Parameters: `command` (required)

14. **runPythonFile**: Execute a Python file and return output
    - Parameters: `filePath` (required), `args` (optional)

### Search Operations

15. **grep**: Search for patterns in files
    - Parameters: `pattern` (required), `filePaths` (required), `useRegex` (optional), `caseSensitive` (optional), `beforeContext` (optional), `afterContext` (optional), `maxMatches` (optional), `encoding` (optional)

### Advanced Operations

16. **combinationTask**: Run a sequence of operations with a common working directory
    - Parameters: `workingDir` (required), `tasks` (required), `stopOnError` (optional)

## How to Use MCP

### Using cURL

cURL is a simple way to test and interact with the MCP server from the command line.

#### 1. Establish SSE Connection

```bash
curl -N http://172.16.16.54:8080/sse
```

This returns a client ID:
```
data: {"type":"connected","clientId":"1747810889393"}
```

#### 2. Invoke a Tool

```bash
curl -X POST http://172.16.16.54:8080/messages \
  -H "Content-Type: application/json" \
  -d '{"id":"msg-1","type":"invoke_tool","content":{"name":"readDirectory","parameters":{}},"clientId":"1747810889393"}'
```

### Using PowerShell

PowerShell provides more robust handling for Windows environments.

#### Create a PowerShell MCP Client

```powershell
# First get a client ID by connecting to the SSE endpoint
$clientId = $null
$request = [System.Net.WebRequest]::Create("http://172.16.16.54:8080/sse")
$request.Method = "GET"

try {
    $response = $request.GetResponse()
    $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
    
    # Get the first line which should contain our client ID
    $line = $reader.ReadLine()
    if ($line -match 'data: (.+)') {
        $data = $matches[1] | ConvertFrom-Json
        $clientId = $data.clientId
        Write-Host "Connected with client ID: $clientId"
    }
} catch {
    Write-Host "Error connecting to SSE: $_"
    exit
}

# Invoke a tool
if ($clientId) {
    $messageId = "msg-" + (Get-Random)
    $body = @{
        id = $messageId
        type = "invoke_tool"
        content = @{
            name = "readDirectory"
            parameters = @{}
        }
        clientId = $clientId
    } | ConvertTo-Json -Depth 10

    $headers = @{
        "Content-Type" = "application/json"
    }

    Write-Host "Invoking tool with message ID: $messageId"
    $result = Invoke-RestMethod -Uri "http://172.16.16.54:8080/messages" -Method Post -Headers $headers -Body $body
    $result | ConvertTo-Json -Depth 10
}
```

### Using Python

Python offers the most robust and feature-complete way to interact with the MCP server.

#### Create a Python MCP Client

```python
import json
import requests
import sseclient
import threading
import time

class MCPClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.client_id = None
        self.sse_thread = None
        self.running = False
        self.connected = False
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def connect(self):
        """Establish SSE connection and get client ID"""
        self.running = True
        self.sse_thread = threading.Thread(target=self._sse_listener)
        self.sse_thread.daemon = True
        self.sse_thread.start()
        
        # Wait for connection to be established
        timeout = 10
        start_time = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not self.connected:
            raise TimeoutError("Failed to establish SSE connection within timeout period")
        
        return self.client_id
    
    def _sse_listener(self):
        """Background thread that listens to SSE events"""
        try:
            response = requests.get(f"{self.server_url}/sse", stream=True)
            client = sseclient.SSEClient(response)
            
            for event in client.events():
                if not self.running:
                    break
                
                if event.data:
                    try:
                        data = json.loads(event.data)
                        
                        # Handle connected event
                        if data.get("type") == "connected":
                            self.client_id = data.get("clientId")
                            self.connected = True
                            print(f"Connected with client ID: {self.client_id}")
                        
                        # Handle tool result
                        elif data.get("type") == "tool_result":
                            print(f"Received tool result: {json.dumps(data, indent=2)}")
                    
                    except json.JSONDecodeError:
                        print(f"Received non-JSON data: {event.data}")
        
        except Exception as e:
            print(f"SSE connection error: {e}")
            self.running = False
    
    def invoke_tool(self, tool_name, parameters=None):
        """Invoke an MCP tool with the given parameters"""
        if not self.client_id:
            raise ValueError("Not connected. Call connect() first")
        
        if parameters is None:
            parameters = {}
        
        message_id = f"msg-{int(time.time())}"
        payload = {
            "id": message_id,
            "type": "invoke_tool",
            "content": {
                "name": tool_name,
                "parameters": parameters
            },
            "clientId": self.client_id
        }
        
        print(f"Invoking tool: {tool_name}")
        response = requests.post(
            f"{self.server_url}/messages",
            headers=self.headers,
            json=payload
        )
        return response.json()
    
    def disconnect(self):
        """Disconnect from the SSE stream"""
        self.running = False
        if self.sse_thread:
            self.sse_thread.join(timeout=1)
        self.connected = False
        self.client_id = None

# Usage example
if __name__ == "__main__":
    server_url = "http://172.16.16.54:8080"
    client = MCPClient(server_url)
    
    try:
        print(f"Connecting to MCP server at {server_url}...")
        client.connect()
        
        # Read directory
        result = client.invoke_tool("readDirectory")
        print(json.dumps(result, indent=2))
        
        # Create a test file
        result = client.invoke_tool("createFile", {
            "filePath": "test-file.txt",
            "content": "This is a test file created via MCP."
        })
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        client.disconnect()
        print("MCP client test complete!")
```

## Important Considerations

### 1. Connection Management

- **Session Persistence**: The MCP server maintains your session state as long as the SSE connection remains open.
- **Client ID Validation**: All tool invocations must include a valid `clientId`, or they will be rejected.
- **Reconnection Logic**: Implement retry logic in production applications for dropped connections.

### 2. Error Handling

Common error scenarios include:

- **No active SSE connection**: Occurs if you try to invoke a tool without an active SSE connection or with an invalid client ID
- **Tool not found**: Happens when you try to invoke a non-existent tool
- **Invalid parameters**: When required parameters are missing or have incorrect types
- **Tool execution errors**: When the tool execution fails on the server side

Example error response:
```json
{
  "id": "msg-1",
  "type": "error",
  "error": {
    "message": "No active SSE connection",
    "code": "SESSION_NOT_FOUND"
  }
}
```

### 3. Security Considerations

- **Access Control**: The MCP server may implement authentication and authorization mechanisms
- **Rate Limiting**: Be mindful of how frequently you invoke tools to avoid overwhelming the server
- **File Access**: Only access files and directories within the allowed workspace paths

## Best Practices

1. **Maintain SSE Connection**: Keep the SSE connection alive for the duration of your session
2. **Graceful Disconnection**: Always close the SSE connection when finished
3. **Error Handling**: Implement robust error handling for all tool invocations
4. **Batching Operations**: Use the `combinationTask` tool for complex sequences of operations
5. **Progress Tracking**: For long-running operations, monitor progress through the SSE connection

## Conclusion

The Model Context Protocol (MCP) provides a powerful interface for AI applications to interact with external systems in a controlled, standardized way. By understanding the connection flow, available tools, and implementation patterns, you can effectively leverage the MCP server to perform a wide range of operations from file management to command execution.

For production applications, consider implementing more robust error handling, reconnection logic, and security measures to ensure reliable and secure interactions with the MCP server. 
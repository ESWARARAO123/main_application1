# Terminal MCP Orchestrator

A Python-based implementation of Model Context Protocol (MCP) tools for interacting with an MCP server.

## Overview

This project provides command-line tools that implement the Model Context Protocol (MCP) specification. These tools can be used to perform various operations on the file system, execute commands, and interact with the environment through an MCP server.

The architecture consists of:

1. **Orchestrator:** Routes requests to a remote MCP server
2. **MCP Client:** Connects to a remote MCP server via SSE for real-time tool execution

## Tools Implemented

The following MCP tools are available through the MCP server:

1. `runShellCommand`: Run a terminal command in the system shell
2. `runPythonFile`: Execute a Python file and return output
3. `readDirectory`: List files and folders in a directory
4. `copyFile`: Copy a file from one location to another
5. `createFile`: Create a new file with specified contents
6. `readFile`: Read the content of a file
7. `editFile`: Edit the content of an existing file
8. `deleteFile`: Delete a file from the filesystem
9. `moveFile`: Move a file from one location to another
10. `createDirectory`: Create a new directory
11. `moveDirectory`: Move a directory from one location to another
12. `copyDirectory`: Copy a directory from one location to another
13. `deleteDirectory`: Delete a directory and its contents
14. `getDirectoryTree`: Get a hierarchical representation of a directory
15. `grep`: Search for patterns in files
16. `combinationTask`: Run a sequence of operations with a common working directory

## Remote Execution

This system operates by connecting to a remote MCP server:

- **Remote Execution:** The orchestrator communicates with a remote MCP server (e.g., http://172.16.16.54:8080) to execute tools
- **No Local Mode:** This implementation only uses tools available on the MCP server and does not implement local versions
- **Direct Execution:** The orchestrator directly sends tool requests to the MCP server

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Orchestrator Usage

You can use the orchestrator directly to execute tools on the MCP server:

```bash
python orchestrator.py --server <server_url> <tool_name> <parameters_json>
```

For example:

```bash
# List available tools on the server
python orchestrator.py --server http://172.16.16.54:8080 --list
```

### Command Examples for Different Shells

#### Linux/macOS (Bash/Zsh)

In Linux or macOS terminals, you can use the following format:

```bash
# List files in the current directory
python orchestrator.py --server http://172.16.16.54:8080 readDirectory '{"dirPath": "."}'

# Run a shell command
python orchestrator.py --server http://172.16.16.54:8080 runShellCommand '{"command": "ls -la"}'

# Create a file
python orchestrator.py --server http://172.16.16.54:8080 createFile '{"filePath": "test.txt", "content": "Hello World"}'
```

#### Windows (PowerShell)

In PowerShell, you need to escape the JSON parameters differently:

```powershell
# List files in the current directory
python orchestrator.py --server "http://172.16.16.54:8080" readDirectory '{\"dirPath\": \".\"}'

# Run a shell command
python orchestrator.py --server "http://172.16.16.54:8080" runShellCommand '{\"command\": \"dir\"}'

# Create a file
python orchestrator.py --server "http://172.16.16.54:8080" createFile '{\"filePath\": \"test.txt\", \"content\": \"Hello World\"}'
```

Alternatively, in PowerShell you can use variables to avoid escaping issues:

```powershell
$params = '{"dirPath": "."}'
python orchestrator.py --server "http://172.16.16.54:8080" readDirectory $params
```

#### Windows (Command Prompt)

In Windows Command Prompt:

```cmd
python orchestrator.py --server "http://172.16.16.54:8080" readDirectory "{\"dirPath\": \".\"}"
```

### Example Commands for Each Tool (Bash/Linux Format)

Here are example commands for each of the available tools:

```bash
# 1. Run a shell command
python orchestrator.py --server http://172.16.16.54:8080 runShellCommand '{"command": "echo Hello, World!"}'

# 2. Execute a Python file
python orchestrator.py --server http://172.16.16.54:8080 runPythonFile '{"filePath": "script.py", "args": "--verbose"}'

# 3. List files in a directory
python orchestrator.py --server http://172.16.16.54:8080 readDirectory '{"dirPath": "./src"}'

# 4. Copy a file
python orchestrator.py --server http://172.16.16.54:8080 copyFile '{"sourcePath": "source.txt", "destinationPath": "destination.txt"}'

# 5. Create a file
python orchestrator.py --server http://172.16.16.54:8080 createFile '{"filePath": "newfile.txt", "content": "Hello, this is a new file!"}'

# 6. Read a file
python orchestrator.py --server http://172.16.16.54:8080 readFile '{"filePath": "example.txt"}'

# 7. Edit a file
python orchestrator.py --server http://172.16.16.54:8080 editFile '{"filePath": "example.txt", "operation": "append", "content": "\nNew content appended"}'

# 8. Delete a file
python orchestrator.py --server http://172.16.16.54:8080 deleteFile '{"filePath": "unwanted.txt"}'

# 9. Move a file
python orchestrator.py --server http://172.16.16.54:8080 moveFile '{"sourcePath": "oldlocation.txt", "destinationPath": "newlocation.txt"}'

# 10. Create a directory
python orchestrator.py --server http://172.16.16.54:8080 createDirectory '{"dirPath": "new_directory", "recursive": true}'

# 11. Move a directory
python orchestrator.py --server http://172.16.16.54:8080 moveDirectory '{"sourcePath": "old_dir", "destinationPath": "new_dir"}'

# 12. Copy a directory
python orchestrator.py --server http://172.16.16.54:8080 copyDirectory '{"sourcePath": "source_dir", "destinationPath": "dest_dir", "overwrite": true}'

# 13. Delete a directory
python orchestrator.py --server http://172.16.16.54:8080 deleteDirectory '{"dirPath": "unwanted_dir", "recursive": true}'

# 14. Get directory tree
python orchestrator.py --server http://172.16.16.54:8080 getDirectoryTree '{"dirPath": "project", "maxDepth": 3, "includeFiles": true}'

# 15. Search for patterns in files
python orchestrator.py --server http://172.16.16.54:8080 grep '{"pattern": "function", "filePaths": ["*.js", "*.ts"], "useRegex": true}'

# 16. Run a combination of tasks
python orchestrator.py --server http://172.16.16.54:8080 combinationTask '{"workingDir": "project", "tasks": [
  {"tool": "createDirectory", "parameters": {"dirPath": "build"}},
  {"tool": "runShellCommand", "parameters": {"command": "npm run build"}}
]}'
```



### Using the Test Script

The repository includes a convenient shell script for testing with the MCP server:

```bash
# Run full test suite against the default MCP server
./mcp_test.sh test

# Specify a different MCP server
./mcp_test.sh --server http://localhost:8080 test

# Test specific tools on the MCP server
./mcp_test.sh read /home          # Test readDirectory
./mcp_test.sh cmd "ls -la"        # Test runShellCommand
./mcp_test.sh file                # Test file operations
```

Note: All test script commands connect to the MCP server to execute tools. No local execution is supported.

## Direct MCP Client Usage

You can use the MCP client directly in Python code:

```python
from mcp_client import MCPClient

# Create client
client = MCPClient("http://172.16.16.54:8080")

# Connect to server
client.connect()

# List available tools
tools = client.get_tools()
print(f"Available tools: {[t['name'] for t in tools]}")

# Invoke a tool
result = client.invoke_tool("readDirectory", {"dirPath": "."})
print(f"Files: {result.get('files', [])}")

# Disconnect
client.disconnect()
```

## Architecture

The system consists of the following components:

1. **Orchestrator (`orchestrator.py`)**:
   - Connects to the MCP server
   - Forwards tool requests to the server
   - Returns results to the caller

2. **MCP Client (`mcp_client.py`)**:
   - Establishes an SSE connection to the MCP server
   - Sends tool invocation requests
   - Receives and processes tool results

## Note on Local Execution

This implementation is designed to work exclusively with a remote MCP server. If you need local execution capabilities, you should create a separate implementation in `assets/scripts/mcpscripts/localexecution`.

## License

MIT License
# MCP Local Execution

This directory contains a standalone version of the MCP (Model Context Protocol) tools that can be executed locally without requiring a remote MCP server.

## Overview

The local execution version provides a subset of the MCP tools that can be used by AI agents to interact with the local file system and execute commands. This implementation is designed to be simple, lightweight, and easy to use.

## Available Tools

The following MCP tools are implemented:

- `readDirectory`: List files and folders in a directory
- `runShellCommand`: Run a terminal command in the system shell
- `createFile`: Create a new file with specified content
- `readFile`: Read the content of a file
- `deleteFile`: Delete a file from the filesystem

## Usage

### Basic Usage

You can use the orchestrator directly to execute tools:

```bash
# Navigate to the localexecution directory
cd assets/scripts/mcpscripts/localexecution

# List files in the current directory
python orchestrator.py readDirectory {"dirPath": "."}

# Run a shell command
python orchestrator.py runShellCommand {"command": "echo Hello, World!"}

# Create a file
python orchestrator.py createFile {"filePath": "test.txt", "content": "Hello, World!"}

# Read a file
python orchestrator.py readFile {"filePath": "test.txt"}

# Delete a file
python orchestrator.py deleteFile {"filePath": "test.txt"}

# List available tools
python orchestrator.py --list
```

### PowerShell Usage

When using PowerShell, you need to be careful with the JSON parameters. Use single quotes around the JSON string:

```powershell
python orchestrator.py readDirectory '{"dirPath": "."}'
```

## Differences from the Terminal MCP AI Agent

This local execution version differs from the `terminal-mcp-ai-agent` in the following ways:

1. **No Remote Server**: This version only executes tools locally and does not connect to a remote MCP server.
2. **Simplified Implementation**: The code is more straightforward and does not include the complexity of handling both local and remote execution.
3. **Limited Tool Set**: Only the most essential tools are implemented.
4. **No AI Agent Connector**: This version does not include the AI agent connector for processing AI-generated tool requests.

## Extending

To add a new tool:

1. Implement the tool function in the `orchestrator.py` file
2. Add the tool name to the `AVAILABLE_TOOLS` list
3. Add the tool function to the `TOOL_FUNCTIONS` dictionary

## Example

Here's an example of how to use the local execution version:

```python
import subprocess
import json

# Execute a tool
command = 'python orchestrator.py readDirectory \'{"dirPath": "."}\''
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Parse the result
output = result.stdout
try:
    # Find the JSON part of the output
    json_start = output.find('{')
    if json_start >= 0:
        json_output = output[json_start:]
        data = json.loads(json_output)
        print(f"Files: {data.get('files', [])}")
        print(f"Directories: {data.get('directories', [])}")
    else:
        print(f"Raw output: {output}")
except json.JSONDecodeError:
    print(f"Raw output: {output}")
```

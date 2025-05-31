#!/usr/bin/env python3
"""
MCP Tool Orchestrator - Routes commands to the MCP server

Usage:
  python orchestrator.py --server SERVER_URL <tool_name> <parameters_json>

Example:
  # Use MCP server:
  python orchestrator.py --server http://172.16.16.54:8080 readDirectory {"dirPath": "."}
"""

import sys
import json
import logging
import argparse
from typing import Dict, Any, List

# Import MCP client
from mcp_client import MCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mcp_orchestrator')

# Available MCP tools
AVAILABLE_TOOLS = [
    "runShellCommand",
    "runPythonFile",
    "readDirectory",
    "copyFile",
    "createFile",
    "readFile",
    "editFile",
    "deleteFile",
    "moveFile",
    "createDirectory",
    "moveDirectory",
    "copyDirectory",
    "deleteDirectory",
    "getDirectoryTree",
    "grep",
    "combinationTask"
]

# No local tool execution in this version - only remote MCP server is supported

def execute_remote_tool(server_url: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the specified tool on a remote MCP server

    Args:
        server_url: URL of the MCP server
        tool_name: Name of the tool to execute
        parameters: Parameters to pass to the tool

    Returns:
        Tool execution result
    """
    try:
        # Create MCP client
        client = MCPClient(server_url)

        # Connect to server
        logger.info(f"Connecting to MCP server: {server_url}")
        client.connect()

        # Execute the tool
        logger.info(f"Executing remote tool: {tool_name}")
        result = client.invoke_tool(tool_name, parameters)

        # Disconnect
        client.disconnect()

        return result
    except Exception as e:
        logger.error(f"Error executing remote tool {tool_name}: {str(e)}")
        return {"error": f"Error executing remote tool {tool_name}: {str(e)}"}

def list_available_tools(server_url: str) -> List[Dict[str, Any]]:
    """
    Get information about all available tools from the MCP server

    Args:
        server_url: URL of the MCP server to query

    Returns:
        List of tool information objects
    """
    try:
        client = MCPClient(server_url)
        client.connect()
        tools = client.get_tools()
        client.disconnect()
        return tools
    except Exception as e:
        logger.error(f"Error getting tools from server: {str(e)}")
        return []

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="URL of the MCP server to use")
    parser.add_argument("tool_name", nargs="?", help="Name of the tool to execute")
    parser.add_argument("parameters", nargs="?", help="Tool parameters as JSON string")
    parser.add_argument("--list", action="store_true", help="List available tools")

    args = parser.parse_args()

    # Handle list command
    if args.list:
        print(f"Listing tools from server: {args.server}")
        tools = list_available_tools(args.server)

        print("\nAvailable MCP Tools:")
        for tool in tools:
            print(f"- {tool['name']}: {tool.get('description', 'No description')}")
        sys.exit(0)

    # Check for required arguments
    if not args.tool_name:
        parser.print_help()
        sys.exit(1)

    # Parse parameters
    parameters = {}
    if args.parameters:
        try:
            # First try to parse as JSON directly
            parameters = json.loads(args.parameters)
        except json.JSONDecodeError:
            try:
                # If that fails, try to handle escaped JSON (common in PowerShell)
                # Remove escaped quotes and then parse
                cleaned_params = args.parameters.replace('\\"', '"').replace('\\\\', '\\')
                if cleaned_params.startswith('{\\') or cleaned_params.startswith('{"'):
                    # Handle PowerShell escaped JSON format
                    cleaned_params = cleaned_params.replace('\\', '')
                    parameters = json.loads(cleaned_params)
                else:
                    # If still not valid JSON, treat as a single string parameter
                    parameters = {"input": args.parameters}
            except json.JSONDecodeError:
                # If all parsing attempts fail, treat as a single string parameter
                parameters = {"input": args.parameters}

    return args.server, args.tool_name, parameters

def main():
    """Main entry point"""
    server_url, tool_name, parameters = parse_arguments()

    # Check if server URL is provided
    if not server_url:
        print("Error: Server URL is required. Use --server option.")
        sys.exit(1)

    # Execute the tool on remote server
    print(f"Executing tool on remote server: {server_url}")
    result = execute_remote_tool(server_url, tool_name, parameters)

    # Output the result as JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
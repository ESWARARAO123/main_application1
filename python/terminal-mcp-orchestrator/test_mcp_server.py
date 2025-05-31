#!/usr/bin/env python3
"""
Test MCP Server - Verifies the connection to the MCP server and executes basic tools

Usage:
  python test_mcp_server.py <server_url>

Example:
  python test_mcp_server.py http://172.16.16.54:8080
"""

import sys
import os
import json
import argparse
import time
from mcp_client import MCPClient

def test_connection(server_url: str):
    """Test connection to the MCP server"""
    print("\n=== Testing Connection ===")
    client = MCPClient(server_url)
    
    try:
        client_id = client.connect()
        print(f"✅ Connected successfully! Client ID: {client_id}")
        
        # Get server info
        server_info = client.get_server_info()
        print(f"Server name: {server_info.get('name', 'Unknown')}")
        print(f"Server version: {server_info.get('version', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False
    finally:
        client.disconnect()

def test_list_tools(server_url: str):
    """Test listing available tools from the MCP server"""
    print("\n=== Testing Tool Listing ===")
    client = MCPClient(server_url)
    
    try:
        client.connect()
        tools = client.get_tools()
        
        print(f"✅ Retrieved {len(tools)} tools:")
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
        
        return tools
    except Exception as e:
        print(f"❌ Tool listing failed: {str(e)}")
        return []
    finally:
        client.disconnect()

def test_read_directory(server_url: str, directory: str = "."):
    """Test the readDirectory tool"""
    print(f"\n=== Testing readDirectory Tool (path: {directory}) ===")
    client = MCPClient(server_url)
    
    try:
        client.connect()
        result = client.invoke_tool("readDirectory", {"dirPath": directory})
        
        if "error" in result:
            print(f"❌ Tool execution failed: {result['error']}")
            return False
        
        print("✅ Tool execution successful!")
        print(f"Files: {', '.join(result.get('files', []))}")
        print(f"Directories: {', '.join(result.get('directories', []))}")
        
        return True
    except Exception as e:
        print(f"❌ Tool execution failed: {str(e)}")
        return False
    finally:
        client.disconnect()

def test_run_command(server_url: str, command: str = "echo Hello from MCP!"):
    """Test the runShellCommand tool"""
    print(f"\n=== Testing runShellCommand Tool (command: {command}) ===")
    client = MCPClient(server_url)
    
    try:
        client.connect()
        result = client.invoke_tool("runShellCommand", {"command": command})
        
        if "error" in result:
            print(f"❌ Tool execution failed: {result['error']}")
            return False
        
        print("✅ Tool execution successful!")
        print(f"Exit code: {result.get('exitCode', 'Unknown')}")
        print(f"Output: {result.get('stdout', '')}")
        
        return True
    except Exception as e:
        print(f"❌ Tool execution failed: {str(e)}")
        return False
    finally:
        client.disconnect()

def test_create_and_read_file(server_url: str):
    """Test creating and reading a file"""
    print("\n=== Testing File Creation and Reading ===")
    client = MCPClient(server_url)
    
    test_file = f"test_file_{int(time.time())}.txt"
    test_content = "Hello, MCP!\nThis is a test file created by the MCP client."
    
    try:
        client.connect()
        
        # Create file
        print(f"Creating file: {test_file}")
        create_result = client.invoke_tool("createFile", {
            "filePath": test_file,
            "content": test_content,
            "overwrite": True
        })
        
        if "error" in create_result:
            print(f"❌ File creation failed: {create_result['error']}")
            return False
        
        print(f"✅ File created successfully: {create_result.get('filePath', test_file)}")
        
        # Read file
        print(f"Reading file: {test_file}")
        read_result = client.invoke_tool("readFile", {
            "filePath": test_file
        })
        
        if "error" in read_result:
            print(f"❌ File reading failed: {read_result['error']}")
            return False
        
        content = read_result.get("content", "")
        print(f"✅ File read successfully ({len(content)} characters)")
        print(f"Content: {content}")
        
        # Verify content
        content_matches = content == test_content
        if content_matches:
            print("✅ Content verification: Content matches!")
        else:
            print("❌ Content verification: Content doesn't match!")
        
        return content_matches
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False
    finally:
        # Clean up
        try:
            print(f"Cleaning up test file: {test_file}")
            client.invoke_tool("deleteFile", {"filePath": test_file})
        except:
            pass
        
        client.disconnect()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_url", help="URL of the MCP server to test")
    args = parser.parse_args()
    
    server_url = args.server_url
    
    print(f"Testing MCP server at: {server_url}")
    
    # Test connection
    if not test_connection(server_url):
        print("\n❌ Connection test failed. Aborting remaining tests.")
        sys.exit(1)
    
    # Test listing tools
    tools = test_list_tools(server_url)
    if not tools:
        print("\n❌ Tool listing test failed. Aborting remaining tests.")
        sys.exit(1)
    
    # Test readDirectory
    test_read_directory(server_url)
    
    # Test runShellCommand
    test_run_command(server_url)
    
    # Test file operations
    test_create_and_read_file(server_url)
    
    print("\n=== Test Summary ===")
    print("All tests completed.")

if __name__ == "__main__":
    main() 
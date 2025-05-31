#!/usr/bin/env python3
"""
MCP Explorer - A tool to explore and test all available MCP tools
"""

import json
import requests # type: ignore
import sseclient # type: ignore
import threading
import time
import uuid
import os
import sys

class MCPClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.client_id = None
        self.sse_thread = None
        self.running = False
        self.connected = False
        self.tools = []
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
        
        # Get list of available tools
        self.get_tools()
        
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
                        
                        # Handle error
                        elif data.get("type") == "error":
                            print(f"Received error: {json.dumps(data, indent=2)}")
                    
                    except json.JSONDecodeError:
                        print(f"Received non-JSON data: {event.data}")
        
        except Exception as e:
            print(f"SSE connection error: {e}")
            self.running = False
    
    def get_tools(self):
        """Get list of available tools from server"""
        try:
            response = requests.get(f"{self.server_url}/tools")
            response.raise_for_status()
            
            data = response.json()
            self.tools = data.get("tools", [])
            print(f"Retrieved {len(self.tools)} available tools")
            
            return self.tools
        except requests.RequestException as e:
            print(f"Error getting tools: {e}")
            return []
    
    def invoke_tool(self, tool_name, parameters=None):
        """Invoke an MCP tool with the given parameters"""
        if not self.client_id:
            raise ValueError("Not connected. Call connect() first")
        
        if parameters is None:
            parameters = {}
        
        message_id = f"msg-{uuid.uuid4()}"
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
        try:
            response = requests.post(
                f"{self.server_url}/messages",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error invoking tool: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            return None
    
    def disconnect(self):
        """Disconnect from the SSE stream"""
        self.running = False
        if self.sse_thread:
            self.sse_thread.join(timeout=1)
        self.connected = False
        self.client_id = None


class MCPExplorer:
    def __init__(self, server_url):
        self.client = MCPClient(server_url)
        self.test_dir = "mcp_test"
        self.test_file = os.path.join(self.test_dir, "test_file.txt")
    
    def setup(self):
        """Connect to MCP server and prepare test environment"""
        print(f"Connecting to MCP server at {self.client.server_url}...")
        self.client.connect()
        time.sleep(1)  # Wait for connection to stabilize
    
    def cleanup(self):
        """Clean up test files and disconnect"""
        print("Cleaning up test files...")
        try:
            # Try to delete test directory if it exists
            self.client.invoke_tool("deleteDirectory", {
                "dirPath": self.test_dir,
                "recursive": True
            })
            time.sleep(1)
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        print("Disconnecting from MCP server...")
        self.client.disconnect()
    
    def print_tools(self):
        """Display all available tools"""
        print("\n=== AVAILABLE MCP TOOLS ===")
        
        for idx, tool in enumerate(self.client.tools, 1):
            name = tool.get("name", "Unknown")
            desc = tool.get("description", "No description")
            
            print(f"{idx}. {name}")
            print(f"   Description: {desc}")
            
            # Print parameters
            parameters = tool.get("parameters", {})
            if parameters:
                print("   Parameters:")
                for param_name, param_info in parameters.items():
                    required = param_info.get("required", False)
                    param_desc = param_info.get("description", "No description")
                    req_text = "(Required)" if required else "(Optional)"
                    print(f"     - {param_name} {req_text}: {param_desc}")
            
            print()
    
    def test_file_operations(self):
        """Test file-related operations"""
        print("\n=== TESTING FILE OPERATIONS ===")
        
        # Create test directory
        print("\n>> Creating test directory")
        dir_result = self.client.invoke_tool("createDirectory", {
            "dirPath": self.test_dir
        })
        
        # Create a test file
        print("\n>> Creating test file")
        file_content = f"This is a test file created at {time.ctime()}\nLine 2\nLine 3\nLine 4\nLine 5"
        create_result = self.client.invoke_tool("createFile", {
            "filePath": self.test_file,
            "content": file_content
        })
        
        # Read the file
        print("\n>> Reading test file")
        read_result = self.client.invoke_tool("readFile", {
            "filePath": self.test_file
        })
        
        # Edit the file
        print("\n>> Editing test file (append)")
        edit_result = self.client.invoke_tool("editFile", {
            "filePath": self.test_file,
            "operation": "append",
            "content": "\nThis line was appended"
        })
        
        # Read the file again
        print("\n>> Reading edited file")
        read_again = self.client.invoke_tool("readFile", {
            "filePath": self.test_file
        })
        
        # Copy the file
        print("\n>> Copying file")
        copy_result = self.client.invoke_tool("copyFile", {
            "sourcePath": self.test_file,
            "destinationPath": os.path.join(self.test_dir, "copy_file.txt")
        })
        
        # List directory
        print("\n>> Listing test directory")
        list_result = self.client.invoke_tool("readDirectory", {
            "dirPath": self.test_dir
        })
    
    def test_shell_commands(self):
        """Test shell command execution"""
        print("\n=== TESTING SHELL COMMANDS ===")
        
        # Run a simple directory listing
        print("\n>> Running 'dir' command")
        dir_result = self.client.invoke_tool("runShellCommand", {
            "command": "dir"
        })
        
        # Run an echo command
        print("\n>> Running echo command")
        echo_result = self.client.invoke_tool("runShellCommand", {
            "command": "echo Hello from MCP Explorer!"
        })
    
    def test_directory_operations(self):
        """Test directory-related operations"""
        print("\n=== TESTING DIRECTORY OPERATIONS ===")
        
        # Create nested directory
        nested_dir = os.path.join(self.test_dir, "nested")
        print("\n>> Creating nested directory")
        nested_result = self.client.invoke_tool("createDirectory", {
            "dirPath": nested_dir
        })
        
        # Create a file in the nested directory
        print("\n>> Creating file in nested directory")
        nested_file = self.client.invoke_tool("createFile", {
            "filePath": os.path.join(nested_dir, "nested_file.txt"),
            "content": "This is a file in the nested directory"
        })
        
        # Get directory tree
        print("\n>> Getting directory tree")
        tree_result = self.client.invoke_tool("getDirectoryTree", {
            "dirPath": self.test_dir,
            "maxDepth": 3,
            "includeFiles": True
        })
        
        # Copy directory
        copy_dir = os.path.join(self.test_dir, "copied_dir")
        print("\n>> Copying directory")
        copy_dir_result = self.client.invoke_tool("copyDirectory", {
            "sourcePath": nested_dir,
            "destinationPath": copy_dir
        })
        
        # List directories after copy
        print("\n>> Listing directory after copy")
        list_again = self.client.invoke_tool("readDirectory", {
            "dirPath": self.test_dir
        })
    
    def test_grep(self):
        """Test the grep tool"""
        print("\n=== TESTING GREP TOOL ===")
        
        # Create multiple files with different content
        self.client.invoke_tool("createFile", {
            "filePath": os.path.join(self.test_dir, "grep_test1.txt"),
            "content": "This file contains the word 'example' in it\nAnd has multiple lines\nFor testing grep"
        })
        
        self.client.invoke_tool("createFile", {
            "filePath": os.path.join(self.test_dir, "grep_test2.txt"),
            "content": "This file does not have the word\nBut it does have multiple lines\nFor testing"
        })
        
        # Use grep to search for pattern
        print("\n>> Searching for 'example' pattern")
        grep_result = self.client.invoke_tool("grep", {
            "pattern": "example",
            "filePaths": os.path.join(self.test_dir, "*.txt"),
            "useRegex": True,
            "caseSensitive": False
        })
    
    def run_all_tests(self):
        """Run all tests"""
        try:
            self.setup()
            self.print_tools()
            
            # Run tests
            self.test_file_operations()
            self.test_directory_operations()
            self.test_shell_commands()
            self.test_grep()
            
            print("\n=== ALL TESTS COMPLETED SUCCESSFULLY ===")
        
        except Exception as e:
            print(f"Error during tests: {e}")
        
        finally:
            # Keep connection alive for a bit to receive any pending SSE messages
            print("\nKeeping connection alive for 5 seconds to receive any pending results...")
            time.sleep(5)
            
            self.cleanup()
            print("MCP Explorer test complete!")


def main():
    """Main function"""
    server_url = "http://172.16.16.54:8080"
    
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    
    explorer = MCPExplorer(server_url)
    explorer.run_all_tests()


if __name__ == "__main__":
    main() 
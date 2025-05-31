#!/usr/bin/env python3
"""
Test Specific MCP Tool - A focused test of a particular MCP tool
"""

import json
import requests
import sseclient
import threading
import time
import uuid
import sys

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
            print(f"Connecting to SSE endpoint: {self.server_url}/sse")
            response = requests.get(f"{self.server_url}/sse", stream=True)
            client = sseclient.SSEClient(response)
            
            for event in client.events():
                if not self.running:
                    break
                
                if event.data:
                    try:
                        print(f"Raw SSE event: {event.data}")
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
                
                # Handle ping
                elif event.event == "ping":
                    print("Received ping")
        
        except Exception as e:
            print(f"SSE connection error: {e}")
            self.running = False
    
    def get_info(self):
        """Get server info"""
        try:
            print(f"Getting server info from: {self.server_url}/info")
            response = requests.get(f"{self.server_url}/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting server info: {e}")
            return None
    
    def get_tools(self):
        """Get list of available tools"""
        try:
            print(f"Getting available tools from: {self.server_url}/tools")
            response = requests.get(f"{self.server_url}/tools")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting tools: {e}")
            return None
    
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
        print(f"Using payload: {json.dumps(payload, indent=2)}")
        try:
            response = requests.post(
                f"{self.server_url}/messages",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            print(f"Immediate response: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
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


def main():
    """Main function to test a specific MCP tool"""
    if len(sys.argv) < 2:
        print("Usage: python test_specific_tool.py SERVER_URL [TOOL_NAME]")
        print("Example: python test_specific_tool.py http://172.16.16.54:8080 readDirectory")
        return
    
    server_url = sys.argv[1]
    
    # Default tool to test
    tool_name = "readDirectory"
    if len(sys.argv) > 2:
        tool_name = sys.argv[2]
    
    client = MCPClient(server_url)
    
    try:
        # Get server info
        print("\n=== SERVER INFO ===")
        info = client.get_info()
        if info:
            print(json.dumps(info, indent=2))
        
        # Get available tools
        print("\n=== AVAILABLE TOOLS ===")
        tools = client.get_tools()
        if tools:
            print(json.dumps(tools, indent=2))
        
        # Connect to the server
        print(f"\nConnecting to MCP server at {server_url}...")
        client.connect()
        
        # Test the specified tool
        print(f"\n=== TESTING TOOL: {tool_name} ===")
        
        # Default parameters based on tool name
        params = {}
        if tool_name == "readDirectory":
            params = {}  # empty for root dir
        elif tool_name == "createFile":
            params = {
                "filePath": "test-file-from-specific-test.txt",
                "content": f"This is a test file created at {time.ctime()}"
            }
        elif tool_name == "runShellCommand":
            params = {
                "command": "dir"
            }
        elif tool_name == "getDirectoryTree":
            params = {
                "dirPath": ".",
                "maxDepth": 2
            }
        
        # Invoke the tool
        result = client.invoke_tool(tool_name, params)
        
        # Wait for SSE events
        print("\nKeeping connection alive for 10 seconds to receive SSE events...")
        time.sleep(10)
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        print("Disconnecting from MCP server...")
        client.disconnect()
        print("Test complete!")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
MCP Client - A Python client for interacting with the Model Context Protocol (MCP) server
"""

import json
import requests # type: ignore
import sseclient # type: ignore
import threading
import time
import uuid

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


def main():
    """Main function to demonstrate MCP client usage"""
    server_url = "http://172.16.16.54:8080"
    client = MCPClient(server_url)
    
    try:
        print(f"Connecting to MCP server at {server_url}...")
        client.connect()
        
        # Wait a moment to ensure connection is fully established
        time.sleep(1)
        
        # Test various tools
        
        # 1. Read Directory
        print("\n--- Reading Directory ---")
        dir_result = client.invoke_tool("readDirectory")
        if dir_result:
            print(json.dumps(dir_result, indent=2))
        
        # 2. Create Test File
        print("\n--- Creating Test File ---")
        create_result = client.invoke_tool("createFile", {
            "filePath": "mcp-python-test.txt",
            "content": f"This is a test file created by Python MCP client at {time.ctime()}"
        })
        if create_result:
            print(json.dumps(create_result, indent=2))
        
        # 3. Read Test File
        print("\n--- Reading Test File ---")
        read_result = client.invoke_tool("readFile", {
            "filePath": "mcp-python-test.txt"
        })
        if read_result:
            print(json.dumps(read_result, indent=2))
        
        # 4. Run Shell Command
        print("\n--- Running Shell Command ---")
        shell_result = client.invoke_tool("runShellCommand", {
            "command": "dir"
        })
        if shell_result:
            print(json.dumps(shell_result, indent=2))
        
        # Keep the connection alive for a moment to receive results via SSE
        print("\nKeeping connection alive for 5 seconds to receive SSE events...")
        time.sleep(5)
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        print("Disconnecting from MCP server...")
        client.disconnect()
        print("MCP client test complete!")


if __name__ == "__main__":
    main() 
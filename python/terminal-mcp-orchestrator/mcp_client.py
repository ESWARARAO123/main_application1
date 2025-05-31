#!/usr/bin/env python3
"""
MCP Client - Connects to an MCP server via SSE and executes tools

This client:
1. Establishes an SSE connection to get a clientId
2. Invokes tools through the messages endpoint
3. Returns the results

Usage:
  python mcp_client.py <server_url> <tool_name> <parameters_json>
  
Example:
  python mcp_client.py http://172.16.16.54:8080 readDirectory {"dirPath": "."}
"""

import sys
import json
import time
import requests
import threading
import argparse
import uuid
import sseclient
from typing import Dict, Any, Optional, List

class MCPClient:
    def __init__(self, server_url: str):
        """
        Initialize the MCP client
        
        Args:
            server_url: URL of the MCP server (e.g., http://172.16.16.54:8080)
        """
        self.server_url = server_url.rstrip('/')
        self.client_id = None
        self.sse_thread = None
        self.running = False
        self.connected = False
        self.headers = {
            "Content-Type": "application/json"
        }
        self.tools_cache = None
        self.tool_results = {}
        self.tool_events = {}
        self.connection_event = threading.Event()
    
    def connect(self, timeout: int = 10) -> str:
        """
        Establish SSE connection and get client ID
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            client_id: The MCP client ID
        """
        if self.connected:
            return self.client_id
        
        self.running = True
        self.sse_thread = threading.Thread(target=self._sse_listener)
        self.sse_thread.daemon = True
        self.sse_thread.start()
        
        # Wait for connection to be established
        if not self.connection_event.wait(timeout=timeout):
            raise TimeoutError("Failed to establish SSE connection within timeout period")
        
        return self.client_id
    
    def _sse_listener(self):
        """Background thread that listens to SSE events"""
        try:
            print(f"Connecting to SSE endpoint: {self.server_url}/sse")
            response = requests.get(f"{self.server_url}/sse", stream=True)
            response.raise_for_status()
            
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
                            self.connection_event.set()
                        
                        # Handle tool result
                        elif data.get("type") == "tool_result":
                            message_id = data.get("id")
                            print(f"Received tool result for message: {message_id}")
                            
                            self.tool_results[message_id] = data
                            
                            # Set event for the message
                            if message_id in self.tool_events:
                                self.tool_events[message_id].set()
                    
                    except json.JSONDecodeError:
                        print(f"Received non-JSON data: {event.data}")
        
        except Exception as e:
            print(f"SSE connection error: {e}")
            self.running = False
            self.connection_event.set()  # Set event to unblock connect method
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        try:
            response = requests.get(f"{self.server_url}/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting server info: {e}")
            return {}
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from server"""
        if self.tools_cache:
            return self.tools_cache
            
        try:
            print(f"Getting available tools from: {self.server_url}/tools")
            response = requests.get(f"{self.server_url}/tools")
            response.raise_for_status()
            
            data = response.json()
            tools = data.get("tools", [])
            print(f"Retrieved {len(tools)} available tools")
            self.tools_cache = tools
            return tools
        except Exception as e:
            print(f"Error getting tools: {e}")
            return []
    
    def invoke_tool(self, tool_name: str, parameters: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """
        Invoke an MCP tool and wait for the result
        
        Args:
            tool_name: Name of the tool to invoke
            parameters: Parameters to pass to the tool
            timeout: Maximum time to wait for result in seconds
            
        Returns:
            Result of the tool invocation
        """
        message_id = f"msg-{uuid.uuid4()}"
        self.tool_events[message_id] = threading.Event()
        
        result = self._send_tool_invocation(message_id, tool_name, parameters)
        if "error" in result:
            return result
        
        # Wait for the result via SSE
        print(f"Waiting for result of message: {message_id}")
        if not self.tool_events[message_id].wait(timeout=timeout):
            del self.tool_events[message_id]
            return {"error": f"Timeout waiting for tool result after {timeout} seconds"}
        
        # Get the result
        tool_result = self.tool_results.get(message_id)
        
        # Clean up
        del self.tool_events[message_id]
        if message_id in self.tool_results:
            del self.tool_results[message_id]
        
        # Extract the content from the result
        if tool_result and "content" in tool_result and "content" in tool_result["content"]:
            content_items = tool_result["content"]["content"]
            if content_items and isinstance(content_items, list) and "text" in content_items[0]:
                result_text = content_items[0]["text"]
                # Try to parse JSON
                try:
                    result_data = json.loads(result_text)
                    return result_data
                except json.JSONDecodeError:
                    return {"text": result_text}
        
        return tool_result or {"error": "No result received"}
    
    def _send_tool_invocation(self, message_id: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send a tool invocation to the server"""
        if not self.client_id:
            return {"error": "Not connected. Call connect() first"}
        
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
        print(f"Parameters: {json.dumps(parameters, indent=2)}")
        
        try:
            response = requests.post(
                f"{self.server_url}/messages",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error sending tool invocation: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            return {"error": str(e)}
    
    def disconnect(self):
        """Disconnect from the SSE stream"""
        self.running = False
        if self.sse_thread:
            self.sse_thread.join(timeout=1)
        self.connected = False
        self.client_id = None
        self.tools_cache = None
        print("Disconnected from MCP server")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_url", help="URL of the MCP server")
    parser.add_argument("tool_name", help="Name of the tool to invoke")
    parser.add_argument("parameters", help="Tool parameters as JSON string")
    args = parser.parse_args()
    
    # Parse parameters
    try:
        parameters = json.loads(args.parameters)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON parameters: {args.parameters}")
        sys.exit(1)
    
    # Create client
    client = MCPClient(args.server_url)
    
    try:
        # Connect to server
        client.connect()
        
        # Get server info
        server_info = client.get_server_info()
        print(f"Server: {server_info.get('name', 'Unknown')} {server_info.get('version', 'Unknown')}")
        
        # Invoke tool
        result = client.invoke_tool(args.tool_name, parameters)
        
        # Print result
        print("\nResult:")
        print(json.dumps(result, indent=2))
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        # Disconnect
        client.disconnect()

if __name__ == "__main__":
    main() 
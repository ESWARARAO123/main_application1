#!/usr/bin/env python3
"""
MCP Agent Demo - A simplified demonstration of an AI agent using the MCP server
"""

import json
import requests
import sseclient
import threading
import time
import uuid
import sys
import os
import re
from typing import Dict, List, Any, Optional, Union

# MCP Client class - handles communication with the MCP server
class MCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client_id = None
        self.sse_thread = None
        self.running = False
        self.connected = False
        self.headers = {
            "Content-Type": "application/json"
        }
        self.tools_cache = None
    
    def connect(self) -> str:
        """Establish SSE connection and get client ID"""
        if self.connected:
            return self.client_id
            
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
                        data = json.loads(event.data)
                        
                        # Handle connected event
                        if data.get("type") == "connected":
                            self.client_id = data.get("clientId")
                            self.connected = True
                            print(f"Connected with client ID: {self.client_id}")
                        
                        # Handle tool result
                        elif data.get("type") == "tool_result":
                            print(f"Received tool result via SSE: {json.dumps(data, indent=2)}")
                    
                    except json.JSONDecodeError:
                        print(f"Received non-JSON data: {event.data}")
        
        except Exception as e:
            print(f"SSE connection error: {e}")
            self.running = False
    
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
    
    def invoke_tool(self, tool_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        print(f"Parameters: {json.dumps(parameters, indent=2)}")
        
        try:
            response = requests.post(
                f"{self.server_url}/messages",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the content from the result
            if "content" in result and "content" in result["content"]:
                content_items = result["content"]["content"]
                if content_items and isinstance(content_items, list) and "text" in content_items[0]:
                    result_text = content_items[0]["text"]
                    # Try to parse JSON
                    try:
                        result_data = json.loads(result_text)
                        return result_data
                    except json.JSONDecodeError:
                        return {"text": result_text}
            
            return result
        except Exception as e:
            print(f"Error invoking tool: {e}")
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


class MCPAgent:
    """Simulated AI agent that uses MCP tools to respond to user queries"""
    
    def __init__(self, mcp_server_url: str):
        self.client = MCPClient(mcp_server_url)
        self.tools = []
        self.server_info = {}
    
    def initialize(self):
        """Initialize the agent by connecting to MCP and getting available tools"""
        print("Initializing MCP Agent...")
        
        # Connect to MCP server
        self.client.connect()
        
        # Get server info
        self.server_info = self.client.get_server_info()
        print(f"Connected to MCP server: {self.server_info.get('name', 'Unknown')} version {self.server_info.get('version', 'Unknown')}")
        
        # Get available tools
        self.tools = self.client.get_tools()
        
        print("MCP Agent initialization complete!")
        print(f"Available tools: {', '.join(t['name'] for t in self.tools)}")
    
    def generate_system_prompt(self) -> str:
        """Generate a system prompt that teaches the AI about available tools"""
        tools_desc = []
        
        for tool in self.tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
            
            param_desc = []
            for param_name, param_info in params.items():
                required = "(Required)" if param_info.get("required", False) else "(Optional)"
                param_desc.append(f"    - {param_name} {required}: {param_info.get('description', '')}")
            
            tool_desc = f"{name}: {desc}\nParameters:\n" + "\n".join(param_desc)
            tools_desc.append(tool_desc)
        
        tools_text = "\n\n".join(tools_desc)
        
        system_prompt = f"""
You are a helpful AI assistant that can use tools to help users. When you need to use a tool, follow this format:

THINK: [Your reasoning about what tool to use]
TOOL: {{
  "tool": "toolName",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}

After you receive the tool result, provide your final answer.

Available tools:
{tools_text}

Example:
User: What files are in the current directory?
Assistant:
THINK: I need to check the contents of the current directory to answer this question.
TOOL: {{
  "tool": "readDirectory",
  "parameters": {{
    "dirPath": "."
  }}
}}

[After receiving tool result]
The current directory contains the following files and directories: [list based on result]
"""
        return system_prompt
    
    def extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a tool call from the AI's response"""
        try:
            # Look for TOOL: marker followed by JSON
            match = re.search(r'TOOL:\s*(\{[\s\S]*?\})', text)
            if match:
                json_str = match.group(1)
                tool_call = json.loads(json_str)
                
                if "tool" in tool_call and "parameters" in tool_call:
                    return {
                        "tool_name": tool_call["tool"],
                        "parameters": tool_call["parameters"]
                    }
            return None
        except Exception as e:
            print(f"Error extracting tool call: {e}")
            return None
    
    def simulate_ai_response(self, prompt: str) -> str:
        """
        Simulate an AI response to the user prompt
        In a real implementation, this would call an actual AI model
        """
        # Simple simulation for demo purposes
        prompt_lower = prompt.lower()
        
        if "files" in prompt_lower and "directory" in prompt_lower:
            return """
THINK: The user is asking about files in a directory. I should use the readDirectory tool to list the files and directories in the current workspace.

TOOL: {
  "tool": "readDirectory",
  "parameters": {
    "dirPath": "."
  }
}
"""
        elif "create" in prompt_lower and "file" in prompt_lower:
            current_time = time.ctime()
            return f"""
THINK: The user wants to create a file. I should use the createFile tool to create a new file with some sample content.

TOOL: {{
  "tool": "createFile",
  "parameters": {{
    "filePath": "sample-file.txt",
    "content": "This is a sample file created by the MCP Agent at {current_time}"
  }}
}}
"""
        elif "run" in prompt_lower and "command" in prompt_lower:
            return """
THINK: The user wants to run a shell command. I should use the runShellCommand tool to execute a simple command like 'dir' to list files.

TOOL: {
  "tool": "runShellCommand",
  "parameters": {
    "command": "dir"
  }
}
"""
        else:
            return "I don't need to use any tools to answer this question. How can I help you with something else?"
    
    def process_query(self, query: str) -> str:
        """Process a user query and return a response using MCP tools if needed"""
        print(f"\nProcessing query: {query}")
        
        # 1. Get AI's initial response (simulated for this demo)
        ai_response = self.simulate_ai_response(query)
        print(f"\nAI response:\n{ai_response}")
        
        # 2. Extract tool call if present
        tool_call = self.extract_tool_call(ai_response)
        
        if tool_call:
            print(f"\nExtracted tool call: {json.dumps(tool_call, indent=2)}")
            
            # 3. Execute the tool
            print("\n[DEMO] Would you like to execute this tool? (yes/no)")
            user_approval = input("> ").strip().lower()
            
            if user_approval in ["yes", "y"]:
                try:
                    # Execute the tool
                    tool_name = tool_call["tool_name"]
                    parameters = tool_call["parameters"]
                    
                    print(f"\nExecuting tool: {tool_name}")
                    result = self.client.invoke_tool(tool_name, parameters)
                    
                    # Format the result
                    if isinstance(result, dict):
                        result_str = json.dumps(result, indent=2)
                    else:
                        result_str = str(result)
                    
                    print(f"\nTool result:\n{result_str}")
                    
                    # 4. Generate final response (simulated)
                    if "error" in result:
                        return f"I tried to help, but encountered an error: {result['error']}"
                    
                    # Simulate AI analyzing the result
                    if tool_name == "readDirectory":
                        files = result.get("files", [])
                        dirs = result.get("directories", [])
                        return f"I found {len(files)} files and {len(dirs)} directories in the current location."
                    elif tool_name == "createFile":
                        return f"I've created the file '{parameters['filePath']}' with the content you requested."
                    elif tool_name == "runShellCommand":
                        return f"I ran the command and here's what I found: {result_str}"
                    else:
                        return f"I executed the tool and got this result: {result_str}"
                    
                except Exception as e:
                    print(f"Error executing tool: {e}")
                    return f"I tried to help, but encountered an error: {str(e)}"
            else:
                return "You declined to run the tool. Is there something else I can help with?"
        else:
            # No tool call needed
            return ai_response
    
    def shutdown(self):
        """Shut down the agent and disconnect from MCP"""
        self.client.disconnect()
        print("MCP Agent has been shut down.")


def main():
    """Main function to demonstrate MCP agent"""
    if len(sys.argv) < 2:
        print("Usage: python mcp_agent_demo.py SERVER_URL")
        print("Example: python mcp_agent_demo.py http://172.16.16.54:8080")
        return
    
    server_url = sys.argv[1]
    
    # Initialize agent
    agent = MCPAgent(server_url)
    
    try:
        # Initialize the agent
        agent.initialize()
        
        # Generate and show the system prompt
        system_prompt = agent.generate_system_prompt()
        print("\n=== SYSTEM PROMPT ===")
        print(system_prompt)
        
        # Interactive session
        print("\n=== MCP AGENT DEMO ===")
        print("Type your questions or 'exit' to quit")
        
        while True:
            print("\nYou: ", end="")
            query = input().strip()
            
            if query.lower() in ["exit", "quit", "bye"]:
                break
            
            if not query:
                continue
            
            # Process the query
            response = agent.process_query(query)
            
            # Display the response
            print(f"\nAssistant: {response}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Shut down the agent
        agent.shutdown()
        print("MCP Agent demo complete")


if __name__ == "__main__":
    main() 
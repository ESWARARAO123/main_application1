#!/usr/bin/env python3
"""
Example script demonstrating how to use the local MCP tools
"""

import subprocess
import json
import os
import sys

def run_tool(tool_name, parameters):
    """
    Run an MCP tool and return the result
    
    Args:
        tool_name: Name of the tool to execute
        parameters: Dictionary of parameters
        
    Returns:
        Dictionary containing the tool result
    """
    # Convert parameters to JSON string
    params_json = json.dumps(parameters)
    
    # Construct the command
    if sys.platform == 'win32':
        # For Windows, use single quotes around the JSON
        command = f'python orchestrator.py {tool_name} \'{params_json}\''
    else:
        # For Unix-like systems
        command = f'python orchestrator.py {tool_name} \'{params_json}\''
    
    # Execute the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Parse the result
    output = result.stdout
    try:
        # Find the JSON part of the output
        json_start = output.find('{')
        if json_start >= 0:
            json_output = output[json_start:]
            return json.loads(json_output)
        else:
            return {"error": f"No JSON found in output: {output}"}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON in output: {output}"}

def main():
    """Main entry point"""
    print("MCP Local Execution Example")
    print("==========================")
    
    # Example 1: List files in the current directory
    print("\n1. Listing files in the current directory:")
    result = run_tool("readDirectory", {"dirPath": "."})
    if "files" in result:
        print(f"Files: {result['files']}")
        print(f"Directories: {result['directories']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Example 2: Run a shell command
    print("\n2. Running a shell command:")
    result = run_tool("runShellCommand", {"command": "echo Hello, World!"})
    if "stdout" in result:
        print(f"Output: {result['stdout'].strip()}")
        print(f"Exit code: {result['exitCode']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Example 3: Create a file
    print("\n3. Creating a file:")
    test_file = "test_example.txt"
    result = run_tool("createFile", {
        "filePath": test_file,
        "content": "This is a test file created by the MCP example script."
    })
    if result.get("success"):
        print(f"File created: {result['filePath']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Example 4: Read the file
    print("\n4. Reading the file:")
    result = run_tool("readFile", {"filePath": test_file})
    if "content" in result:
        print(f"Content: {result['content']}")
        print(f"Total lines: {result['totalLines']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Example 5: Delete the file
    print("\n5. Deleting the file:")
    result = run_tool("deleteFile", {"filePath": test_file})
    if result.get("success"):
        print(f"File deleted: {result['filePath']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Verify the file was deleted
    print("\n6. Verifying the file was deleted:")
    if not os.path.exists(test_file):
        print(f"File {test_file} no longer exists.")
    else:
        print(f"File {test_file} still exists!")

if __name__ == "__main__":
    main()

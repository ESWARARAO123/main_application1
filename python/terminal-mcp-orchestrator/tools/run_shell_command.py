#!/usr/bin/env python3
"""
runShellCommand - Executes a command in the system shell
"""

import subprocess
import json
import shlex
import os
import platform
from typing import Dict, Any

# Tool information
TOOL_NAME = "runShellCommand"
DESCRIPTION = "Run a terminal command in the system shell"
PARAMETERS = {
    "command": {
        "type": "string",
        "description": "The shell command to execute",
        "required": True
    },
    "workingDir": {
        "type": "string",
        "description": "Working directory for command execution (default: current directory)",
        "required": False
    },
    "timeout": {
        "type": "number",
        "description": "Maximum execution time in seconds (default: 60)",
        "required": False
    },
    "shell": {
        "type": "boolean",
        "description": "Whether to execute through the shell (default: true)",
        "required": False
    }
}

def execute(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a shell command
    
    Args:
        parameters: Dictionary containing:
            - command: The command to execute
            - workingDir: Working directory (optional)
            - timeout: Maximum execution time in seconds (optional)
            - shell: Whether to use shell (optional)
            
    Returns:
        Dictionary containing:
            - stdout: Command standard output
            - stderr: Command standard error
            - exitCode: Command exit code
            - error: Error message if execution failed
    """
    try:
        # Extract parameters
        command = parameters.get("command")
        if not command:
            return {"error": "Command parameter is required"}
        
        working_dir = parameters.get("workingDir", os.getcwd())
        timeout = parameters.get("timeout", 60)
        use_shell = parameters.get("shell", True)
        
        # Prepare command
        if platform.system() == "Windows" and not use_shell:
            # On Windows, we need to handle command splitting differently
            cmd_args = command
        else:
            # On Unix or when using shell=True
            cmd_args = command
        
        # Execute the command
        process = subprocess.run(
            cmd_args,
            shell=use_shell,
            cwd=working_dir,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        
        # Return the results
        return {
            "stdout": process.stdout,
            "stderr": process.stderr,
            "exitCode": process.returncode
        }
    
    except subprocess.TimeoutExpired:
        return {
            "error": f"Command timed out after {timeout} seconds",
            "exitCode": -1
        }
    except Exception as e:
        return {
            "error": f"Error executing command: {str(e)}",
            "exitCode": -1
        }

# For testing directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
        result = execute({"command": command})
    else:
        # Default test command
        if platform.system() == "Windows":
            result = execute({"command": "dir"})
        else:
            result = execute({"command": "ls -la"})
    
    print(json.dumps(result, indent=2)) 
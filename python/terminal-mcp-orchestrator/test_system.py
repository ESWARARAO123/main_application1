#!/usr/bin/env python3
"""
Test System - Tests the MCP tools orchestrator with various tools

This script will test the orchestrator with various tools to ensure
that the entire system is working correctly.
"""

import json
import os
import platform
import tempfile
import subprocess
from typing import Dict, Any

def run_test(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Run a test with the orchestrator"""
    params_json = json.dumps(parameters)
    
    print(f"\nTesting: {tool_name}")
    print(f"Parameters: {params_json}")
    
    process = subprocess.run(
        ["python", "orchestrator.py", tool_name, params_json],
        capture_output=True,
        text=True
    )
    
    print(f"Exit code: {process.returncode}")
    
    if process.returncode != 0:
        print(f"Error: {process.stderr}")
        return {"error": process.stderr}
    
    try:
        result = json.loads(process.stdout)
        print(f"Result: {json.dumps(result, indent=2)}")
        return result
    except json.JSONDecodeError:
        print(f"Output (not JSON): {process.stdout}")
        return {"error": "Invalid JSON output", "output": process.stdout}

def test_read_directory():
    """Test the readDirectory tool"""
    return run_test("readDirectory", {"dirPath": "."})

def test_run_shell_command():
    """Test the runShellCommand tool"""
    command = "dir" if platform.system() == "Windows" else "ls -la"
    return run_test("runShellCommand", {"command": command})

def test_create_and_read_file():
    """Test creating and reading a file"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp_name = tmp.name
    
    try:
        # Create the file
        content = "Hello, world! This is a test file."
        create_result = run_test("createFile", {
            "filePath": tmp_name,
            "content": content,
            "overwrite": True
        })
        
        if "error" in create_result:
            return create_result
        
        # Read the file
        read_result = run_test("readFile", {
            "filePath": tmp_name
        })
        
        return {
            "create_result": create_result,
            "read_result": read_result,
            "content_match": read_result.get("content") == content
        }
    
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

def main():
    """Run all tests"""
    tests = [
        ("Read Directory", test_read_directory),
        ("Run Shell Command", test_run_shell_command),
        ("Create and Read File", test_create_and_read_file)
    ]
    
    results = {}
    
    print("=" * 60)
    print("MCP Tools Orchestrator System Test")
    print("=" * 60)
    
    for name, test_func in tests:
        print(f"\n{'-' * 60}")
        print(f"Test: {name}")
        print(f"{'-' * 60}")
        
        try:
            result = test_func()
            success = "error" not in result
            results[name] = {
                "success": success,
                "result": result
            }
            print(f"Result: {'Success' if success else 'Failed'}")
        except Exception as e:
            print(f"Exception: {str(e)}")
            results[name] = {
                "success": False,
                "error": str(e)
            }
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    successful = 0
    total = len(tests)
    
    for name, result in results.items():
        status = "✓ Passed" if result["success"] else "✗ Failed"
        print(f"{name}: {status}")
        if result["success"]:
            successful += 1
    
    print(f"\nTotal: {successful}/{total} tests passed")
    
    # Return non-zero exit code if any tests failed
    return 0 if successful == total else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 
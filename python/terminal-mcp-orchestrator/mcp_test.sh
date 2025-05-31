#!/bin/bash
# MCP Test Script - Test MCP integration with different tools

# Default MCP server URL
DEFAULT_SERVER="http://172.16.16.54:8080"

# Check if we're on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    python="python"
else
    python="python3"
fi

# Function to display help
show_help() {
    echo "MCP Test Script - Test MCP integration with different tools"
    echo ""
    echo "Usage: $0 [options] [command]"
    echo ""
    echo "Options:"
    echo "  -s, --server URL   MCP server URL (default: $DEFAULT_SERVER)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Commands:"
    echo "  test               Run the full test suite"
    echo "  read [path]        Test reading a directory (default: current dir)"
    echo "  cmd [command]      Test running a shell command"
    echo "  file               Test file creation and reading"
    echo "  tools              List available tools"
    echo ""
    echo "Examples:"
    echo "  $0 test"
    echo "  $0 --server http://localhost:8080 read /home"
    echo "  $0 cmd \"echo Hello, MCP!\""
}

# Parse arguments
server="$DEFAULT_SERVER"
command=""
arg=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--server)
            server="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        test|read|cmd|file|tools)
            command="$1"
            if [[ $# -gt 1 && "$2" != -* ]]; then
                arg="$2"
                shift 2
            else
                shift 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute command
case "$command" in
    test)
        echo "Running full test suite against MCP server: $server"
        $python test_mcp_server.py "$server"
        ;;
    read)
        if [[ -z "$arg" ]]; then
            arg="."
        fi
        echo "Reading directory: $arg on server: $server"
        $python orchestrator.py --server "$server" readDirectory "{\"dirPath\": \"$arg\"}"
        ;;
    cmd)
        if [[ -z "$arg" ]]; then
            arg="echo Hello from MCP!"
        fi
        echo "Running command: $arg on server: $server"
        $python orchestrator.py --server "$server" runShellCommand "{\"command\": \"$arg\"}"
        ;;
    file)
        echo "Testing file operations on server: $server"
        timestamp=$(date +%s)
        file_path="test_file_$timestamp.txt"
        content="Hello from MCP test script! Created at $(date)"

        echo "Creating file: $file_path"
        $python orchestrator.py --server "$server" createFile "{\"filePath\": \"$file_path\", \"content\": \"$content\", \"overwrite\": true}"

        echo ""
        echo "Reading file: $file_path"
        $python orchestrator.py --server "$server" readFile "{\"filePath\": \"$file_path\"}"

        echo ""
        echo "Deleting file: $file_path"
        $python orchestrator.py --server "$server" deleteFile "{\"filePath\": \"$file_path\"}"
        ;;
    tools)
        echo "Listing available tools on server: $server"
        $python orchestrator.py --server "$server" --list
        ;;
    # No AI agent connector anymore
        ;;
    *)
        show_help
        exit 1
        ;;
esac
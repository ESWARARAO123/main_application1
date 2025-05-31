# MCP Test Script - Test MCP integration with different tools
# PowerShell version for Windows users

# Default MCP server URL
$DEFAULT_SERVER = "http://172.16.16.54:8080"

# Function to display help
function Show-Help {
    Write-Host "MCP Test Script - Test MCP integration with different tools"
    Write-Host ""
    Write-Host "Usage: .\mcp_test.ps1 [options] [command]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Server URL      MCP server URL (default: $DEFAULT_SERVER)"
    Write-Host "  -Help            Show this help message"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  test             Run the full test suite"
    Write-Host "  read [path]      Test reading a directory (default: current dir)"
    Write-Host "  cmd [command]    Test running a shell command"
    Write-Host "  file             Test file creation and reading"
    Write-Host "  tools            List available tools"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\mcp_test.ps1 test"
    Write-Host "  .\mcp_test.ps1 -Server http://localhost:8080 read /home"
    Write-Host "  .\mcp_test.ps1 cmd 'echo Hello, MCP!'"
}

# Parse arguments
param (
    [string]$Server = $DEFAULT_SERVER,
    [switch]$Help,
    [Parameter(Position = 0)]
    [string]$Command,
    [Parameter(Position = 1)]
    [string]$Arg
)

if ($Help) {
    Show-Help
    exit 0
}

if (-not $Command) {
    Show-Help
    exit 1
}

# Execute command
switch ($Command) {
    "test" {
        Write-Host "Running full test suite against MCP server: $Server"
        python test_mcp_server.py $Server
    }
    "read" {
        if (-not $Arg) {
            $Arg = "."
        }
        Write-Host "Reading directory: $Arg on server: $Server"
        python orchestrator.py --server $Server readDirectory "{""dirPath"": ""$Arg""}"
    }
    "cmd" {
        if (-not $Arg) {
            $Arg = "echo Hello from MCP!"
        }
        Write-Host "Running command: $Arg on server: $Server"
        python orchestrator.py --server $Server runShellCommand "{""command"": ""$Arg""}"
    }
    "file" {
        Write-Host "Testing file operations on server: $Server"
        $timestamp = [int][double]::Parse((Get-Date -UFormat %s))
        $file_path = "test_file_$timestamp.txt"
        $content = "Hello from MCP test script! Created at $(Get-Date)"

        Write-Host "Creating file: $file_path"
        python orchestrator.py --server $Server createFile "{""filePath"": ""$file_path"", ""content"": ""$content"", ""overwrite"": true}"

        Write-Host ""
        Write-Host "Reading file: $file_path"
        python orchestrator.py --server $Server readFile "{""filePath"": ""$file_path""}"

        Write-Host ""
        Write-Host "Deleting file: $file_path"
        python orchestrator.py --server $Server deleteFile "{""filePath"": ""$file_path""}"
    }
    "tools" {
        Write-Host "Listing available tools on server: $Server"
        python orchestrator.py --server $Server --list
    }
    # AI agent connector has been removed
    default {
        Write-Host "Unknown command: $Command"
        Show-Help
        exit 1
    }
}
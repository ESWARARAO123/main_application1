# MCP Toolkit Script
# A script to test various MCP tools using direct HTTP requests

# Configuration
$baseUrl = "http://172.16.16.54:8080"
$clientId = "client-" + (Get-Random)
$headers = @{
    "Content-Type" = "application/json"
}

function Invoke-McpTool {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ToolName,
        
        [Parameter(Mandatory=$false)]
        [hashtable]$Parameters = @{}
    )
    
    $messageId = "msg-" + (Get-Random)
    
    $body = @{
        id = $messageId
        type = "invoke_tool"
        content = @{
            name = $ToolName
            parameters = $Parameters
        }
        clientId = $clientId
    } | ConvertTo-Json -Depth 10
    
    Write-Host "Invoking tool '$ToolName' with message ID: $messageId" -ForegroundColor Cyan
    
    try {
        $result = Invoke-RestMethod -Uri "$baseUrl/messages" -Method Post -Headers $headers -Body $body
        Write-Host "Tool invocation successful!" -ForegroundColor Green
        return $result
    } catch {
        Write-Host "Error invoking tool: $_" -ForegroundColor Red
        return $null
    }
}

# Display welcome message
Write-Host "MCP Toolkit - Testing MCP tools on $baseUrl" -ForegroundColor Yellow
Write-Host "Client ID: $clientId" -ForegroundColor Yellow
Write-Host ""

# Menu of available tools
function Show-ToolMenu {
    Write-Host "Available Tools:" -ForegroundColor Magenta
    Write-Host "1. Read Directory"
    Write-Host "2. Create File"
    Write-Host "3. Read File"
    Write-Host "4. Run Shell Command"
    Write-Host "5. Get Directory Tree"
    Write-Host "6. Edit File"
    Write-Host "7. Exit"
    Write-Host ""
    
    $choice = Read-Host "Select a tool to test (1-7)"
    return $choice
}

# Main program loop
$running = $true
while ($running) {
    $choice = Show-ToolMenu
    
    switch ($choice) {
        "1" {
            # Read Directory
            $dirPath = Read-Host "Enter directory path (leave empty for root)"
            $params = @{}
            if ($dirPath) { $params["dirPath"] = $dirPath }
            
            $result = Invoke-McpTool -ToolName "readDirectory" -Parameters $params
            $result | ConvertTo-Json -Depth 10
        }
        "2" {
            # Create File
            $filePath = Read-Host "Enter file path"
            $content = Read-Host "Enter file content"
            
            $params = @{
                filePath = $filePath
                content = $content
            }
            
            $result = Invoke-McpTool -ToolName "createFile" -Parameters $params
            $result | ConvertTo-Json -Depth 10
        }
        "3" {
            # Read File
            $filePath = Read-Host "Enter file path"
            
            $params = @{
                filePath = $filePath
            }
            
            $result = Invoke-McpTool -ToolName "readFile" -Parameters $params
            $result | ConvertTo-Json -Depth 10
        }
        "4" {
            # Run Shell Command
            $command = Read-Host "Enter shell command"
            
            $params = @{
                command = $command
            }
            
            $result = Invoke-McpTool -ToolName "runShellCommand" -Parameters $params
            $result | ConvertTo-Json -Depth 10
        }
        "5" {
            # Get Directory Tree
            $dirPath = Read-Host "Enter directory path"
            $maxDepth = Read-Host "Enter max depth (leave empty for unlimited)"
            
            $params = @{
                dirPath = $dirPath
            }
            
            if ($maxDepth) { $params["maxDepth"] = [int]$maxDepth }
            
            $result = Invoke-McpTool -ToolName "getDirectoryTree" -Parameters $params
            $result | ConvertTo-Json -Depth 10
        }
        "6" {
            # Edit File
            $filePath = Read-Host "Enter file path"
            $operation = Read-Host "Enter operation (append, prepend, replace, insert)"
            $content = Read-Host "Enter content to add"
            
            $params = @{
                filePath = $filePath
                operation = $operation
                content = $content
            }
            
            if ($operation -eq "insert") {
                $lineNumber = Read-Host "Enter line number"
                $params["lineNumber"] = [int]$lineNumber
            }
            elseif ($operation -eq "replace") {
                $startLine = Read-Host "Enter start line"
                $endLine = Read-Host "Enter end line"
                $params["startLine"] = [int]$startLine
                $params["endLine"] = [int]$endLine
            }
            
            $result = Invoke-McpTool -ToolName "editFile" -Parameters $params
            $result | ConvertTo-Json -Depth 10
        }
        "7" {
            # Exit
            $running = $false
            Write-Host "Exiting MCP Toolkit" -ForegroundColor Yellow
        }
        default {
            Write-Host "Invalid choice, please try again" -ForegroundColor Red
        }
    }
    
    if ($running) {
        Write-Host ""
        Read-Host "Press Enter to continue"
        Clear-Host
    }
} 
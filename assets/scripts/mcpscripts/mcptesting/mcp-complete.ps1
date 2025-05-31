# Complete MCP Client Solution
# This script establishes an SSE connection first and then invokes tools

# Configuration
$baseUrl = "http://172.16.16.54:8080"
$headers = @{
    "Content-Type" = "application/json"
}

# First, establish SSE connection to get a clientId
Write-Host "Establishing SSE connection..." -ForegroundColor Cyan

# We need to use a background job to handle the SSE connection while keeping the script running
$sseJob = Start-Job -ScriptBlock {
    param($baseUrl)
    
    # This function simulates an SSE connection and returns a clientId
    function ConnectToSSE {
        $request = [System.Net.WebRequest]::Create("$baseUrl/sse")
        $request.Timeout = 5000  # 5 second timeout
        
        try {
            $response = $request.GetResponse()
            $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
            
            # Read the first data line to get the clientId
            $line = $reader.ReadLine()
            
            if ($line -match 'data: (.+)') {
                $data = $matches[1] | ConvertFrom-Json
                return $data.clientId
            }
        } catch {
            Write-Error "Failed to connect to SSE: $_"
            return $null
        }
    }
    
    # Connect and return the clientId
    ConnectToSSE $baseUrl
} -ArgumentList $baseUrl

# Wait a moment for the job to start
Start-Sleep -Seconds 2

# Get the clientId from the job result
$clientId = Receive-Job -Job $sseJob -ErrorAction SilentlyContinue
Remove-Job -Job $sseJob -Force

if (-not $clientId) {
    Write-Host "Failed to get a clientId. Using a fallback method..." -ForegroundColor Yellow
    
    # As a fallback (since we can't keep an SSE connection open indefinitely in a script),
    # we'll try to create a connection directly and parse the response
    try {
        $wr = Invoke-WebRequest -Uri "$baseUrl/sse" -TimeoutSec 5 -ErrorAction Stop
        
        # Try to extract clientId from response content
        if ($wr.Content -match 'data: ({.*})') {
            $data = $matches[1] | ConvertFrom-Json
            $clientId = $data.clientId
        }
    } catch {
        Write-Host "Fallback method failed. Using a hardcoded clientId as last resort." -ForegroundColor Red
        # As last resort, use a hardcoded clientId that may be active
        $clientId = "1747810889393"  # This was the clientId we received earlier
    }
}

Write-Host "Using clientId: $clientId" -ForegroundColor Green

# Function to invoke an MCP tool
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
    
    Write-Host "Invoking tool '$ToolName'..." -ForegroundColor Cyan
    
    try {
        $result = Invoke-RestMethod -Uri "$baseUrl/messages" -Method Post -Headers $headers -Body $body
        Write-Host "Tool execution successful!" -ForegroundColor Green
        return $result
    } catch {
        Write-Host "Error invoking tool: $_" -ForegroundColor Red
        return $null
    }
}

# Test a few tools

# 1. Read Directory
Write-Host "`n--- Reading Directory ---" -ForegroundColor Magenta
$dirResult = Invoke-McpTool -ToolName "readDirectory"
$dirResult | ConvertTo-Json -Depth 10

# 2. Create a test file
Write-Host "`n--- Creating Test File ---" -ForegroundColor Magenta
$createResult = Invoke-McpTool -ToolName "createFile" -Parameters @{
    filePath = "mcp-test-file.txt"
    content = "This is a test file created by the MCP client script at $(Get-Date)"
}
$createResult | ConvertTo-Json -Depth 10

# 3. Read the test file
Write-Host "`n--- Reading Test File ---" -ForegroundColor Magenta
$readResult = Invoke-McpTool -ToolName "readFile" -Parameters @{
    filePath = "mcp-test-file.txt"
}
$readResult | ConvertTo-Json -Depth 10

# 4. Run a shell command
Write-Host "`n--- Running Shell Command ---" -ForegroundColor Magenta
$shellResult = Invoke-McpTool -ToolName "runShellCommand" -Parameters @{
    command = "dir"
}
$shellResult | ConvertTo-Json -Depth 10

Write-Host "`nMCP testing complete!" -ForegroundColor Yellow 
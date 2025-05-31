# MCP AI Agent Integration Test Script
# This script tests the specific flow needed for the AI Agent integration

$mcpServer = "172.16.16.54"
$mcpPort = "8080"
$baseUrl = "http://${mcpServer}:${mcpPort}"

Write-Host "===== MCP AI Agent Integration Tests =====" -ForegroundColor Cyan
Write-Host "Testing against MCP server at $baseUrl" -ForegroundColor Yellow

# Function to output JSON in a more readable format
function Format-Json {
    param([Parameter(Mandatory, ValueFromPipeline)][String] $json)
    $indent = 0
    ($json -Replace '(?m)^\s*(\}|\])',"`n`${1}" -Replace '(?m)^(\{|\[)',"`${1}`n" -Replace '(?m)(\{|\[|,)$',"`${1}`n" -Split "`n" |
    ForEach-Object {
        if ($_ -Match '[\}\]]') { $indent-- }
        $line = (' ' * $indent * 2) + $_.TrimStart().Replace(':  ', ': ')
        if ($_ -Match '[\{\[]') { $indent++ }
        $line
    }) -Join "`n"
}

# Step 1: Get server information
Write-Host "`n===== Step 1: Getting MCP Server Info =====" -ForegroundColor Green
$infoResult = Invoke-RestMethod -Uri "$baseUrl/info" -Method GET
Write-Host "Server Info:" -ForegroundColor Yellow
$infoResult | ConvertTo-Json | Format-Json

# Step 2: Get available tools
Write-Host "`n===== Step 2: Getting Available Tools =====" -ForegroundColor Green
$toolsResult = Invoke-RestMethod -Uri "$baseUrl/tools" -Method GET
Write-Host "Found $($toolsResult.count) tools" -ForegroundColor Yellow
$toolsResult.tools | Select-Object -First 3 -Property name, description | Format-Table

# Step 3: Try to connect to the SSE endpoint
Write-Host "`n===== Step 3: Testing SSE Connection =====" -ForegroundColor Green
Write-Host "SSE connections require a long-running operation. Testing with limited timeout."
Write-Host "Run this command separately for proper testing:"
Write-Host "curl -N $baseUrl/sse" -ForegroundColor Magenta

try {
    $headers = @{
        'Accept' = 'text/event-stream'
        'Cache-Control' = 'no-cache'
    }
    $cts = New-Object System.Threading.CancellationTokenSource
    $cts.CancelAfter(2000) # Cancel after 2 seconds
    
    $response = Invoke-WebRequest -Uri "$baseUrl/sse" -Headers $headers -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
    Write-Host "SSE Response headers:" -ForegroundColor Yellow
    $response.Headers | Format-Table -AutoSize
} catch {
    # Expected to fail due to timeout
    Write-Host "SSE connection test complete (timeout expected)" -ForegroundColor Yellow
}

# Step 4: Test various message formats to find the correct one

# Test Format 1: Tool and parameters only
Write-Host "`n===== Test Format 1: Tool and parameters only =====" -ForegroundColor Green
$format1 = @{
    tool = "readDirectory"
    parameters = @{}
} | ConvertTo-Json

try {
    $result1 = Invoke-RestMethod -Uri "$baseUrl/messages" -Method POST -Body $format1 -ContentType "application/json" -ErrorAction SilentlyContinue
    Write-Host "Result:" -ForegroundColor Yellow
    $result1 | ConvertTo-Json | Format-Json
} catch {
    Write-Host "Error:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    Write-Host $_.ErrorDetails.Message
}

# Test Format 2: With ClientID
Write-Host "`n===== Test Format 2: With ClientID =====" -ForegroundColor Green
$format2 = @{
    clientId = "test-agent-123"
    tool = "readDirectory"
    parameters = @{}
} | ConvertTo-Json

try {
    $result2 = Invoke-RestMethod -Uri "$baseUrl/messages" -Method POST -Body $format2 -ContentType "application/json" -ErrorAction SilentlyContinue
    Write-Host "Result:" -ForegroundColor Yellow
    $result2 | ConvertTo-Json | Format-Json
} catch {
    Write-Host "Error:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    Write-Host $_.ErrorDetails.Message
}

# Test Format 3: Claude-style format
Write-Host "`n===== Test Format 3: Claude-style format =====" -ForegroundColor Green
$format3 = @{
    clientId = "test-agent-123"
    type = "invoke"
    invoke = @{
        name = "readDirectory"
        parameters = @{}
    }
} | ConvertTo-Json -Depth 3

try {
    $result3 = Invoke-RestMethod -Uri "$baseUrl/messages" -Method POST -Body $format3 -ContentType "application/json" -ErrorAction SilentlyContinue
    Write-Host "Result:" -ForegroundColor Yellow
    $result3 | ConvertTo-Json | Format-Json
} catch {
    Write-Host "Error:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    Write-Host $_.ErrorDetails.Message
}

# Test Format 4: Natural language format
Write-Host "`n===== Test Format 4: Natural language format =====" -ForegroundColor Green
$format4 = @{
    clientId = "test-agent-123"
    message = "list all files in the current directory"
} | ConvertTo-Json

try {
    $result4 = Invoke-RestMethod -Uri "$baseUrl/messages" -Method POST -Body $format4 -ContentType "application/json" -ErrorAction SilentlyContinue
    Write-Host "Result:" -ForegroundColor Yellow
    $result4 | ConvertTo-Json | Format-Json
} catch {
    Write-Host "Error:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    Write-Host $_.ErrorDetails.Message
}

# Summary
Write-Host "`n===== Integration Test Summary =====" -ForegroundColor Green
Write-Host "1. The MCP server at $baseUrl provides server info and tools list" -ForegroundColor White
Write-Host "2. SSE connection is required to get a valid clientId" -ForegroundColor White
Write-Host "3. Need to test various message formats with a valid clientId from SSE" -ForegroundColor White
Write-Host "4. For a complete test, establish an SSE connection in a separate terminal:" -ForegroundColor White
Write-Host "   curl -N $baseUrl/sse" -ForegroundColor Magenta

Write-Host "`nThese tests provide insight into how to structure the MCP AI Agent." -ForegroundColor Cyan
Write-Host "For further testing, check the clientId from an SSE connection and modify the scripts." -ForegroundColor Cyan 
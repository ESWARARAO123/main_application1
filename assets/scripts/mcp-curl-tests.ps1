# MCP Server Curl Testing Script
# This script tests various MCP endpoints using curl commands

$mcpServer = "172.16.16.54"
$mcpPort = "8080"
$baseUrl = "http://${mcpServer}:${mcpPort}"

Write-Host "===== MCP Server Testing Script =====" -ForegroundColor Cyan
Write-Host "Testing MCP server at $baseUrl" -ForegroundColor Yellow

# 1. Test the info endpoint
Write-Host "`n===== Testing /info endpoint =====" -ForegroundColor Green
Write-Host "curl -Method GET $baseUrl/info" -ForegroundColor Gray
curl -Method GET "$baseUrl/info"

# 2. Test the tools endpoint
Write-Host "`n===== Testing /tools endpoint =====" -ForegroundColor Green
Write-Host "curl -Method GET $baseUrl/tools" -ForegroundColor Gray
curl -Method GET "$baseUrl/tools"

# 3. Connect to SSE endpoint (Echo the command but don't execute as it will hang the script)
Write-Host "`n===== SSE Connection Command (run separately) =====" -ForegroundColor Green
Write-Host "curl -Method GET $baseUrl/sse -Headers @{'Accept'='text/event-stream'}" -ForegroundColor Gray

# 4. Test a session creation (if this endpoint exists)
Write-Host "`n===== Testing /session endpoint =====" -ForegroundColor Green
$sessionBody = @{
    action = "create_session"
} | ConvertTo-Json
Write-Host "curl -Method POST $baseUrl/session -Body '$sessionBody' -ContentType 'application/json'" -ForegroundColor Gray
curl -Method POST "$baseUrl/session" -Body $sessionBody -ContentType "application/json"

# 5. Try to test the readDirectory command without clientId
Write-Host "`n===== Testing /messages with readDirectory (without clientId) =====" -ForegroundColor Green
$readDirBody = @{
    tool = "readDirectory"
    parameters = @{}
} | ConvertTo-Json
Write-Host "curl -Method POST $baseUrl/messages -Body '$readDirBody' -ContentType 'application/json'" -ForegroundColor Gray
curl -Method POST "$baseUrl/messages" -Body $readDirBody -ContentType "application/json"

# 6. Try to test the readDirectory command with a dummy clientId
Write-Host "`n===== Testing /messages with readDirectory (with dummy clientId) =====" -ForegroundColor Green
$readDirWithClientBody = @{
    clientId = "test-client-id"
    tool = "readDirectory"
    parameters = @{}
} | ConvertTo-Json
Write-Host "curl -Method POST $baseUrl/messages -Body '$readDirWithClientBody' -ContentType 'application/json'" -ForegroundColor Gray
curl -Method POST "$baseUrl/messages" -Body $readDirWithClientBody -ContentType "application/json"

# 7. Test executing a command using the invoke format
Write-Host "`n===== Testing /messages with invoke format =====" -ForegroundColor Green
$invokeBody = @{
    clientId = "test-client-id"
    type = "invoke"
    invoke = @{
        name = "readDirectory"
        parameters = @{}
    }
} | ConvertTo-Json -Depth 3
Write-Host "curl -Method POST $baseUrl/messages -Body '$invokeBody' -ContentType 'application/json'" -ForegroundColor Gray
curl -Method POST "$baseUrl/messages" -Body $invokeBody -ContentType "application/json"

# 8. Test the createFile command format
Write-Host "`n===== Testing /messages with createFile command =====" -ForegroundColor Green
$createFileBody = @{
    clientId = "test-client-id"
    tool = "createFile"
    parameters = @{
        filePath = "test-file.txt"
        content = "Hello from MCP test script!"
    }
} | ConvertTo-Json -Depth 3
Write-Host "curl -Method POST $baseUrl/messages -Body '$createFileBody' -ContentType 'application/json'" -ForegroundColor Gray
curl -Method POST "$baseUrl/messages" -Body $createFileBody -ContentType "application/json"

# 9. Test the readFile command format
Write-Host "`n===== Testing /messages with readFile command =====" -ForegroundColor Green
$readFileBody = @{
    clientId = "test-client-id"
    tool = "readFile"
    parameters = @{
        filePath = "test-file.txt"
    }
} | ConvertTo-Json -Depth 3
Write-Host "curl -Method POST $baseUrl/messages -Body '$readFileBody' -ContentType 'application/json'" -ForegroundColor Gray
curl -Method POST "$baseUrl/messages" -Body $readFileBody -ContentType "application/json"

# 10. Test the runShellCommand format
Write-Host "`n===== Testing /messages with runShellCommand =====" -ForegroundColor Green
$runCommandBody = @{
    clientId = "test-client-id"
    tool = "runShellCommand"
    parameters = @{
        command = "ls -la"
    }
} | ConvertTo-Json -Depth 3
Write-Host "curl -Method POST $baseUrl/messages -Body '$runCommandBody' -ContentType 'application/json'" -ForegroundColor Gray
curl -Method POST "$baseUrl/messages" -Body $runCommandBody -ContentType "application/json"

# 11. Test alternative message format
Write-Host "`n===== Testing alternative message format =====" -ForegroundColor Green
$alternativeBody = @{
    clientId = "test-client-id"
    message = "list files in current directory"
} | ConvertTo-Json
Write-Host "curl -Method POST $baseUrl/messages -Body '$alternativeBody' -ContentType 'application/json'" -ForegroundColor Gray
curl -Method POST "$baseUrl/messages" -Body $alternativeBody -ContentType "application/json"

Write-Host "`n===== Testing Completed =====" -ForegroundColor Cyan
Write-Host "Check the results above to understand MCP server API requirements" -ForegroundColor Yellow 
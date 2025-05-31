# Direct MCP Tool Execution
# This script directly invokes an MCP tool without interactive menus

$baseUrl = "http://172.16.16.54:8080"
$clientId = "direct-" + (Get-Random)
$headers = @{
    "Content-Type" = "application/json"
}

# Choose a simple tool to test - readDirectory
$toolName = "readDirectory"
$params = @{}  # No parameters needed for root directory

$messageId = "msg-" + (Get-Random)
$body = @{
    id = $messageId
    type = "invoke_tool"
    content = @{
        name = $toolName
        parameters = $params
    }
    clientId = $clientId
} | ConvertTo-Json -Depth 10

Write-Host "Testing MCP with tool: $toolName" -ForegroundColor Yellow
Write-Host "Using client ID: $clientId" -ForegroundColor Yellow
Write-Host "Sending request to: $baseUrl/messages" -ForegroundColor Yellow

try {
    $result = Invoke-RestMethod -Uri "$baseUrl/messages" -Method Post -Headers $headers -Body $body
    Write-Host "Request successful!" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Cyan
    $result | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error invoking tool: $_" -ForegroundColor Red
} 
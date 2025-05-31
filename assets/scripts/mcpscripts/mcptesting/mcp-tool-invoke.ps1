# MCP Tool Invocation Script
# This script invokes a tool directly using a predefined clientId

# Set a fixed clientId (this should match one obtained from a previous SSE connection)
$clientId = "1747810889393"  # Use the clientId we received earlier

$messageId = "msg-" + (Get-Random)
$toolName = "readDirectory"  # Change this to use other tools

$body = @{
    id = $messageId
    type = "invoke_tool"
    content = @{
        name = $toolName
        parameters = @{}
    }
    clientId = $clientId
} | ConvertTo-Json -Depth 10

$headers = @{
    "Content-Type" = "application/json"
}

Write-Host "Invoking tool '$toolName' with message ID: $messageId and client ID: $clientId"
try {
    $result = Invoke-RestMethod -Uri "http://172.16.16.54:8080/messages" -Method Post -Headers $headers -Body $body
    Write-Host "Tool invocation successful"
    $result | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error invoking tool: $_"
} 
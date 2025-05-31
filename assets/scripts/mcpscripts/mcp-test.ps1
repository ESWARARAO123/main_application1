# MCP Server Test Script
$mcpHost = "172.16.16.54"
$mcpPort = "8080"
$mcpBaseUrl = "http://${mcpHost}:${mcpPort}"

# Step 1: Get MCP Server Info
Write-Host "Getting MCP Server Info..." -ForegroundColor Cyan
$serverInfo = Invoke-RestMethod -Uri "${mcpBaseUrl}/info" -Method GET
Write-Host "MCP Server Info:" -ForegroundColor Green
$serverInfo | ConvertTo-Json -Depth 5

# Step 2: Prepare tools info for reference
Write-Host "`nGetting available tools..." -ForegroundColor Cyan
$toolsInfo = Invoke-RestMethod -Uri "${mcpBaseUrl}/tools" -Method GET
Write-Host "Found $($toolsInfo.count) tools available" -ForegroundColor Green

# Step 3: Create a client ID (this would typically come from SSE)
$clientId = [Guid]::NewGuid().ToString()
Write-Host "`nCreated client ID: $clientId" -ForegroundColor Cyan

# Step 4: Test the readDirectory tool
Write-Host "`nTesting readDirectory tool..." -ForegroundColor Cyan
$body = @{
    clientId = $clientId
    tool = "readDirectory"
    parameters = @{}
} | ConvertTo-Json

try {
    $result = Invoke-RestMethod -Uri "${mcpBaseUrl}/messages" -Method POST -Body $body -ContentType "application/json"
    Write-Host "Directory listing:" -ForegroundColor Green
    $result | ConvertTo-Json -Depth 5
} catch {
    Write-Host "Error executing readDirectory:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    
    # Let's try with a new session-based authentication approach
    Write-Host "`nTrying with session-based approach..." -ForegroundColor Yellow
    
    # Create a session first
    $sessionBody = @{
        action = "create_session"
    } | ConvertTo-Json
    
    try {
        $sessionResult = Invoke-RestMethod -Uri "${mcpBaseUrl}/session" -Method POST -Body $sessionBody -ContentType "application/json"
        Write-Host "Session created with ID: $($sessionResult.sessionId)" -ForegroundColor Green
        
        # Now use the session ID for commands
        $body2 = @{
            sessionId = $sessionResult.sessionId
            tool = "readDirectory"
            parameters = @{}
        } | ConvertTo-Json
        
        $result2 = Invoke-RestMethod -Uri "${mcpBaseUrl}/messages" -Method POST -Body $body2 -ContentType "application/json"
        Write-Host "Directory listing:" -ForegroundColor Green
        $result2 | ConvertTo-Json -Depth 5
    } catch {
        Write-Host "Error with session approach:" -ForegroundColor Red
        Write-Host $_.Exception.Message
    }
} 
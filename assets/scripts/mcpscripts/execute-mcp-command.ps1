# MCP Server Configuration
$MCP_HOST = "172.16.16.54"
$MCP_PORT = "8080"

# First, establish an SSE connection to get a clientId
Write-Host "Connecting to MCP server at $MCP_HOST`:$MCP_PORT..."

# Get server info
try {
    $serverInfo = Invoke-RestMethod -Uri "http://$MCP_HOST`:$MCP_PORT/info" -Method Get
    Write-Host "Connected to MCP server:"
    Write-Host "  Name: $($serverInfo.name)"
    Write-Host "  Version: $($serverInfo.version)"
    Write-Host "  Platform: $($serverInfo.platform)"
    Write-Host "  Workspace: $($serverInfo.workspace)"
    Write-Host "  Active Sessions: $($serverInfo.activeSessions)"
} catch {
    Write-Host "Error connecting to MCP server: $_"
    exit 1
}

# Get available tools
try {
    $toolsResponse = Invoke-RestMethod -Uri "http://$MCP_HOST`:$MCP_PORT/tools" -Method Get
    $tools = $toolsResponse.tools
    
    Write-Host "`nAvailable Tools ($($toolsResponse.count)):"
    foreach ($tool in $tools) {
        Write-Host "  - $($tool.name): $($tool.description)"
    }
} catch {
    Write-Host "Error fetching tools: $_"
    exit 1
}

# Now we need to establish an SSE connection to get a clientId
# Since PowerShell doesn't have built-in SSE support, we'll use a workaround
# We'll create a temporary HTML file that connects to the SSE endpoint and displays the clientId

$tempHtmlPath = [System.IO.Path]::GetTempFileName() + ".html"
$html = @"
<!DOCTYPE html>
<html>
<head>
    <title>MCP SSE Connection</title>
</head>
<body>
    <h1>MCP SSE Connection</h1>
    <div id="status">Connecting...</div>
    <div id="clientId"></div>
    <script>
        const eventSource = new EventSource('http://$MCP_HOST`:$MCP_PORT/sse');
        
        eventSource.onopen = function() {
            document.getElementById('status').textContent = 'Connected to SSE';
        };
        
        eventSource.onerror = function(error) {
            document.getElementById('status').textContent = 'Error: ' + error;
        };
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                console.log('SSE message received:', data);
                
                if (data.type === 'connected' && data.clientId) {
                    document.getElementById('clientId').textContent = 'Client ID: ' + data.clientId;
                    document.title = 'MCP Client ID: ' + data.clientId;
                }
            } catch (error) {
                console.error('Error parsing SSE event:', error);
            }
        };
    </script>
</body>
</html>
"@

Set-Content -Path $tempHtmlPath -Value $html
Write-Host "`nOpening SSE connection in browser. Please note the Client ID that appears and enter it below."
Start-Process $tempHtmlPath

# Ask the user for the clientId
$clientId = Read-Host "`nEnter the Client ID from the browser window"

if (-not $clientId) {
    Write-Host "No Client ID provided. Exiting."
    exit 1
}

Write-Host "Using Client ID: $clientId"

# Now let's execute a command
$toolName = Read-Host "`nEnter the name of the tool to execute (e.g., runShellCommand)"

if (-not $toolName) {
    Write-Host "No tool name provided. Exiting."
    exit 1
}

# Find the tool to get its parameters
$selectedTool = $tools | Where-Object { $_.name -eq $toolName }

if (-not $selectedTool) {
    Write-Host "Tool '$toolName' not found. Please choose from the available tools."
    exit 1
}

# Collect parameters
$parameters = @{}
if ($selectedTool.parameters) {
    Write-Host "`nEnter parameters for $toolName:"
    foreach ($paramName in $selectedTool.parameters.PSObject.Properties.Name) {
        $paramInfo = $selectedTool.parameters.$paramName
        $required = $paramInfo.required -eq $true ? " (required)" : ""
        $paramValue = Read-Host "  $paramName$required"
        
        if ($paramValue) {
            $parameters[$paramName] = $paramValue
        }
    }
}

# Prepare the payload
$payload = @{
    clientId = $clientId
    tool = $toolName
    parameters = $parameters
}

# Execute the command
try {
    Write-Host "`nExecuting $toolName..."
    $result = Invoke-RestMethod -Uri "http://$MCP_HOST`:$MCP_PORT/messages" -Method Post -Body ($payload | ConvertTo-Json) -ContentType "application/json"
    
    Write-Host "`nCommand Result:"
    if ($result.content) {
        foreach ($item in $result.content) {
            if ($item.type -eq "text") {
                Write-Host $item.text
            } else {
                $item | ConvertTo-Json -Depth 10
            }
        }
    } else {
        $result | ConvertTo-Json -Depth 10
    }
} catch {
    Write-Host "Error executing command: $_"
    exit 1
}

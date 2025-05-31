# MCP Client Script

# First get a client ID by connecting to the SSE endpoint
$clientId = $null
$request = [System.Net.WebRequest]::Create("http://172.16.16.54:8080/sse")
$request.Method = "GET"

try {
    $response = $request.GetResponse()
    $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
    
    # Get the first line which should contain our client ID
    $line = $reader.ReadLine()
    if ($line -match 'data: (.+)') {
        $data = $matches[1] | ConvertFrom-Json
        $clientId = $data.clientId
        Write-Host "Connected with client ID: $clientId"
    }
} catch {
    Write-Host "Error connecting to SSE: $_"
    exit
}

# Now use the client ID to invoke a tool
if ($clientId) {
    $messageId = "msg-" + (Get-Random)
    $body = @{
        id = $messageId
        type = "invoke_tool"
        content = @{
            name = "readDirectory"
            parameters = @{}
        }
        clientId = $clientId
    } | ConvertTo-Json -Depth 10

    $headers = @{
        "Content-Type" = "application/json"
    }

    Write-Host "Invoking tool with message ID: $messageId"
    $result = Invoke-RestMethod -Uri "http://172.16.16.54:8080/messages" -Method Post -Headers $headers -Body $body
    $result | ConvertTo-Json -Depth 10
} 
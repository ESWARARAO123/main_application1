// Import required modules
const http = require('http');
const https = require('https');
const EventSource = require('eventsource');
const fs = require('fs');

// MCP Server configuration
const MCP_HOST = '172.16.16.54';
const MCP_PORT = '8080';
const MCP_BASE_URL = `http://${MCP_HOST}:${MCP_PORT}`;

// Create a log file for outputs
const logStream = fs.createWriteStream('./mcp-test-logs.txt', { flags: 'a' });
function log(message) {
  const timestamp = new Date().toISOString();
  const formattedMessage = `[${timestamp}] ${message}`;
  console.log(formattedMessage);
  logStream.write(formattedMessage + '\n');
}

// Utility function to make HTTP requests
function makeRequest(url, method = 'GET', data = null) {
  return new Promise((resolve, reject) => {
    const options = {
      method: method,
      headers: {
        'Content-Type': 'application/json',
      },
    };

    log(`Making ${method} request to ${url}`);
    if (data) {
      log(`Request data: ${JSON.stringify(data, null, 2)}`);
    }

    const req = http.request(url, options, (res) => {
      let responseData = '';

      res.on('data', (chunk) => {
        responseData += chunk;
      });

      res.on('end', () => {
        log(`Response status: ${res.statusCode}`);
        try {
          const parsedData = JSON.parse(responseData);
          log(`Response data: ${JSON.stringify(parsedData, null, 2)}`);
          resolve(parsedData);
        } catch (e) {
          log(`Response is not JSON: ${responseData}`);
          resolve(responseData); // Return as string if not JSON
        }
      });
    });

    req.on('error', (error) => {
      log(`Request error: ${error.message}`);
      reject(error);
    });

    if (data) {
      req.write(JSON.stringify(data));
    }

    req.end();
  });
}

// Main function to orchestrate the MCP interactions
async function testMCPServer() {
  try {
    // Step 1: Get server info
    log('Getting MCP server info...');
    const serverInfo = await makeRequest(`${MCP_BASE_URL}/info`);
    log(`MCP Server Info: ${JSON.stringify(serverInfo, null, 2)}`);

    // Step 2: Get available tools
    log('\nGetting available tools...');
    const toolsInfo = await makeRequest(`${MCP_BASE_URL}/tools`);
    log(`Found ${toolsInfo.count} tools available`);
    
    // Log the first few tools with their descriptions
    if (toolsInfo.tools && toolsInfo.tools.length > 0) {
      log(`Sample tools available:`);
      toolsInfo.tools.slice(0, 3).forEach(tool => {
        log(`- ${tool.name}: ${tool.description}`);
      });
    }

    // Step 3: Establish SSE connection
    log('\nEstablishing SSE connection...');
    const eventSource = new EventSource(`${MCP_BASE_URL}/sse`);
    
    // Variable to store the client ID from the SSE connection
    let clientId = null;
    let connectionTimeout = setTimeout(() => {
      log('SSE connection timeout - no connection event received');
      eventSource.close();
      process.exit(1);
    }, 10000);

    // Handle SSE connection events
    eventSource.onopen = () => {
      log('SSE connection opened');
    };

    eventSource.onerror = (error) => {
      log(`SSE connection error: ${JSON.stringify(error)}`);
      clearTimeout(connectionTimeout);
      eventSource.close();
    };

    // Listen for any events to extract the client ID
    eventSource.onmessage = (event) => {
      try {
        log(`Received generic event: ${event.data}`);
        const data = JSON.parse(event.data);
        
        // Check if this is the connected event with a clientId
        if (data.type === 'connected' && data.clientId) {
          clientId = data.clientId;
          log(`Received client ID from connected event: ${clientId}`);
          clearTimeout(connectionTimeout);
          
          // Now that we have a clientId, we can execute a tool
          executeTools(clientId);
        }
      } catch (error) {
        log(`Error parsing event data: ${error.message}`);
      }
    };

    // Listen for tool_result events
    eventSource.addEventListener('tool_result', (event) => {
      try {
        log(`Received tool_result event: ${event.data}`);
        const result = JSON.parse(event.data);
        log(`Tool execution result: ${JSON.stringify(result, null, 2)}`);
      } catch (error) {
        log(`Error parsing tool_result event: ${error.message}`);
      }
    });

    // Listen for error events from MCP
    eventSource.addEventListener('error_event', (event) => {
      try {
        log(`Received error_event: ${event.data}`);
        const error = JSON.parse(event.data);
        log(`MCP error: ${JSON.stringify(error, null, 2)}`);
      } catch (err) {
        log(`Error parsing error_event: ${err.message}`);
      }
    });

    // Function to execute tools using the client ID
    async function executeTools(clientId) {
      try {
        // Example 1: List directory contents
        log('\nExecuting readDirectory tool...');
        const readDirResult = await makeRequest(
          `${MCP_BASE_URL}/messages`,
          'POST',
          {
            clientId: clientId,
            tool: 'readDirectory',
            parameters: {}
          }
        );
        log(`Directory listing response: ${JSON.stringify(readDirResult, null, 2)}`);

        // Wait for a moment to let the SSE events come in
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Example 2: Create a new file
        log('\nExecuting createFile tool...');
        const createFileResult = await makeRequest(
          `${MCP_BASE_URL}/messages`,
          'POST',
          {
            clientId: clientId,
            tool: 'createFile',
            parameters: {
              filePath: 'hello-mcp.txt',
              content: 'Hello from the MCP AI Agent test!'
            }
          }
        );
        log(`File creation response: ${JSON.stringify(createFileResult, null, 2)}`);

        // Wait for a moment to let the SSE events come in
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Example 3: Read the file we just created
        log('\nExecuting readFile tool...');
        const readFileResult = await makeRequest(
          `${MCP_BASE_URL}/messages`,
          'POST',
          {
            clientId: clientId,
            tool: 'readFile',
            parameters: {
              filePath: 'hello-mcp.txt'
            }
          }
        );
        log(`File read response: ${JSON.stringify(readFileResult, null, 2)}`);

        // Wait for a moment to let the SSE events come in
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Example 4: Run a shell command
        log('\nExecuting runShellCommand tool...');
        const shellResult = await makeRequest(
          `${MCP_BASE_URL}/messages`,
          'POST',
          {
            clientId: clientId,
            tool: 'runShellCommand',
            parameters: {
              command: 'ls -la'
            }
          }
        );
        log(`Shell command response: ${JSON.stringify(shellResult, null, 2)}`);

        // Wait for a moment to let the SSE events come in
        await new Promise(resolve => setTimeout(resolve, 3000));

        // Close the SSE connection after we're done
        log('\nClosing SSE connection...');
        eventSource.close();
        log('Test completed.');
        logStream.end();
        process.exit(0);
      } catch (error) {
        log(`Error executing tools: ${error.message}`);
        eventSource.close();
        logStream.end();
        process.exit(1);
      }
    }
  } catch (error) {
    log(`Error testing MCP server: ${error.message}`);
    logStream.end();
    process.exit(1);
  }
}

// Run the test
testMCPServer(); 
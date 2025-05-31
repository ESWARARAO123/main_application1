import axios from 'axios';

// Define types for the service
interface GenerateCommandRequest {
  request: string;
  tools: Array<{
    name: string;
    description: string;
    parameters: any;
  }>;
  mcpServer: {
    host: string;
    port: number;
  };
}

interface GenerateCommandResponse {
  commands: Array<{
    command: string;
    toolName: string;
    parameters: any;
    description: string;
  }>;
}

interface AnalyzeResultRequest {
  result: any;
  originalRequest: string;
  toolName: string;
  parameters: any;
}

interface AnalyzeResultResponse {
  analysis: string;
  followUpCommands?: Array<{
    command: string;
    toolName: string;
    parameters: any;
    description: string;
  }>;
}

// Create the MCP Agent Service
export const mcpAgentService = {
  /**
   * Generate commands based on a user request
   * @param request The user's request
   * @param tools Available MCP tools
   * @param mcpServer MCP server information
   * @returns Generated commands
   */
  generateCommands: async (
    request: string,
    tools: Array<{
      name: string;
      description: string;
      parameters: any;
    }>,
    mcpServer: {
      host: string;
      port: number;
    }
  ): Promise<GenerateCommandResponse> => {
    try {
      // In a real implementation, this would call your backend AI service
      // For now, we'll simulate it with a direct API call

      const payload: GenerateCommandRequest = {
        request,
        tools,
        mcpServer
      };

      const response = await axios.post('/api/ai/generate-command', payload);
      return response.data;
    } catch (error) {
      console.error('Error generating commands:', error);

      // For testing purposes, return simulated commands based on the request
      // In a real implementation, this would be removed

      // Simulate different commands based on the request content
      if (request.toLowerCase().includes('list')) {
        return {
          commands: [
            {
              command: `List files in current directory`,
              toolName: 'runShellCommand',
              parameters: { command: 'ls -la' },
              description: `This command will list all files in the current directory with detailed information.`
            }
          ]
        };
      } else if (request.toLowerCase().includes('status')) {
        return {
          commands: [
            {
              command: `Check system status`,
              toolName: 'runShellCommand',
              parameters: { command: 'systeminfo' },
              description: `This command will display detailed information about the system configuration.`
            }
          ]
        };
      } else if (request.toLowerCase().includes('network')) {
        return {
          commands: [
            {
              command: `Check network configuration`,
              toolName: 'runShellCommand',
              parameters: { command: 'ipconfig /all' },
              description: `This command will display detailed information about the network configuration.`
            }
          ]
        };
      } else {
        return {
          commands: [
            {
              command: `Execute request: "${request}"`,
              toolName: 'runShellCommand',
              parameters: { command: `echo "Processing: ${request}"` },
              description: `This is a simulated command for the request: "${request}"`
            }
          ]
        };
      }
    }
  },

  /**
   * Analyze the result of a command execution
   * @param result The command execution result
   * @param originalRequest The original user request
   * @param toolName The name of the tool that was executed
   * @param parameters The parameters that were used
   * @returns Analysis of the result and potential follow-up commands
   */
  analyzeResult: async (
    result: any,
    originalRequest: string,
    toolName: string,
    parameters: any
  ): Promise<AnalyzeResultResponse> => {
    try {
      // In a real implementation, this would call your backend AI service
      // For now, we'll simulate it with a direct API call

      const payload: AnalyzeResultRequest = {
        result,
        originalRequest,
        toolName,
        parameters
      };

      const response = await axios.post('/api/ai/analyze-result', payload);
      return response.data;
    } catch (error) {
      console.error('Error analyzing result:', error);

      // For testing purposes, return a simulated analysis
      // In a real implementation, this would be removed
      return {
        analysis: `Result from executing ${toolName} with parameters ${JSON.stringify(parameters)}`
      };
    }
  }
};

export default mcpAgentService;

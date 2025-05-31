import { api } from './api';

/**
 * Interface for shell command execution result
 */
export interface ShellCommandResult {
  success: boolean;
  command: string;
  output?: string;
  error?: string;
  stderr?: string;
  timestamp: string;
  serverConfig?: {
    name: string;
    host: string;
    port: number;
  };
  executionLogs?: string[];
  debugInfo?: {
    status?: number;
    statusText?: string;
    url?: string;
    method?: string;
  };
}

/**
 * Interface for MCP server configuration
 */
export interface MCPServerConfig {
  id: string;
  user_id: string;
  server_name: string;
  mcp_host: string;
  mcp_port: string;
  is_default: boolean;
  last_connection_status?: string;
  last_error_message?: string;
  created_at: string;
  updated_at: string;
}

/**
 * Interface for MCP tools response
 */
export interface MCPToolsResult {
  success: boolean;
  tools?: string;
  error?: string;
  serverConfig?: {
    id: string;
    name: string;
    host: string;
    port: string;
  };
  timestamp: string;
}

/**
 * Shell Command Service
 * Provides methods for executing shell commands and managing MCP servers
 */
export const shellCommandService = {
  /**
   * Execute a shell command on the configured MCP server
   * @param command The shell command to execute
   * @returns Promise resolving to command execution result
   */
  executeCommand: async (command: string): Promise<ShellCommandResult> => {
    const executionLogs: string[] = [];
    
    try {
      executionLogs.push(`Starting command execution: ${command}`);
      executionLogs.push(`API endpoint: /ai/tools/runshellcommand`);
      
      const response = await api.post('/ai/tools/runshellcommand', {
        command
      });
      
      executionLogs.push(`Received response from server`);
      
      const result = response.data;
      
      // Parse execution logs from the output if available
      if (result.output && typeof result.output === 'string') {
        try {
          const parsed = JSON.parse(result.output);
          if (parsed.logs && Array.isArray(parsed.logs)) {
            executionLogs.push(...parsed.logs);
          }
        } catch {
          // Not JSON or no logs, continue
        }
      }
      
      executionLogs.push(`Command completed with status: ${result.success ? 'SUCCESS' : 'FAILED'}`);
      
      return {
        ...result,
        executionLogs,
        timestamp: result.timestamp || new Date().toISOString()
      };
    } catch (error: any) {
      executionLogs.push(`Error during command execution: ${error.message}`);
      
      // Add more detailed error information for debugging
      if (error.response) {
        executionLogs.push(`HTTP Status: ${error.response.status} ${error.response.statusText}`);
        executionLogs.push(`Response URL: ${error.response.config?.url}`);
        executionLogs.push(`Response Headers: ${JSON.stringify(error.response.headers)}`);
        if (error.response.data) {
          executionLogs.push(`Response Data: ${JSON.stringify(error.response.data)}`);
        }
      } else if (error.request) {
        executionLogs.push(`Request failed: ${error.request}`);
      } else {
        executionLogs.push(`Error setup: ${error.message}`);
      }
      
      console.error('Shell command execution failed:', error);
      
      // Provide more detailed error message
      let errorMessage = 'Unknown error occurred';
      if (error.response?.status === 404) {
        errorMessage = 'Shell command API endpoint not found (404). Please check if the server is properly configured.';
      } else if (error.response?.status === 503) {
        errorMessage = error.response.data?.error || 'Shell command service is unavailable';
      } else if (error.response?.data?.error) {
        errorMessage = error.response.data.error;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      return {
        success: false,
        command,
        error: errorMessage,
        timestamp: new Date().toISOString(),
        executionLogs,
        // Add debug information
        debugInfo: {
          status: error.response?.status,
          statusText: error.response?.statusText,
          url: error.response?.config?.url,
          method: error.response?.config?.method
        }
      };
    }
  },

  /**
   * Test MCP connection
   * @returns Promise resolving to connection test result
   */
  testConnection: async (): Promise<{ success: boolean; message: string }> => {
    try {
      const response = await api.get('/ai/tools/runshellcommand/test');
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        message: error.response?.data?.error || error.message || 'Connection test failed'
      };
    }
  },

  /**
   * Get available MCP tools
   * @returns Promise resolving to available tools list
   */
  getAvailableTools: async (): Promise<{ success: boolean; tools?: any[]; error?: string }> => {
    try {
      const response = await api.get('/ai/tools/runshellcommand/tools');
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.error || error.message || 'Failed to get tools'
      };
    }
  },

  /**
   * Get user's MCP servers
   * @returns Promise resolving to user's MCP servers
   */
  getMCPServers: async (): Promise<{ success: boolean; servers?: any[]; error?: string }> => {
    try {
      const response = await api.get('/ai/tools/runshellcommand/servers');
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.error || error.message || 'Failed to get servers'
      };
    }
  }
};

/**
 * Common shell command suggestions for AI usage
 */
export const shellCommandSuggestions = {
  fileOperations: [
    'ls -la',
    'pwd',
    'find . -name "*.txt"',
    'cat filename.txt',
    'head -10 filename.txt',
    'tail -10 filename.txt',
    'mkdir directory_name',
    'cp source destination',
    'mv source destination',
    'rm filename.txt'
  ],
  
  systemInfo: [
    'uname -a',
    'whoami',
    'date',
    'uptime',
    'df -h',
    'free -h',
    'top -n 1',
    'ps aux',
    'netstat -tuln',
    'ss -tuln'
  ],
  
  textProcessing: [
    'grep "pattern" filename.txt',
    'wc -l filename.txt',
    'sort filename.txt',
    'uniq filename.txt',
    'cut -d"," -f1 filename.csv',
    'awk "{print $1}" filename.txt',
    'sed "s/old/new/g" filename.txt'
  ],
  
  network: [
    'ping -c 4 google.com',
    'curl -I https://example.com',
    'wget https://example.com/file.txt',
    'nslookup google.com',
    'netstat -rn'
  ]
};

export default shellCommandService; 
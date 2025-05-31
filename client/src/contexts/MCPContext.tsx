import React, { createContext, useContext, useState, useEffect, ReactNode, useRef } from 'react';
import axios from 'axios';
import { useWebSocket } from './WebSocketContext';

interface MCPServer {
  id: string;
  mcp_nickname: string;
  mcp_host: string;
  mcp_port: number;
  mcp_connection_status: string;
  is_default: boolean;
  tools_count?: number;
}

interface MCPTool {
  name: string;
  description: string;
  parameters: Record<string, any>;
}

interface MCPConnection {
  connectionId: string | null;
  clientId: string | null;
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
  error: string | null;
}

interface StoredConnectionInfo {
  connectionId: string;
  clientId: string;
  timestamp: number;
  server: string;
}

interface MCPContextType {
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  defaultServer: MCPServer | null;
  availableServers: MCPServer[];
  availableTools: MCPTool[];
  toggleMCPEnabled: () => void;
  isMCPEnabled: boolean;
  refreshConnection: () => Promise<void>;
  executeCommand: (toolName: string, parameters?: any) => Promise<any>;
  selectServer: (serverId: string) => Promise<void>;
  showServerSelector: boolean;
  setShowServerSelector: (show: boolean) => void;
  connectToServer: (server: MCPServer) => Promise<void>;
  testServerConnection: (server: MCPServer) => Promise<any>;
  mcpConnection: MCPConnection;
  reconnect: () => void;
  getClientId: () => string | null;
  dispatchToolResult: (data: any) => void;
  registerToolResultHandler: (handler: (data: any) => void) => void;
}

const MCPContext = createContext<MCPContextType | undefined>(undefined);

export const useMCP = () => {
  const context = useContext(MCPContext);
  if (context === undefined) {
    throw new Error('useMCP must be used within a MCPProvider');
  }
  return context;
};

interface MCPProviderProps {
  children: ReactNode;
}

export const MCPProvider: React.FC<MCPProviderProps> = ({ children }) => {
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [defaultServer, setDefaultServer] = useState<MCPServer | null>(null);
  const [availableServers, setAvailableServers] = useState<MCPServer[]>([]);
  const [availableTools, setAvailableTools] = useState<MCPTool[]>([]);
  const [showServerSelector, setShowServerSelector] = useState<boolean>(false);
  const [isMCPEnabled, setIsMCPEnabled] = useState<boolean>(() => {
    // Try to get the MCP enabled state from localStorage
    const savedState = localStorage.getItem('mcp_enabled');
    return savedState ? JSON.parse(savedState) : false;
  });

  // WebSocket connection state
  const [mcpConnection, setMCPConnection] = useState<MCPConnection>({
    connectionId: null,
    clientId: null,
    status: 'disconnected',
    error: null
  });

  // Refs for managing reconnection and tool results
  const reconnectTimeoutRef = useRef<number | null>(null);
  const mcpAgentDispatcher = useRef<((data: any) => void) | null>(null);
  const isConnectingRef = useRef(false); // Ref to track connection attempts
  const reconnectAttemptsRef = useRef<number>(0);
  
  // Get WebSocket connection
  const { connected: wsConnected, send, addMessageListener } = useWebSocket();

  // Add a flag to track if we're currently connecting
  const [isConnectingToServer, setIsConnectingToServer] = useState(false);
  
  // Fetch MCP status on mount
  useEffect(() => {
    // Initial load of available servers
    const fetchServers = async () => {
      try {
        setIsLoading(true);
        const response = await axios.get('/api/mcp/server/config');
        if (response.data && response.data.configurations && response.data.configurations.length > 0) {
          setAvailableServers(response.data.configurations);

          // Find default server if any
          const defaultServer = response.data.configurations.find((server: MCPServer) => server.is_default);
          if (defaultServer) {
            setDefaultServer(defaultServer);

            // If MCP is enabled, connect to the default server
            if (isMCPEnabled) {
              connectToServer(defaultServer);
            }
          } else {
            // If no default server but we have servers, use the first one
            setDefaultServer(response.data.configurations[0]);

            // If MCP is enabled, connect to this server
            if (isMCPEnabled) {
              connectToServer(response.data.configurations[0]);
            }
          }
        } else {
          // No servers found
          setAvailableServers([]);
          setDefaultServer(null);
          setError('No MCP servers configured. Please configure a server in the settings page.');

          // If MCP is enabled but no servers, show selector to guide user
          if (isMCPEnabled) {
            setShowServerSelector(true);
          }
        }
      } catch (error) {
        console.error('Failed to fetch MCP servers:', error);
        setAvailableServers([]);
        setDefaultServer(null);
        setError('Failed to retrieve MCP server configurations. Please try again later.');

        // If MCP is enabled but failed to fetch servers, show selector
        if (isMCPEnabled) {
          setShowServerSelector(true);
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchServers();
  }, []);

  // Set up WebSocket message listeners for MCP communication
  useEffect(() => {
    if (!wsConnected) {
      // If WebSocket is not connected, we can't use it for MCP
      return;
    }

    console.log('Setting up MCP WebSocket message listeners');

    // Listen for MCP connections
    const connectedUnsubscribe = addMessageListener('mcp_connected', (message) => {
      console.log('Received MCP connected message:', message);
      
      if (message.connectionId && message.clientId) {
        // Store connection info
        const connectionInfo: StoredConnectionInfo = {
          connectionId: message.connectionId,
          clientId: message.clientId,
          timestamp: Date.now(),
          server: `${message.host}:${message.port}`
        };
        
        localStorage.setItem('mcp_connection_info', JSON.stringify(connectionInfo));
        
        // Update connection state
        setMCPConnection({
          connectionId: message.connectionId,
          clientId: message.clientId,
          status: 'connected',
          error: null
        });
        
        // Clear any reconnect attempts
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
        
        setError(null);
        isConnectingRef.current = false;
      }
    });
    
    // Listen for MCP connecting state
    const connectingUnsubscribe = addMessageListener('mcp_connecting', (message) => {
      console.log('Received MCP connecting message:', message);
      
      setMCPConnection(prev => ({
        ...prev,
        connectionId: message.connectionId || null,
        status: 'connecting',
        error: null
      }));
    });
    
    // Listen for MCP connection errors
    const connectionErrorUnsubscribe = addMessageListener('mcp_connect_error', (message) => {
      console.error('Received MCP connection error:', message);
      
      setMCPConnection(prev => ({
        ...prev,
        status: 'error',
        error: message.error || 'Connection failed'
      }));
      
      setError(message.error || 'Failed to connect to MCP server');
      isConnectingRef.current = false;
    });
    
    // Listen for MCP tool results
    const toolResultUnsubscribe = addMessageListener('mcp_tool_result', (message) => {
      console.log('Received MCP tool result:', message);
      dispatchToolResult(message.result);
    });
    
    // Listen for MCP disconnection
    const disconnectedUnsubscribe = addMessageListener('mcp_disconnected', (message) => {
      console.log('Received MCP disconnected message:', message);
      
      setMCPConnection({
        connectionId: null,
        clientId: null,
        status: 'disconnected',
        error: null
      });
      
      // Remove stored connection info
      localStorage.removeItem('mcp_connection_info');
    });
    
    // Listen for MCP reconnection attempts
    const reconnectingUnsubscribe = addMessageListener('mcp_reconnecting', (message) => {
      console.log('Received MCP reconnecting message:', message);
      
      setMCPConnection(prev => ({
        ...prev,
        status: 'connecting',
        error: null
      }));
    });
    
    // Listen for MCP SSE events
    const sseEventUnsubscribe = addMessageListener('mcp_sse_event', (message) => {
      console.log('Received MCP SSE event:', message);
    });
    
    // Clean up listeners on unmount
    return () => {
      connectedUnsubscribe();
      connectingUnsubscribe();
      connectionErrorUnsubscribe();
      toolResultUnsubscribe();
      disconnectedUnsubscribe();
      reconnectingUnsubscribe();
      sseEventUnsubscribe();
    };
  }, [wsConnected, addMessageListener]);

  // Connect when MCP is enabled
  useEffect(() => {
    if (isMCPEnabled && defaultServer) {
      connectToServer(defaultServer);
    } else if (isMCPEnabled && !defaultServer) {
      // MCP is enabled, but no default server is set yet.
      // Show server selector if servers are available, otherwise show error.
      if (availableServers.length > 0) {
        setShowServerSelector(true);
      } else {
        // fetchServers in the initial useEffect should have already set an error
        // or will set one. If not, ensure an error and selector are shown.
        if (!error) {
          setError('No MCP servers configured. Please configure a server in the settings page.');
        }
        setShowServerSelector(true);
      }
    } else if (!isMCPEnabled) {
      // MCP is disabled, ensure everything is disconnected.
      disconnectFromMCP();
    }
  }, [isMCPEnabled, defaultServer, availableServers, error]);

  // Restore connection from localStorage if possible
  useEffect(() => {
    if (wsConnected && defaultServer && isMCPEnabled) {
      try {
        const savedConnectionStr = localStorage.getItem('mcp_connection_info');
        if (savedConnectionStr) {
          const savedConnection: StoredConnectionInfo = JSON.parse(savedConnectionStr);
          
          // Check if connection is not too old (less than 15 minutes)
          const age = Date.now() - savedConnection.timestamp;
          if (age < 15 * 60 * 1000) {
            // Check if it's for the current server
            const currentServer = `${defaultServer.mcp_host}:${defaultServer.mcp_port}`;
            if (savedConnection.server === currentServer) {
              console.log('Restoring MCP connection from localStorage');
              
              // Set the connection state
              setMCPConnection({
                connectionId: savedConnection.connectionId,
                clientId: savedConnection.clientId,
                status: 'connected',
                error: null
              });
              
              // Verify the connection is still valid
              verifyConnection(savedConnection.connectionId, savedConnection.clientId);
              
              return;
            }
          }
        }
      } catch (e) {
        console.error('Error restoring MCP connection from localStorage:', e);
      }
    }
  }, [wsConnected, defaultServer, isMCPEnabled]);

  // Save MCP enabled state to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('mcp_enabled', JSON.stringify(isMCPEnabled));
  }, [isMCPEnabled]);

  // When welcome message is sent, store a timestamp to avoid duplicates
  useEffect(() => {
    if (mcpConnection.clientId && mcpConnection.status === 'connected') {
      // Store the successful connection time
      localStorage.setItem('mcp_connection_time', Date.now().toString());
    }
  }, [mcpConnection.clientId, mcpConnection.status]);

  // Dispatch tool result to registered handler
  const dispatchToolResult = (data: any) => {
    console.log('Dispatching tool result:', data);
    if (mcpAgentDispatcher.current) {
      mcpAgentDispatcher.current(data);
    }
  };

  // Register a handler for tool results
  const registerToolResultHandler = (handler: (data: any) => void) => {
    mcpAgentDispatcher.current = handler;
  };

  // Verify a connection is still valid
  const verifyConnection = (connectionId: string, clientId: string) => {
    if (!wsConnected) {
      console.log('Cannot verify connection: WebSocket not connected');
      return false;
    }
    
    console.log(`Verifying MCP connection: ${connectionId} with clientId: ${clientId}`);
    
    // Execute a simple test command to verify the connection
    executeCommand('echo', { text: 'connection_test' })
      .then(result => {
        console.log('Connection verification successful:', result);
      })
      .catch(err => {
        console.error('Connection verification failed:', err);
        
        // Set connection to error state
        setMCPConnection({
          connectionId: null,
          clientId: null,
          status: 'error',
          error: 'Connection verification failed'
        });
        
        // Remove stored connection info
        localStorage.removeItem('mcp_connection_info');
        
        // If MCP is still enabled, try to reconnect
        if (isMCPEnabled && defaultServer) {
          reconnect();
        }
      });
  };

  // Connect to server using WebSocket
  const connectToServer = async (server: MCPServer) => {
    setIsLoading(true);
    setError(null);

    try {
      console.log(`Starting connection to MCP server: ${server.mcp_host}:${server.mcp_port}`);
      
      // Basic check for server availability
      try {
        const infoResponse = await axios.get(`http://${server.mcp_host}:${server.mcp_port}/info`, {
          timeout: 3000
        });
        console.log('MCP server info check successful:', infoResponse.data);
      } catch (infoError) {
        console.error('Error connecting to MCP server info endpoint:', infoError);
        throw new Error(`Cannot connect to MCP server: ${infoError.message}`);
      }
      
      // Set the default server
      setDefaultServer(server);
      
      // Connect via WebSocket for client ID
      console.log(`Establishing WebSocket connection to MCP server: ${server.mcp_host}:${server.mcp_port}`);
      
      // Update UI state
      setMCPConnection({
        connectionId: null,
        clientId: null,
        status: 'connecting',
        error: null
      });
      
      // Connect via WebSocket
      if (wsConnected) {
        send({
          type: 'mcp_connect',
          host: server.mcp_host,
          port: server.mcp_port
        });
      } else {
        throw new Error('WebSocket connection to backend is not established. Please refresh the page.');
      }
      
      // Fetch tools in background
      fetchServerTools(server).catch(toolsError => {
        console.warn('Non-critical: Failed to fetch MCP tools:', toolsError);
        setAvailableTools([]);
      });
      
    } catch (err) {
      console.error('Error connecting to MCP server:', err);
      setError(err.message || 'Failed to connect to MCP server');
      setMCPConnection({
        connectionId: null,
        clientId: null,
        status: 'error',
        error: err.message || 'Failed to connect to MCP server'
      });
      
      // Try reconnecting after a delay
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      reconnectTimeoutRef.current = window.setTimeout(() => {
        reconnect();
        reconnectTimeoutRef.current = null;
      }, 5000);
    } finally {
      setIsLoading(false);
    }
  };

  // Disconnect from MCP server
  const disconnectFromMCP = () => {
    if (!wsConnected) {
      // If WebSocket is not connected, just update our local state
      setMCPConnection({
        connectionId: null,
        clientId: null,
        status: 'disconnected',
        error: null
      });
      
      // Clear stored connection info
      localStorage.removeItem('mcp_connection_info');
      
      return;
    }
    
    if (mcpConnection.connectionId) {
      console.log(`Disconnecting from MCP server: ${mcpConnection.connectionId}`);
      
      // Send disconnect request
      send({
        type: 'mcp_disconnect',
        connectionId: mcpConnection.connectionId
      });
    }
    
    // Update state immediately for better UX
    setMCPConnection({
      connectionId: null,
      clientId: null,
      status: 'disconnected',
      error: null
    });
    
    // Clear stored connection info
    localStorage.removeItem('mcp_connection_info');
  };

  // Reconnect to MCP server
  const reconnect = () => {
    // If we already have a valid connection, don't reconnect
    if (isConnectingToServer || (mcpConnection.clientId && mcpConnection.status === 'connected')) {
      console.log('Already connected or connecting, skipping reconnection attempt');
      return;
    }

    // Clear existing connection info
    setMCPConnection({
      connectionId: null,
      clientId: null,
      status: 'connecting',
      error: null
    });

    // Check if we have a stored connection
    const storedConnInfo = localStorage.getItem('mcp_connection_info');
    if (storedConnInfo) {
      try {
        const connInfo = JSON.parse(storedConnInfo);
        
        // If connection info is older than 1 hour, remove it
        const oneHour = 60 * 60 * 1000; // 1 hour in milliseconds
        if (Date.now() - connInfo.timestamp > oneHour) {
          localStorage.removeItem('mcp_connection_info');
        } else {
          console.log('Using stored connection info:', connInfo);
          
          // Try to connect using stored info
          if (connInfo.server && connInfo.clientId) {
            // Find server in available servers
            const server = availableServers.find(s => 
              `${s.mcp_host}:${s.mcp_port}` === connInfo.server);
            
            if (server) {
              console.log('Found matching server, connecting to:', server);
              
              // Set the default server
              setDefaultServer(server);
              
              // Update connection info
              setMCPConnection({
                connectionId: connInfo.connectionId,
                clientId: connInfo.clientId,
                status: 'connected',
                error: null
              });
              
              // Update connection state
              setIsConnectingToServer(false);
              
              // Return early as we've reconnected
              return;
            }
          }
        }
      } catch (err) {
        console.error('Error parsing stored connection info:', err);
        localStorage.removeItem('mcp_connection_info');
      }
    }

    // No valid stored connection, so connect via WebSocket
    setIsConnectingToServer(true);
    
    if (defaultServer) {
      console.log('Connecting to MCP server via WebSocket:', defaultServer);
      
      if (wsConnected) {
        // Use WebSocket to connect
        send({
          type: 'mcp_connect',
          host: defaultServer.mcp_host,
          port: defaultServer.mcp_port,
          immediate: true
        });
      } else {
        console.error('WebSocket not connected, cannot connect to MCP server');
        setMCPConnection({
          ...mcpConnection,
          status: 'error',
          error: 'WebSocket not connected. Please refresh the page and try again.'
        });
        setIsConnectingToServer(false);
      }
    } else {
      console.log('No default server available, showing server selector');
      setShowServerSelector(true);
      setIsConnectingToServer(false);
    }
  };

  // Get current clientId
  const getClientId = () => {
    return mcpConnection.clientId;
  };

  // Fetch tools from a server
  const fetchServerTools = async (server: MCPServer) => {
    try {
      console.log(`Fetching tools from MCP server: ${server.mcp_host}:${server.mcp_port}`);

      // Use the tools endpoint from config.ini
      const toolsEndpoint = '/tools'; // From config.ini: mcp_terminal_name_1_fetch_tool_endpoint

      const response = await axios.get(`http://${server.mcp_host}:${server.mcp_port}${toolsEndpoint}`, {
        timeout: 5000
      });

      // Parse the response based on structure
      let tools: MCPTool[] = [];
      if (response.data) {
        if (response.data.tools && Array.isArray(response.data.tools)) {
          // Format: { tools: [...] }
          tools = response.data.tools;
        } else if (Array.isArray(response.data)) {
          // Format: [...]
          tools = response.data;
        }
      }

      console.log(`Found ${tools.length} MCP tools`);
      setAvailableTools(tools);

      return tools;
    } catch (err: any) {
      console.error('Error fetching MCP tools:', err);
      setAvailableTools([]);
      throw err;
    }
  };

  // Refresh the current connection
  const refreshConnection = async () => {
    // If we're already connecting or already have a client ID, don't try to connect again
    if (isConnectingToServer || mcpConnection.clientId) {
      console.log('Already connecting or have client ID, skipping connection attempt');
      return;
    }
    
    setIsConnectingToServer(true);
    setIsLoading(true);
    
    try {
      if (!defaultServer) {
        setError('No MCP server selected. Please select a server first.');
        setIsLoading(false);
        return;
      }

      // Test the connection
      const testResult = await testServerConnection(defaultServer);

      if (testResult.success) {
        // Reconnect
        reconnect();

        // Refresh tools if available
        if (testResult.toolCount) {
          try {
            await fetchServerTools(defaultServer);
          } catch (toolsError) {
            console.error('Failed to fetch MCP tools:', toolsError);
            setAvailableTools([]);
          }
        }
      } else {
        setError(testResult.error || 'Failed to connect to MCP server');
      }
    } catch (err: any) {
      console.error('Error refreshing MCP connection:', err);
      setError(err.message || 'Failed to refresh MCP connection');
    } finally {
      setIsConnectingToServer(false);
      setIsLoading(false);
    }
  };

  // Toggle MCP enabled state
  const toggleMCPEnabled = () => {
    const newState = !isMCPEnabled;
    setIsMCPEnabled(newState);

    // If turning off, disconnect
    if (!newState) {
      disconnectFromMCP();
    }
  };

  // Select a server by ID
  const selectServer = async (serverId: string) => {
    const selectedServer = availableServers.find(s => s.id === serverId);
    if (selectedServer) {
      await connectToServer(selectedServer);
      setShowServerSelector(false);
    } else {
      setError('Invalid server selected');
    }
  };

  // Add a direct client ID retrieval method for fallback scenarios
  const retrieveClientIdDirectly = async (server: MCPServer): Promise<string | null> => {
    try {
      console.log(`Attempting direct client ID retrieval from server: ${server.mcp_host}:${server.mcp_port}`);
      
      // Call the server-side API endpoint that can retrieve a client ID directly
      const response = await axios.get('/api/mcp/get-client-id', {
        params: {
          host: server.mcp_host,
          port: server.mcp_port
        },
        timeout: 10000 // 10 second timeout
      });
      
      if (response.data && response.data.clientId) {
        const clientId = response.data.clientId;
        console.log(`Successfully retrieved client ID directly: ${clientId}`);
        
        // Store it in the connection info
        setMCPConnection(prev => ({
          ...prev,
          clientId,
          status: 'connected',
          error: null
        }));
        
        // Also save to localStorage for persistence
        const connectionInfo: StoredConnectionInfo = {
          connectionId: mcpConnection.connectionId || `direct-${Date.now()}`,
          clientId,
          timestamp: Date.now(),
          server: `${server.mcp_host}:${server.mcp_port}`
        };
        
        localStorage.setItem('mcp_connection_info', JSON.stringify(connectionInfo));
        
        return clientId;
      }
      
      return null;
    } catch (err) {
      console.error('Failed to retrieve client ID directly:', err);
      return null;
    }
  };

  // Enhance executeCommand with fallback for clientId acquisition
  const executeCommand = async (toolName: string, parameters: any = {}) => {
    if (!mcpConnection.clientId || mcpConnection.status !== 'connected' || !mcpConnection.connectionId) {
      // If we don't have a client ID but server connection is active, try to retrieve it directly
      if (defaultServer && mcpConnection.status === 'connected' && !mcpConnection.clientId) {
        console.log('No client ID available. Attempting direct retrieval before command execution...');
        const clientId = await retrieveClientIdDirectly(defaultServer);
        if (!clientId) {
          throw new Error('Failed to acquire client ID. Please reconnect to the MCP server.');
        }
        // Now we have a clientId, proceed with execution
      } else {
        throw new Error('MCP is not connected');
      }
    }

    try {
      console.log(`Executing MCP command: ${toolName}`, parameters);

      // Create a unique execution ID
      const executeId = `exec-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

      // Send the command execution request via WebSocket
      send({
        type: 'mcp_execute_tool',
        connectionId: mcpConnection.connectionId,
        executeId,
        tool: toolName,
        parameters
      });

      // Wait for the result using a Promise
      return new Promise((resolve, reject) => {
        // Use a timeout to prevent hanging forever
        const timeoutId = setTimeout(() => {
          // Remove the listener
          resultListener();
          errorListener();
          reject(new Error('Command execution timed out'));
        }, 30000); // 30 second timeout

        // Listen for the execution result
        const resultListener = addMessageListener('mcp_execute_result', (message) => {
          if (message.executeId === executeId) {
            clearTimeout(timeoutId);
            // Remove the listeners
            resultListener();
            errorListener();
            
            // Resolve with the result
            resolve(message.result);
          }
        });

        // Listen for execution errors
        const errorListener = addMessageListener('mcp_execute_error', (message) => {
          if (message.executeId === executeId) {
            clearTimeout(timeoutId);
            // Remove the listeners
            resultListener();
            errorListener();
            
            // Handle specific error case for clientId issues
            if (message.error && 
                (message.error.includes('clientId') || 
                 message.error.includes('Client ID') ||
                 message.details?.error === 'Missing clientId')) {
              
              console.log('ClientId issue detected in command execution. Attempting recovery...');
              
              // Try to recover the clientId and retry
              retrieveClientIdDirectly(defaultServer!)
                .then(clientId => {
                  if (clientId) {
                    console.log('Successfully recovered clientId. Command will need to be retried.');
                    reject(new Error('Client ID was recovered. Please try the command again.'));
                  } else {
                    reject(new Error('Command failed: Invalid client ID and recovery failed.'));
                  }
                })
                .catch(recoveryError => {
                  reject(new Error(`Command failed and clientId recovery failed: ${recoveryError.message}`));
                });
            } else {
              // Regular command error
              reject(new Error(message.error || 'Command execution failed'));
            }
          }
        });
      });
    } catch (err: any) {
      console.error('Error executing MCP command:', err);

      // Check if it's a connection issue
      if (mcpConnection.status !== 'connected') {
        setError('Lost connection to MCP server. Attempting to reconnect...');
        reconnect();
      }

      throw err;
    }
  };

  const isConnected = mcpConnection.status === 'connected' && !!mcpConnection.clientId;

  // Add the connectToMCPServer function
  const connectToMCPServer = (server: MCPServer) => {
    if (!wsConnected) {
      setError('WebSocket is not connected. Cannot connect to MCP server.');
      return;
    }
    
    if (isConnectingRef.current) {
      console.log('Connection attempt already in progress');
      return;
    }
    
    isConnectingRef.current = true;
    setError(null);
    
    // Update the UI state
    setMCPConnection({
      connectionId: null,
      clientId: null,
      status: 'connecting',
      error: null
    });
    
    console.log(`Connecting to MCP server: ${server.mcp_host}:${server.mcp_port} via WebSocket`);
    
    // Send connection request through WebSocket
    send({
      type: 'mcp_connect',
      host: server.mcp_host,
      port: server.mcp_port
    });
    
    // Set a timeout to fail if we don't get a clientId
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    reconnectTimeoutRef.current = window.setTimeout(() => {
      if (mcpConnection.status !== 'connected') {
        console.error('Timeout waiting for MCP clientId');
        
        // Update connection status to error
        setMCPConnection(prev => ({
          ...prev,
          status: 'error',
          error: 'Timeout waiting for clientId from MCP server'
        }));
        
        // Attempt to reconnect automatically
        reconnect();
      }
      reconnectTimeoutRef.current = null;
    }, 10000); // 10-second timeout
  };

  // Add the testServerConnection function
  const testServerConnection = async (server: MCPServer) => {
    try {
      console.log(`Testing connection to MCP server: ${server.mcp_host}:${server.mcp_port}`);

      // Use the information endpoint from config.ini
      const infoEndpoint = '/info'; // From config.ini: mcp_terminal_name_1_information_endpoint
      const toolsEndpoint = '/tools'; // From config.ini: mcp_terminal_name_1_fetch_tool_endpoint

      // Direct connection to the MCP server info endpoint
      const response = await axios.get(`http://${server.mcp_host}:${server.mcp_port}${infoEndpoint}`, {
        timeout: 5000
      });

      console.log('MCP server info response:', response.data);

      if (response.status === 200 && response.data) {
        try {
          // Also fetch tools to check if they're available
          const toolsResponse = await axios.get(`http://${server.mcp_host}:${server.mcp_port}${toolsEndpoint}`, {
            timeout: 5000
          });

          console.log('MCP server tools response:', toolsResponse.data);

          // Success response with server info and tool count
          return {
            success: true,
            server: response.data,
            toolCount: toolsResponse.data.count || (Array.isArray(toolsResponse.data) ? toolsResponse.data.length : 0)
          };
        } catch (toolsError) {
          console.error('Error fetching MCP tools:', toolsError);

          // Even if tools fetch fails, consider the connection successful
          // as long as the info endpoint worked
          return {
            success: true,
            server: response.data,
            toolCount: 0
          };
        }
      }

      return {
        success: false,
        error: 'Invalid response from MCP server'
      };
    } catch (err: any) {
      console.error('Error testing MCP connection:', err);
      return {
        success: false,
        error: err.message || 'Failed to connect to MCP server'
      };
    }
  };

  return (
    <MCPContext.Provider
      value={{
        isConnected,
        isLoading,
        error,
        defaultServer,
        availableServers,
        availableTools,
        toggleMCPEnabled,
        isMCPEnabled,
        refreshConnection,
        executeCommand,
        selectServer,
        showServerSelector,
        setShowServerSelector,
        connectToServer,
        testServerConnection,
        mcpConnection,
        reconnect,
        getClientId,
        dispatchToolResult,
        registerToolResultHandler
      }}
    >
      {children}
    </MCPContext.Provider>
  );
};

export default MCPContext;
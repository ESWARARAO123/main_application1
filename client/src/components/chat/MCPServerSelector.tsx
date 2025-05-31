import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  XMarkIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  ServerIcon,
  PlusIcon,
  InformationCircleIcon,
  ArrowRightCircleIcon
} from '@heroicons/react/24/outline';
import { useMCP } from '../../contexts/MCPContext';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import { useWebSocket } from '../../contexts/WebSocketContext';

interface MCPServer {
  id: string;
  mcp_nickname: string;
  mcp_host: string;
  mcp_port: number;
  mcp_connection_status: string;
  is_default: boolean;
  tools_count?: number;
  mcp_server_version?: string;
}

interface MCPServerSelectorProps {
  isOpen: boolean;
  onClose: () => void;
  onServerSelect: (serverId: string) => void;
}

const MCPServerSelector: React.FC<MCPServerSelectorProps> = ({
  isOpen,
  onClose,
  onServerSelect
}) => {
  const { testServerConnection, connectToServer, mcpConnection, defaultServer } = useMCP();
  const { isAgentEnabled, toggleAgent } = useMCPAgent();
  const { connected: wsConnected, isFullyReady } = useWebSocket();
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedServerId, setSelectedServerId] = useState<string | null>(null);
  const [testingServer, setTestingServer] = useState<string | null>(null);
  const [defaultMCPPort, setDefaultMCPPort] = useState<string>('');
  const [connecting, setConnecting] = useState(false);
  const [connectionPhase, setConnectionPhase] = useState<string>('');

  // Direct connection states
  const [showDirectConnect, setShowDirectConnect] = useState(false);
  const [directHost, setDirectHost] = useState('');
  const [directPort, setDirectPort] = useState('');
  const [directName, setDirectName] = useState('');
  const [directConnecting, setDirectConnecting] = useState(false);
  const [sseConnectionStatus, setSSEConnectionStatus] = useState<string>('');

  // New state to track direct SSE success
  const [directSseSuccess, setDirectSseSuccess] = useState<string | null>(null);

  // Update the useEffect that monitors connection status to show detailed phases
  useEffect(() => {
    setSSEConnectionStatus(mcpConnection.status);
    
    // Check WebSocket readiness first
    if (connecting && (!wsConnected || !isFullyReady)) {
      setConnectionPhase('websocket');
      return;
    }
    
    // Update connection phase based on status
    if (connecting) {
      if (mcpConnection.status === 'connecting') {
        setConnectionPhase('mcp');
      } else if (mcpConnection.status === 'connected') {
        setConnectionPhase('complete');
      }
    }
    
    // If SSE connection is established, close the dialog
    if (
      mcpConnection.status === 'connected' &&
      mcpConnection.clientId &&
      connecting
    ) {
      setTimeout(() => {
        setConnecting(false);
        setConnectionPhase('');
        onClose();
        // Only toggle the agent if it's not already enabled
        if (!isAgentEnabled) {
          console.log('Enabling MCP Agent after successful connection');
          toggleAgent();
        } else {
          console.log('MCP Agent already enabled, not toggling');
        }
      }, 500); // shorter delay for snappier UX
    }
    
    // If there's an error with the SSE connection, show it
    if (mcpConnection.status === 'error' && connecting) {
      setConnecting(false);
      setConnectionPhase('');
      setError(mcpConnection.error || 'Failed to establish connection to MCP server');
    }
  }, [mcpConnection, connecting, wsConnected, isFullyReady, onClose, isAgentEnabled, toggleAgent]);

  // Fetch default MCP port from configuration when component mounts
  useEffect(() => {
    const fetchMCPConfig = async () => {
      try {
        const response = await axios.get('/api/mcp/config');
        if (response.data && response.data.defaultTool && response.data.defaultTool.defaultPort) {
          setDefaultMCPPort(response.data.defaultTool.defaultPort.toString());
          // Set default port for direct connection if not already set
          if (!directPort) {
            setDirectPort(response.data.defaultTool.defaultPort.toString());
          }
        } else {
          // Fallback to a sensible default
          setDefaultMCPPort('8080');
          if (!directPort) {
            setDirectPort('8080');
          }
        }
      } catch (error) {
        console.error('Error fetching MCP config:', error);
        // Fallback to a sensible default
        setDefaultMCPPort('8080');
        if (!directPort) {
          setDirectPort('8080');
        }
      }
    };

    fetchMCPConfig();
  }, [directPort]);

  // Fetch MCP servers when the component mounts
  useEffect(() => {
    if (isOpen) {
      fetchServers();
    }
  }, [isOpen]);

  // Fetch MCP servers from the API
  const fetchServers = async () => {
    try {
      setLoading(true);
      setError(null);
      setSelectedServerId(null); // Clear previous selection before fetching
      setServers([]); // Clear previous server list

      const response = await axios.get('/api/mcp/server/config');

      if (response.data && (response.data.servers || response.data.configurations)) {
        const serverData = response.data.servers || response.data.configurations || [];
        setServers(serverData);

        const defaultServer = serverData.find((server: MCPServer) => server.is_default);
        if (defaultServer) {
          setSelectedServerId(defaultServer.id);
        } else if (serverData.length > 0) {
          setSelectedServerId(serverData[0].id);
        }
      } else {
        setServers([]); // Ensure servers is empty if response is not as expected
      }
    } catch (err: any) {
      console.error('Error fetching MCP servers:', err);
      if (err.response && err.response.status === 401) {
        setError('Authentication failed. Please log in to see MCP servers.');
      } else {
        setError(err.response?.data?.error || 'Failed to fetch MCP servers. Check console for details.');
      }
      setServers([]); // Ensure servers list is empty on error
      setSelectedServerId(null); // Clear selected server ID on error
    } finally {
      setLoading(false);
    }
  };

  // Test connection to a server
  const testConnection = async (server: MCPServer) => {
    try {
      setTestingServer(server.id);
      setError(null);
      setDirectSseSuccess(null); // Clear any previous direct SSE success message

      // Use the simplified test connection from MCPContext (checks /info and /tools only)
      const testResult = await testServerConnection(server);

      if (testResult.success) {
        // Update server status in the UI - marks as generally available
        setServers(prevServers =>
          prevServers.map(s =>
            s.id === server.id
              ? {
                  ...s,
                  mcp_connection_status: 'connecting', // Mark as connecting since we're proceeding
                  mcp_server_version: testResult.server?.version || 'unknown',
                  tools_count: testResult.toolCount || 0
                }
              : s
          )
        );
        
        // Automatically connect if test is successful
        console.log(`Test successful for ${server.mcp_nickname}. Auto-connecting...`);
        setSelectedServerId(server.id);
        setConnecting(true);
        setSSEConnectionStatus('connecting');
        
        // Call onServerSelect to notify parent component
        onServerSelect(server.id);
        
        // Call connectToServer from MCPContext
        try {
          await connectToServer(server);
          // Success/failure will be handled by useEffect monitoring mcpConnection
        } catch (connectError) {
          console.error('Error connecting to MCP server after successful test:', connectError);
          setError(connectError.message || 'Failed to connect after successful test');
          setConnecting(false);
          setSSEConnectionStatus('error');
          
          // Update server status to error
          setServers(prevServers =>
            prevServers.map(s =>
              s.id === server.id
                ? { ...s, mcp_connection_status: 'error' }
                : s
            )
          );
        }
      } else {
        // Update server status to error
        setServers(prevServers =>
          prevServers.map(s =>
            s.id === server.id
              ? { ...s, mcp_connection_status: 'error' } 
              : s
          )
        );
        setError(testResult.error || 'Test connection failed for /info or /tools.');
      }
    } catch (err: any) {
      console.error('Error testing MCP connection in component:', err);
      setServers(prevServers =>
        prevServers.map(s =>
          s.id === server.id
            ? { ...s, mcp_connection_status: 'error' }
            : s
        )
      );
      setError(err.message || 'Failed to connect to MCP server during test.');
    } finally {
      setTestingServer(null);
    }
  };

  // Handle direct connection: This function is called directly after direct test
  const handleDirectConnect = async () => {
    if (!directHost.trim()) {
      setError('Please enter a valid host');
      return;
    }
    if (!directName.trim()) {
      setError('Please enter a name for the server');
      return;
    }
    if (isNaN(parseInt(directPort))) {
      setError('Please enter a valid port number');
      return;
    }

    setDirectConnecting(true); // Specific to direct connect UI
    setConnecting(true); // General connecting flag
    setSSEConnectionStatus('connecting');
    setError(null);
    setDirectSseSuccess(null);

    // Try to test the connection first
    try {
      // Quick test of /info and /tools
      const tempServer: MCPServer = {
        id: `direct-${Date.now()}`,
        mcp_nickname: directName,
        mcp_host: directHost,
        mcp_port: parseInt(directPort),
        mcp_connection_status: 'connecting',
        is_default: true
      };

      // Test the connection
      const testResult = await testServerConnection(tempServer);
      
      if (!testResult.success) {
        // If test fails, show error
        setError(testResult.error || 'Server test failed');
        setDirectConnecting(false);
        setConnecting(false);
        setSSEConnectionStatus('error');
        return;
      }
      
      // If test passes, connect
      console.log(`Direct connection test successful. Auto-connecting...`);
      
      // Call onServerSelect to notify parent component
      onServerSelect(tempServer.id);
      
      // Call connectToServer from MCPContext
      await connectToServer(tempServer);
      
      // Save the server configuration for future use
      try {
        const response = await axios.post('/api/mcp/server/config', {
          server_name: directName,
          mcp_host: directHost,
          mcp_port: parseInt(directPort),
          is_default: true
        });
        
        if (response.data && response.data.id) {
          console.log('Server configuration saved with ID:', response.data.id);
        }
      } catch (saveError: any) {
        console.warn('Server connection initiated, but configuration could not be saved:', saveError.message);
      }
      
    } catch (err: any) {
      console.error('Error during direct connection test/connect:', err);
      setError(err.message || 'Failed to connect directly to MCP server');
      setDirectConnecting(false);
      setConnecting(false);
      setSSEConnectionStatus('error');
    }
  };

  // Update the renderConnectionStatus function to show detailed phases and direct SSE success
  const renderConnectionStatus = () => {
    // Show direct SSE success if available
    if (directSseSuccess) {
      return (
        <div className="mb-4 p-3 rounded-md bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 flex items-center">
          <CheckCircleIcon className="w-5 h-5 mr-2" />
          <div>
            <p className="font-medium">Direct SSE Connection Successful!</p>
            <p className="text-xs mt-1">Client ID: {directSseSuccess}</p>
          </div>
        </div>
      );
    }
    
    if (!connecting) return null;
    
    if (!wsConnected || !isFullyReady) {
      return (
        <div className="mb-4 p-3 rounded-md bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 flex items-center">
          <ArrowPathIcon className="w-5 h-5 mr-2 animate-spin" />
          <div>
            <p className="font-medium">Initializing WebSocket connection...</p>
            <p className="text-xs mt-1">This step is required before connecting to the MCP server.</p>
          </div>
        </div>
      );
    }
    
    if (connectionPhase === 'mcp') {
      return (
        <div className="mb-4 p-3 rounded-md bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 flex items-center">
          <ArrowPathIcon className="w-5 h-5 mr-2 animate-spin" />
          <div>
            <p className="font-medium">Establishing MCP connection...</p>
            <p className="text-xs mt-1">Connecting to server and acquiring client ID.</p>
          </div>
        </div>
      );
    }
    
    if (connectionPhase === 'complete') {
      return (
        <div className="mb-4 p-3 rounded-md bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 flex items-center">
          <CheckCircleIcon className="w-5 h-5 mr-2" />
          <div>
            <p className="font-medium">Connection successful!</p>
            <p className="text-xs mt-1">Client ID acquired. Redirecting to chat...</p>
          </div>
        </div>
      );
    }
    
    return (
      <div className="mb-4 p-3 rounded-md bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 flex items-center">
        <ArrowPathIcon className="w-5 h-5 mr-2 animate-spin" />
        <span>Establishing connection to server...</span>
      </div>
    );
  };

  // Render the server list
  const renderServerList = () => {
    if (loading) {
      return (
        <div className="flex justify-center py-8">
          <ArrowPathIcon className="w-8 h-8 animate-spin" style={{ color: 'var(--color-primary)' }} />
        </div>
      );
    }

    if (servers.length === 0) {
      return (
        <div className="p-4 rounded-md mb-4" style={{ backgroundColor: 'var(--color-surface-light)' }}>
          <div className="flex items-start mb-3">
            <InformationCircleIcon className="w-5 h-5 mr-2 mt-0.5 flex-shrink-0" style={{ color: 'var(--color-primary)' }} />
            <div>
              <p className="mb-2" style={{ color: 'var(--color-text)' }}>
                No MCP servers configured. You can:
              </p>
              <ul className="list-disc pl-5 mb-3 text-sm" style={{ color: 'var(--color-text)' }}>
                <li className="mb-1">Create a direct connection below</li>
                <li className="mb-1">Add servers in the settings page</li>
              </ul>
              <button
                className="px-4 py-2 rounded-md text-sm flex items-center"
                style={{
                  backgroundColor: 'var(--color-primary)',
                  color: 'white'
                }}
                onClick={() => setShowDirectConnect(true)}
              >
                <PlusIcon className="w-4 h-4 mr-1" />
                Create Direct Connection
              </button>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="mb-4 overflow-auto max-h-64">
        {servers.map(server => (
          <div
            key={server.id}
            className={`p-3 mb-2 rounded-md flex items-center cursor-pointer transition-all ${
              selectedServerId === server.id
                ? 'border-l-2'
                : 'border-l-1 hover:border-l-2'
            }`}
            style={{
              backgroundColor: selectedServerId === server.id ? 'var(--color-surface-light)' : 'var(--color-surface-dark)',
              borderLeftColor: selectedServerId === server.id ? 'var(--color-primary)' : 'transparent',
              color: 'var(--color-text)'
            }}
            onClick={() => setSelectedServerId(server.id)}
          >
            <div className="flex-1">
              <div className="font-medium mb-1">{server.mcp_nickname}</div>
              <div className="text-xs flex items-center" style={{ color: 'var(--color-text-muted)' }}>
                <ServerIcon className="w-3 h-3 mr-1" />
                {server.mcp_host}:{server.mcp_port}
              </div>
            </div>
            <div className="flex items-center">
              {server.mcp_connection_status === 'connected' ? (
                <div className="flex items-center text-xs mr-2 px-2 py-1 rounded-full" style={{ 
                  backgroundColor: 'var(--color-success-bg)', 
                  color: 'var(--color-success)'
                }}>
                  <CheckCircleIcon className="w-3 h-3 mr-1" />
                  Connected
                </div>
              ) : server.mcp_connection_status === 'error' ? (
                <div className="flex items-center text-xs mr-2 px-2 py-1 rounded-full" style={{ 
                  backgroundColor: 'var(--color-error-bg)', 
                  color: 'var(--color-error)'
                }}>
                  <ExclamationCircleIcon className="w-3 h-3 mr-1" />
                  Error
                </div>
              ) : server.mcp_connection_status === 'available' ? (
                <div className="flex items-center text-xs mr-2 px-2 py-1 rounded-full" style={{ 
                  backgroundColor: 'var(--color-info-bg)',  // Use a different color for 'available'
                  color: 'var(--color-info)'
                }}>
                  <InformationCircleIcon className="w-3 h-3 mr-1" />
                  Available
                </div>
              ) : server.mcp_connection_status === 'connecting' ? (
                <div className="flex items-center text-xs mr-2 px-2 py-1 rounded-full" style={{ 
                  backgroundColor: 'var(--color-warning-bg)', 
                  color: 'var(--color-warning)'
                }}>
                  <ArrowPathIcon className="w-3 h-3 mr-1 animate-spin" />
                  Connecting
                </div>
              ) : (
                <div className="flex items-center text-xs mr-2 px-2 py-1 rounded-full" style={{ 
                  backgroundColor: 'var(--color-primary-translucent)', 
                  color: 'var(--color-text-muted)'
                }}>
                  Unknown
                </div>
              )}
              <button
                className="p-1.5 rounded-md transition-colors"
                style={{
                  backgroundColor: 'var(--color-surface)',
                  color: 'var(--color-text-muted)'
                }}
                onClick={e => {
                  e.stopPropagation();
                  testConnection(server);
                }}
                disabled={testingServer === server.id || connecting}
              >
                {testingServer === server.id ? (
                  <ArrowPathIcon className="w-4 h-4 animate-spin" />
                ) : (
                  <ArrowPathIcon className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>
        ))}
      </div>
    );
  };

  // Render the dialog content
  return (
    <div
      className={`fixed inset-0 flex items-center justify-center z-50 transition-opacity duration-300 ${
        isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
      }`}
      style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}
    >
      <div
        className="relative max-w-lg w-full mx-4 rounded-lg shadow-xl transition-all transform"
        style={{
          backgroundColor: 'var(--color-surface)',
          color: 'var(--color-text)',
          backdropFilter: 'blur(8px)',
          border: '1px solid var(--color-surface-light)'
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b" style={{ borderColor: 'var(--color-border)' }}>
          <h3 className="text-lg font-medium">Select MCP Server</h3>
          <button
            className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700"
            onClick={onClose}
          >
            <XMarkIcon className="w-5 h-5" style={{ color: 'var(--color-text-muted)' }} />
          </button>
        </div>

        {/* Body */}
        <div className="p-4">
          {/* Connection status indicator */}
          {renderConnectionStatus()}
          
          {/* SSE Connection Status */}
          {sseConnectionStatus && !connecting && (
            <div className={`mb-4 p-3 rounded-md text-sm flex items-start ${
              sseConnectionStatus === 'connected' 
                ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400'
                : sseConnectionStatus === 'error'
                  ? 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400'
                  : 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400'
            }`}>
              {sseConnectionStatus === 'connected' ? (
                <CheckCircleIcon className="w-5 h-5 mr-2 flex-shrink-0" />
              ) : sseConnectionStatus === 'error' ? (
                <ExclamationCircleIcon className="w-5 h-5 mr-2 flex-shrink-0" />
              ) : (
                <InformationCircleIcon className="w-5 h-5 mr-2 flex-shrink-0" />
              )}
              <div>
                <p className="font-medium">
                  {sseConnectionStatus === 'connected' 
                    ? 'Connected to MCP Server'
                    : sseConnectionStatus === 'error'
                      ? 'MCP Connection Error'
                      : sseConnectionStatus === 'connecting'
                        ? 'Connecting to MCP Server...'
                        : 'MCP Connection Status'}
                </p>
                <p className="text-xs mt-1">
                  {sseConnectionStatus === 'connected' 
                    ? `Successfully established SSE connection${mcpConnection.clientId ? ` with client ID: ${mcpConnection.clientId}` : ''}`
                    : sseConnectionStatus === 'error'
                      ? 'Failed to establish SSE connection. Please try another server or check your network.'
                      : sseConnectionStatus === 'connecting'
                        ? 'Attempting to establish direct connection with the SSE endpoint...'
                        : 'No active MCP connection.'}
                </p>
                {mcpConnection.clientId && sseConnectionStatus === 'connected' && (
                  <p className="text-xs mt-1 font-medium">Client ID: {mcpConnection.clientId}</p>
                )}
              </div>
            </div>
          )}

          {/* Instructions */}
          <div className="mb-4 p-3 rounded-md text-sm" style={{ backgroundColor: 'var(--color-primary-translucent)' }}>
            <div className="flex items-start">
              <InformationCircleIcon className="w-5 h-5 mr-2 mt-0.5 flex-shrink-0" style={{ color: 'var(--color-primary)' }} />
              <div>
                <p className="mb-1 font-medium">Select an MCP server to connect to</p>
                <ol className="list-decimal pl-5 space-y-1">
                  <li>Select a server from the list</li>
                  <li>Click the refresh button to test and automatically connect to the server</li>
                </ol>
              </div>
            </div>
          </div>

          {/* Error message */}
          {error && (
            <div className="mb-4 p-3 rounded-md text-sm flex items-start" style={{ backgroundColor: 'var(--color-error-bg)', color: 'var(--color-error)' }}>
              <ExclamationCircleIcon className="w-5 h-5 mr-2 mt-0.5 flex-shrink-0" />
              <div>{error}</div>
            </div>
          )}

          {/* Server list section */}
          {!showDirectConnect && (
            <>
              <div className="mb-2 flex justify-between items-center">
                <h4 className="font-medium">Available Servers</h4>
                <button
                  className="text-xs px-2 py-1 rounded-md flex items-center"
                  style={{
                    backgroundColor: 'var(--color-surface-light)',
                    color: 'var(--color-text-muted)'
                  }}
                  onClick={fetchServers}
                >
                  <ArrowPathIcon className="w-3 h-3 mr-1" />
                  Refresh
                </button>
              </div>

              {renderServerList()}

              <div className="flex items-center justify-between mt-4">
                <button
                  className="px-3 py-1.5 rounded-md flex items-center text-sm"
                  style={{
                    backgroundColor: 'var(--color-surface-light)',
                    color: 'var(--color-text-muted)'
                  }}
                  onClick={() => setShowDirectConnect(true)}
                >
                  <PlusIcon className="w-4 h-4 mr-1" />
                  New Connection
                </button>
              </div>
            </>
          )}

          {/* Direct connection form */}
          {showDirectConnect && (
            <div>
              <div className="mb-2 flex justify-between items-center">
                <h4 className="font-medium">Direct Connection</h4>
                <button
                  className="text-xs px-2 py-1 rounded-md flex items-center"
                  style={{
                    backgroundColor: 'var(--color-surface-light)',
                    color: 'var(--color-text-muted)'
                  }}
                  onClick={() => setShowDirectConnect(false)}
                >
                  Back to Server List
                </button>
              </div>

              <div className="mb-3">
                <label className="block text-sm font-medium mb-1" style={{ color: 'var(--color-text-muted)' }}>
                  Server Name
                </label>
                <input
                  type="text"
                  placeholder="My MCP Server"
                  value={directName}
                  onChange={e => setDirectName(e.target.value)}
                  className="w-full p-2 rounded-md"
                  style={{
                    backgroundColor: 'var(--color-surface-dark)',
                    color: 'var(--color-text)',
                    border: '1px solid var(--color-border)'
                  }}
                />
              </div>

              <div className="mb-3">
                <label className="block text-sm font-medium mb-1" style={{ color: 'var(--color-text-muted)' }}>
                  Host Address
                </label>
                <input
                  type="text"
                  placeholder="192.168.1.100 or localhost"
                  value={directHost}
                  onChange={e => setDirectHost(e.target.value)}
                  className="w-full p-2 rounded-md"
                  style={{
                    backgroundColor: 'var(--color-surface-dark)',
                    color: 'var(--color-text)',
                    border: '1px solid var(--color-border)'
                  }}
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-1" style={{ color: 'var(--color-text-muted)' }}>
                  Port
                </label>
                <input
                  type="text"
                  placeholder={defaultMCPPort || '8080'}
                  value={directPort}
                  onChange={e => setDirectPort(e.target.value)}
                  className="w-full p-2 rounded-md"
                  style={{
                    backgroundColor: 'var(--color-surface-dark)',
                    color: 'var(--color-text)',
                    border: '1px solid var(--color-border)'
                  }}
                />
              </div>

              <div className="mt-4 text-center">
                <button
                  className="px-4 py-2 w-full rounded-md flex items-center justify-center text-sm"
                  style={{
                    backgroundColor: 'var(--color-primary)',
                    color: 'white',
                    opacity: (directConnecting || connecting || loading) ? 0.7 : 1
                  }}
                  onClick={handleDirectConnect}
                  disabled={directConnecting || connecting || loading}
                >
                  {directConnecting || (connecting && !selectedServerId) ? (
                    <>
                      <ArrowPathIcon className="w-4 h-4 mr-1 animate-spin" />
                      Testing & Connecting...
                    </>
                  ) : (
                    <>
                      <ServerIcon className="w-4 h-4 mr-1" />
                      Test & Connect
                    </>
                  )}
                </button>
              </div>

              {/* Direct SSE Connection information - Can be removed or rephrased */}
              <div className="mt-4 p-3 rounded-md text-xs" style={{ backgroundColor: 'var(--color-primary-translucent)' }}>
                <div className="flex items-start">
                  <InformationCircleIcon className="w-4 h-4 mr-1 mt-0.5 flex-shrink-0" style={{ color: 'var(--color-primary)' }} />
                  <div>
                    <p className="mb-1">Test & Connect</p>
                    <p>Clicking this button will test the server and automatically connect if it's available.</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MCPServerSelector;

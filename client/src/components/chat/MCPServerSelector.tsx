import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import {
  XMarkIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  ServerIcon,
  PlusIcon,
  InformationCircleIcon,
  ArrowRightCircleIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';
import { useMCP } from '../../contexts/MCPContext';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import { useWebSocket } from '../../contexts/WebSocketContext';
import { useTheme } from '../../contexts/ThemeContext';

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
  const { currentTheme } = useTheme();
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

  // Get theme colors
  const isDarkMode = currentTheme === 'dark';
  const isMidnight = currentTheme === 'midnight';
  const isLight = currentTheme === 'light';
  
  // Use CSS variables for consistent theming
  const textColor = isLight ? '#1e293b' : isMidnight ? '#f9fafb' : '#f3f4f6';
  const mutedTextColor = isLight ? 'rgba(71, 85, 105, 0.8)' : isMidnight ? 'rgba(249, 250, 251, 0.7)' : 'rgba(156, 163, 175, 0.8)';
  const placeholderColor = isLight ? 'rgba(71, 85, 105, 0.6)' : isMidnight ? 'rgba(249, 250, 251, 0.5)' : 'rgba(156, 163, 175, 0.5)';
  
  const cardBackground = isLight
    ? 'linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.9))'
    : isMidnight 
      ? 'linear-gradient(135deg, rgba(17, 24, 39, 0.8), rgba(15, 23, 42, 0.6))'
      : 'linear-gradient(135deg, rgba(35, 42, 61, 0.8), rgba(21, 28, 44, 0.6))';
      
  const cardBorder = isLight
    ? '1px solid rgba(148, 163, 184, 0.3)'
    : isMidnight
      ? '1px solid rgba(55, 65, 81, 0.4)'
      : '1px solid rgba(42, 51, 73, 0.4)';
      
  const headerBackground = isLight
    ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.06), rgba(139, 92, 246, 0.06))'
    : isMidnight
      ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(236, 72, 153, 0.1))'
      : 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1))';
      
  const statusColors = {
    connected: {
      bg: isLight 
        ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(16, 185, 129, 0.06))'
        : isMidnight
          ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1))'
          : 'linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.08))',
      border: isLight ? 'rgba(16, 185, 129, 0.4)' : 'rgba(16, 185, 129, 0.3)',
      text: isLight ? '#065f46' : isMidnight ? '#6ee7b7' : '#d1fae5'
    },
    error: {
      bg: isLight 
        ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(239, 68, 68, 0.06))'
        : isMidnight
          ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1))'
          : 'linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.08))',
      border: isLight ? 'rgba(239, 68, 68, 0.4)' : 'rgba(239, 68, 68, 0.3)',
      text: isLight ? '#991b1b' : isMidnight ? '#fca5a5' : '#fecaca'
    },
    connecting: {
      bg: isLight 
        ? 'linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(245, 158, 11, 0.06))'
        : isMidnight
          ? 'linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1))'
          : 'linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.08))',
      border: isLight ? 'rgba(245, 158, 11, 0.4)' : 'rgba(245, 158, 11, 0.3)',
      text: isLight ? '#92400e' : isMidnight ? '#fcd34d' : '#fef3c7'
    }
  };

  // Inject sophisticated animations
  useEffect(() => {
    const style = document.createElement('style');
    style.innerHTML = `
      @keyframes mcpGlow {
        0%, 100% { 
          box-shadow: 0 0 20px ${isLight ? 'rgba(99, 102, 241, 0.15)' : isMidnight ? 'rgba(139, 92, 246, 0.4)' : 'rgba(99, 102, 241, 0.3)'};
        }
        50% { 
          box-shadow: 0 0 30px ${isLight ? 'rgba(99, 102, 241, 0.25)' : isMidnight ? 'rgba(139, 92, 246, 0.6)' : 'rgba(99, 102, 241, 0.5)'};
        }
      }

      @keyframes mcpShimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
      }

      @keyframes mcpFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-2px); }
      }

      .mcp-glow { 
        animation: mcpGlow 3s infinite;
        overflow: hidden;
        /* Prevent scrollbar by containing layout */
        contain: layout style paint;
      }
      .mcp-shimmer {
        background: linear-gradient(45deg, transparent 30%, ${isLight ? 'rgba(99, 102, 241, 0.08)' : isMidnight ? 'rgba(139, 92, 246, 0.15)' : 'rgba(255,255,255,0.1)'} 50%, transparent 70%);
        background-size: 200% 100%;
        animation: mcpShimmer 4s infinite;
      }
      .mcp-float { 
        animation: mcpFloat 4s ease-in-out infinite;
        will-change: transform;
        /* Prevent layout shifts */
        contain: layout;
      }

      .mcp-scrollbar::-webkit-scrollbar {
        width: 6px;
      }
      
      .mcp-scrollbar::-webkit-scrollbar-track {
        background: ${isLight ? 'rgba(148, 163, 184, 0.2)' : isMidnight ? 'rgba(55, 65, 81, 0.3)' : 'rgba(42, 51, 73, 0.3)'};
        border-radius: 3px;
      }
      
      .mcp-scrollbar::-webkit-scrollbar-thumb {
        background: ${isLight ? 'rgba(148, 163, 184, 0.5)' : isMidnight ? 'rgba(55, 65, 81, 0.6)' : 'rgba(42, 51, 73, 0.6)'};
        border-radius: 3px;
        transition: all 0.2s ease;
      }
      
      .mcp-scrollbar::-webkit-scrollbar-thumb:hover {
        background: ${isLight ? 'rgba(148, 163, 184, 0.7)' : isMidnight ? 'rgba(55, 65, 81, 0.8)' : 'rgba(42, 51, 73, 0.8)'};
      }

      /* Theme-aware placeholder styles */
      .mcp-input::placeholder {
        color: ${placeholderColor} !important;
        opacity: 1;
      }
      
      .mcp-input::-webkit-input-placeholder {
        color: ${placeholderColor} !important;
      }
      
      .mcp-input::-moz-placeholder {
        color: ${placeholderColor} !important;
        opacity: 1;
      }
      
      .mcp-input:-ms-input-placeholder {
        color: ${placeholderColor} !important;
      }
    `;
    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, [currentTheme, isLight, isMidnight, placeholderColor]);

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
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="mb-4 p-4 rounded-xl backdrop-blur-md"
          style={{
            background: statusColors.connected.bg,
            border: `1px solid ${statusColors.connected.border}`,
          }}
        >
          <div className="flex items-center space-x-3">
            <div 
              className="w-8 h-8 rounded-full flex items-center justify-center"
              style={{
                background: '#10b981',
                boxShadow: '0 0 15px rgba(16, 185, 129, 0.4)',
              }}
            >
              <CheckCircleIcon className="w-4 h-4 text-white" />
            </div>
            <div>
              <p className="font-semibold" style={{ color: statusColors.connected.text }}>Direct SSE Connection Successful!</p>
              <p className="text-sm mt-1" style={{ color: mutedTextColor }}>Client ID: {directSseSuccess}</p>
            </div>
          </div>
        </motion.div>
      );
    }
    
    if (!connecting) return null;
    
    if (!wsConnected || !isFullyReady) {
      return (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-4 rounded-xl backdrop-blur-md"
          style={{
            background: isLight 
              ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05))'
              : isMidnight
                ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(99, 102, 241, 0.1))'
                : 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(59, 130, 246, 0.08))',
            border: `1px solid ${isLight ? 'rgba(59, 130, 246, 0.4)' : 'rgba(59, 130, 246, 0.3)'}`,
          }}
        >
          <div className="flex items-center space-x-3">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              className="w-8 h-8 rounded-full flex items-center justify-center"
              style={{
                background: '#3b82f6',
                boxShadow: '0 0 15px rgba(59, 130, 246, 0.4)',
              }}
            >
              <ArrowPathIcon className="w-4 h-4 text-white" />
            </motion.div>
            <div>
              <p className="font-semibold" style={{ color: isLight ? '#1e3a8a' : isMidnight ? '#93c5fd' : '#dbeafe' }}>Initializing WebSocket connection...</p>
              <p className="text-sm mt-1" style={{ color: mutedTextColor }}>This step is required before connecting to the MCP server.</p>
            </div>
          </div>
        </motion.div>
      );
    }
    
    if (connectionPhase === 'mcp') {
      return (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-4 rounded-xl backdrop-blur-md"
          style={{
            background: isLight 
              ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05))'
              : isMidnight
                ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(99, 102, 241, 0.1))'
                : 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(59, 130, 246, 0.08))',
            border: `1px solid ${isLight ? 'rgba(59, 130, 246, 0.4)' : 'rgba(59, 130, 246, 0.3)'}`,
          }}
        >
          <div className="flex items-center space-x-3">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              className="w-8 h-8 rounded-full flex items-center justify-center"
              style={{
                background: '#3b82f6',
                boxShadow: '0 0 15px rgba(59, 130, 246, 0.4)',
              }}
            >
              <ArrowPathIcon className="w-4 h-4 text-white" />
            </motion.div>
            <div>
              <p className="font-semibold" style={{ color: isLight ? '#1e3a8a' : isMidnight ? '#93c5fd' : '#dbeafe' }}>Establishing MCP connection...</p>
              <p className="text-sm mt-1" style={{ color: mutedTextColor }}>Connecting to server and acquiring client ID.</p>
            </div>
          </div>
        </motion.div>
      );
    }
    
    if (connectionPhase === 'complete') {
      return (
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="mb-4 p-4 rounded-xl backdrop-blur-md"
          style={{
            background: statusColors.connected.bg,
            border: `1px solid ${statusColors.connected.border}`,
          }}
        >
          <div className="flex items-center space-x-3">
            <div 
              className="w-8 h-8 rounded-full flex items-center justify-center"
              style={{
                background: '#10b981',
                boxShadow: '0 0 15px rgba(16, 185, 129, 0.4)',
              }}
            >
              <CheckCircleIcon className="w-4 h-4 text-white" />
            </div>
            <div>
              <p className="font-semibold" style={{ color: statusColors.connected.text }}>Connection successful!</p>
              <p className="text-sm mt-1" style={{ color: mutedTextColor }}>Client ID acquired. Redirecting to chat...</p>
            </div>
          </div>
        </motion.div>
      );
    }
    
    return (
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-4 p-4 rounded-xl backdrop-blur-md"
        style={{
          background: isLight 
            ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05))'
            : isMidnight
              ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(99, 102, 241, 0.1))'
              : 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(59, 130, 246, 0.08))',
          border: `1px solid ${isLight ? 'rgba(59, 130, 246, 0.4)' : 'rgba(59, 130, 246, 0.3)'}`,
        }}
      >
        <div className="flex items-center space-x-3">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            className="w-8 h-8 rounded-full flex items-center justify-center"
            style={{
              background: '#3b82f6',
              boxShadow: '0 0 15px rgba(59, 130, 246, 0.4)',
            }}
          >
            <ArrowPathIcon className="w-4 h-4 text-white" />
          </motion.div>
          <span style={{ color: isLight ? '#1e3a8a' : isMidnight ? '#93c5fd' : '#dbeafe' }}>Establishing connection to server...</span>
        </div>
      </motion.div>
    );
  };

  // Render the server list
  const renderServerList = () => {
    if (loading) {
      return (
        <div className="flex justify-center py-12">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            className="w-12 h-12 rounded-full flex items-center justify-center"
            style={{
              background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
              boxShadow: '0 0 25px rgba(99, 102, 241, 0.4)',
            }}
          >
            <ArrowPathIcon className="w-6 h-6 text-white" />
          </motion.div>
        </div>
      );
    }

    if (servers.length === 0) {
      return (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 rounded-xl mb-4 backdrop-blur-md"
          style={{
            background: isLight 
              ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.06), rgba(139, 92, 246, 0.06))'
              : isMidnight
                ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(236, 72, 153, 0.1))'
                : 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1))',
            border: `1px solid ${isLight ? 'rgba(99, 102, 241, 0.4)' : 'rgba(99, 102, 241, 0.3)'}`,
          }}
        >
          <div className="flex items-start space-x-3 mb-4">
            <div 
              className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
              style={{
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                boxShadow: '0 0 15px rgba(99, 102, 241, 0.4)',
              }}
            >
              <InformationCircleIcon className="w-4 h-4 text-white" />
            </div>
            <div>
              <p className="font-semibold mb-2" style={{ color: textColor }}>No MCP servers configured</p>
              <p className="text-sm mb-4" style={{ color: mutedTextColor }}>You can:</p>
              <ul className="list-disc pl-5 mb-4 text-sm space-y-1" style={{ color: mutedTextColor }}>
                <li>Create a direct connection below</li>
                <li>Add servers in the settings page</li>
              </ul>
              <motion.button
                onClick={() => setShowDirectConnect(true)}
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{
                  background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                  color: 'white',
                  boxShadow: '0 0 15px rgba(99, 102, 241, 0.3)',
                }}
                whileHover={{ 
                  scale: 1.05,
                  boxShadow: '0 0 20px rgba(99, 102, 241, 0.4)'
                }}
                whileTap={{ scale: 0.95 }}
              >
                <PlusIcon className="w-4 h-4 mr-2 inline" />
                Create Direct Connection
              </motion.button>
            </div>
          </div>
        </motion.div>
      );
    }

    // Return just the server items without additional container
    return servers.map((server, index) => (
      <motion.div
        key={server.id}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1 }}
        className={`p-4 rounded-xl backdrop-blur-md cursor-pointer transition-all ${
          selectedServerId === server.id ? 'mcp-glow' : ''
        }`}
        style={{
          background: selectedServerId === server.id 
            ? (isLight 
               ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.12))'
               : isMidnight
                 ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(236, 72, 153, 0.2))'
                 : 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2))')
            : (isLight 
               ? 'linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(248, 250, 252, 0.6))'
               : isMidnight
                 ? 'linear-gradient(135deg, rgba(17, 24, 39, 0.6), rgba(15, 23, 42, 0.4))'
                 : 'linear-gradient(135deg, rgba(35, 42, 61, 0.6), rgba(21, 28, 44, 0.4))'),
          border: selectedServerId === server.id 
            ? `1px solid ${isLight ? 'rgba(99, 102, 241, 0.5)' : 'rgba(99, 102, 241, 0.4)'}`
            : cardBorder,
          borderLeft: selectedServerId === server.id 
            ? `4px solid ${isMidnight ? '#a78bfa' : '#6366f1'}`
            : '4px solid transparent',
        }}
        onClick={() => setSelectedServerId(server.id)}
        whileHover={{ 
          scale: 1.01,
          boxShadow: isLight 
            ? '0 6px 20px rgba(0, 0, 0, 0.06)'
            : isMidnight
              ? '0 8px 25px rgba(139, 92, 246, 0.15)'
              : '0 8px 25px rgba(0, 0, 0, 0.15)'
        }}
      >
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="font-semibold mb-2" style={{ color: textColor }}>{server.mcp_nickname}</div>
            <div className="text-sm flex items-center" style={{ color: mutedTextColor }}>
              <ServerIcon className="w-4 h-4 mr-2" />
              {server.mcp_host}:{server.mcp_port}
            </div>
            {server.tools_count !== undefined && (
              <div className="text-xs mt-1" style={{ color: mutedTextColor, opacity: 0.8 }}>
                {server.tools_count} tools available
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Status Badge */}
            <div 
              className="px-3 py-1 rounded-lg text-xs font-medium"
              style={{
                background: server.mcp_connection_status === 'connected' 
                  ? statusColors.connected.bg
                  : server.mcp_connection_status === 'error'
                    ? statusColors.error.bg
                    : server.mcp_connection_status === 'connecting'
                      ? statusColors.connecting.bg
                      : (isLight 
                         ? 'linear-gradient(135deg, rgba(107, 114, 128, 0.12), rgba(107, 114, 128, 0.06))'
                         : isMidnight
                           ? 'linear-gradient(135deg, rgba(156, 163, 175, 0.2), rgba(156, 163, 175, 0.1))'
                           : 'linear-gradient(135deg, rgba(107, 114, 128, 0.15), rgba(107, 114, 128, 0.08))'),
                border: `1px solid ${
                  server.mcp_connection_status === 'connected' ? statusColors.connected.border :
                  server.mcp_connection_status === 'error' ? statusColors.error.border :
                  server.mcp_connection_status === 'connecting' ? statusColors.connecting.border :
                  (isLight ? 'rgba(107, 114, 128, 0.4)' : 'rgba(107, 114, 128, 0.3)')
                }`,
                color: server.mcp_connection_status === 'connected' ? statusColors.connected.text :
                       server.mcp_connection_status === 'error' ? statusColors.error.text :
                       server.mcp_connection_status === 'connecting' ? statusColors.connecting.text :
                       (isLight ? '#4b5563' : isMidnight ? '#d1d5db' : '#e5e7eb'),
              }}
            >
              <div className="flex items-center space-x-1">
                {server.mcp_connection_status === 'connected' ? (
                  <CheckCircleIcon className="w-3 h-3" />
                ) : server.mcp_connection_status === 'error' ? (
                  <ExclamationCircleIcon className="w-3 h-3" />
                ) : server.mcp_connection_status === 'connecting' ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  >
                    <ArrowPathIcon className="w-3 h-3" />
                  </motion.div>
                ) : (
                  <InformationCircleIcon className="w-3 h-3" />
                )}
                <span>
                  {server.mcp_connection_status === 'connected' ? 'Connected' :
                   server.mcp_connection_status === 'error' ? 'Error' :
                   server.mcp_connection_status === 'connecting' ? 'Connecting' :
                   server.mcp_connection_status === 'available' ? 'Available' :
                   'Unknown'}
                </span>
              </div>
            </div>

            {/* Test Button */}
            <motion.button
              onClick={e => {
                e.stopPropagation();
                testConnection(server);
              }}
              disabled={testingServer === server.id || connecting}
              className="p-2 rounded-lg backdrop-blur-md"
              style={{
                background: isLight ? 'rgba(148, 163, 184, 0.15)' : isMidnight ? 'rgba(55, 65, 81, 0.3)' : 'rgba(42, 51, 73, 0.25)',
                border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
                opacity: (testingServer === server.id || connecting) ? 0.5 : 1,
              }}
              whileHover={!(testingServer === server.id || connecting) ? { 
                scale: 1.05,
                background: isLight ? 'rgba(148, 163, 184, 0.25)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.35)'
              } : {}}
              whileTap={!(testingServer === server.id || connecting) ? { scale: 0.9 } : {}}
              title="Test connection"
            >
              {testingServer === server.id ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                >
                  <ArrowPathIcon className="w-4 h-4" style={{ color: textColor }} />
                </motion.div>
              ) : (
                <ArrowPathIcon className="w-4 h-4" style={{ color: textColor }} />
              )}
            </motion.button>
          </div>
        </div>
      </motion.div>
    ));
  };

  // Render the dialog content
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 flex items-center justify-center z-50"
          style={{ backgroundColor: 'rgba(0, 0, 0, 0.6)' }}
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: 'spring', duration: 0.5 }}
            className="relative max-w-2xl w-full mx-4 rounded-2xl shadow-2xl overflow-hidden"
            style={{
              background: cardBackground,
              backdropFilter: 'blur(20px)',
              border: cardBorder,
              boxShadow: isLight 
                ? '0 25px 50px rgba(0, 0, 0, 0.08)'
                : isMidnight
                  ? '0 25px 50px rgba(0, 0, 0, 0.4)'
                  : '0 25px 50px rgba(0, 0, 0, 0.25)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Background Pattern */}
            <motion.div
              className="absolute inset-0 opacity-20"
              style={{
                background: isLight 
                  ? 'linear-gradient(45deg, rgba(99, 102, 241, 0.03), transparent, rgba(139, 92, 246, 0.03))'
                  : isMidnight
                    ? 'linear-gradient(45deg, rgba(139, 92, 246, 0.15), transparent, rgba(236, 72, 153, 0.1))'
                    : 'linear-gradient(45deg, rgba(99, 102, 241, 0.1), transparent, rgba(139, 92, 246, 0.1))',
                backgroundSize: '200% 200%',
              }}
              animate={{
                backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
              }}
              transition={{
                duration: 10,
                repeat: Infinity,
                ease: 'linear',
              }}
            />

            {/* Header */}
            <div
              className="p-6 border-b relative overflow-hidden"
              style={{
                background: headerBackground,
                borderBottom: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.2)' : isMidnight ? 'rgba(55, 65, 81, 0.3)' : 'rgba(42, 51, 73, 0.3)'}`,
              }}
            >
              {/* Header shine effect */}
              <div className="absolute inset-0 mcp-shimmer opacity-30" />
              
              <div className="flex items-center justify-between relative z-10">
                <div className="flex items-center space-x-3">
                  <motion.div
                    className="w-10 h-10 rounded-full flex items-center justify-center mcp-glow"
                    style={{
                      background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                      boxShadow: '0 0 25px rgba(99, 102, 241, 0.4)',
                    }}
                    animate={{ rotate: 360 }}
                    transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                  >
                    <SparklesIcon className="w-5 h-5 text-white" />
                  </motion.div>

                  <div className="flex flex-col">
                    <h3 className="text-xl font-bold" style={{ color: textColor }}>Select MCP Server</h3>
                    <p className="text-sm" style={{ color: mutedTextColor }}>Connect to your Model Context Protocol server</p>
                  </div>
                </div>

                <motion.button
                  onClick={onClose}
                  className="p-2 rounded-lg backdrop-blur-md"
                  style={{
                    background: isLight ? 'rgba(148, 163, 184, 0.15)' : isMidnight ? 'rgba(55, 65, 81, 0.25)' : 'rgba(42, 51, 73, 0.2)',
                    border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
                  }}
                  whileHover={{ 
                    scale: 1.02, // Reduced scaling
                    background: isLight ? 'rgba(148, 163, 184, 0.25)' : isMidnight ? 'rgba(55, 65, 81, 0.35)' : 'rgba(42, 51, 73, 0.3)'
                  }}
                  whileTap={{ scale: 0.98 }}
                >
                  <XMarkIcon className="w-5 h-5" style={{ color: textColor }} />
                </motion.button>
              </div>
            </div>

            {/* Body */}
            <div className="p-6 overflow-y-auto mcp-scrollbar" style={{ maxHeight: 'calc(90vh - 200px)' }}>
              {/* Connection status indicator */}
              {renderConnectionStatus()}
              
              {/* SSE Connection Status */}
              {sseConnectionStatus && !connecting && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mb-4 p-4 rounded-xl backdrop-blur-md"
                  style={{
                    background: sseConnectionStatus === 'connected' 
                      ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05))'
                      : sseConnectionStatus === 'error'
                        ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05))'
                        : 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05))',
                    border: `1px solid ${
                      sseConnectionStatus === 'connected' ? 'rgba(16, 185, 129, 0.3)' :
                      sseConnectionStatus === 'error' ? 'rgba(239, 68, 68, 0.3)' :
                      'rgba(59, 130, 246, 0.3)'
                    }`,
                  }}
                >
                  <div className="flex items-start space-x-3">
                    <div 
                      className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                      style={{
                        background: sseConnectionStatus === 'connected' ? '#10b981' :
                                   sseConnectionStatus === 'error' ? '#ef4444' : '#3b82f6',
                        boxShadow: `0 0 15px ${
                          sseConnectionStatus === 'connected' ? 'rgba(16, 185, 129, 0.4)' :
                          sseConnectionStatus === 'error' ? 'rgba(239, 68, 68, 0.4)' :
                          'rgba(59, 130, 246, 0.4)'
                        }`,
                      }}
                    >
                      {sseConnectionStatus === 'connected' ? (
                        <CheckCircleIcon className="w-4 h-4 text-white" />
                      ) : sseConnectionStatus === 'error' ? (
                        <ExclamationCircleIcon className="w-4 h-4 text-white" />
                      ) : (
                        <InformationCircleIcon className="w-4 h-4 text-white" />
                      )}
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold mb-1" style={{ color: textColor }}>
                        {sseConnectionStatus === 'connected' 
                          ? 'Connected to MCP Server'
                          : sseConnectionStatus === 'error'
                            ? 'MCP Connection Error'
                            : sseConnectionStatus === 'connecting'
                              ? 'Connecting to MCP Server...'
                              : 'MCP Connection Status'}
                      </p>
                      <p className="text-sm leading-relaxed" style={{ color: mutedTextColor }}>
                        {sseConnectionStatus === 'connected' 
                          ? `Successfully established SSE connection${mcpConnection.clientId ? ` with client ID: ${mcpConnection.clientId}` : ''}`
                          : sseConnectionStatus === 'error'
                            ? 'Failed to establish SSE connection. Please try another server or check your network.'
                            : sseConnectionStatus === 'connecting'
                              ? 'Attempting to establish direct connection with the SSE endpoint...'
                              : 'No active MCP connection.'}
                      </p>
                      {mcpConnection.clientId && sseConnectionStatus === 'connected' && (
                        <div className="mt-2 px-3 py-1 rounded-md bg-white/10 inline-block">
                          <p className="text-xs font-mono" style={{ color: textColor }}>Client ID: {mcpConnection.clientId}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Instructions */}
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="mb-4 p-4 rounded-xl backdrop-blur-md"
                style={{
                  background: isLight 
                    ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.06), rgba(139, 92, 246, 0.06))'
                    : isMidnight
                      ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(236, 72, 153, 0.1))'
                      : 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1))',
                  border: `1px solid ${isLight ? 'rgba(99, 102, 241, 0.4)' : 'rgba(99, 102, 241, 0.3)'}`,
                }}
              >
                <div className="flex items-start space-x-3">
                  <div 
                    className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                    style={{
                      background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                      boxShadow: '0 0 15px rgba(99, 102, 241, 0.4)',
                    }}
                  >
                    <InformationCircleIcon className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold mb-2" style={{ color: textColor }}>How to Connect</p>
                    <ol className="list-decimal pl-5 space-y-1 text-sm" style={{ color: mutedTextColor }}>
                      <li>Select a server from the list below</li>
                      <li>Click the test button (🔄) to verify connection</li>
                      <li>Connection will be established automatically after successful test</li>
                    </ol>
                  </div>
                </div>
              </motion.div>

              {/* Error message */}
              {error && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="mb-4 p-4 rounded-xl backdrop-blur-md"
                  style={{
                    background: statusColors.error.bg,
                    border: `1px solid ${statusColors.error.border}`,
                  }}
                >
                  <div className="flex items-start space-x-3">
                    <ExclamationCircleIcon className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: statusColors.error.text }} />
                    <div className="text-sm leading-relaxed" style={{ color: statusColors.error.text }}>{error}</div>
                  </div>
                </motion.div>
              )}

              {/* Server list section */}
              {!showDirectConnect && (
                <>
                  <div className="mb-4 flex justify-between items-center">
                    <h4 className="text-lg font-semibold" style={{ color: textColor }}>Available Servers</h4>
                    <motion.button
                      onClick={fetchServers}
                      className="px-3 py-2 rounded-lg backdrop-blur-md text-sm font-medium"
                      style={{
                        background: isLight ? 'rgba(148, 163, 184, 0.15)' : isMidnight ? 'rgba(55, 65, 81, 0.25)' : 'rgba(42, 51, 73, 0.2)',
                        border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
                        color: textColor,
                      }}
                      whileHover={{ 
                        scale: 1.02,
                        background: isLight ? 'rgba(148, 163, 184, 0.25)' : isMidnight ? 'rgba(55, 65, 81, 0.35)' : 'rgba(42, 51, 73, 0.3)'
                      }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <ArrowPathIcon className="w-4 h-4 mr-1 inline" />
                      Refresh
                    </motion.button>
                  </div>

                  {/* Smart server list with conditional scrolling */}
                  <div 
                    className="space-y-3 overflow-y-auto mcp-scrollbar" 
                    style={{ 
                      maxHeight: servers.length > 2 ? '24rem' : 'auto',
                      minHeight: servers.length > 0 ? 'auto' : '8rem'
                    }}
                  >
                    {renderServerList()}
                  </div>

                  <div className="flex items-center justify-center mt-6">
                    <motion.button
                      onClick={() => setShowDirectConnect(true)}
                      className="px-6 py-3 rounded-xl backdrop-blur-md text-sm font-medium"
                      style={{
                        background: isLight ? 'rgba(148, 163, 184, 0.15)' : isMidnight ? 'rgba(55, 65, 81, 0.25)' : 'rgba(42, 51, 73, 0.2)',
                        border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
                        color: textColor,
                      }}
                      whileHover={{ 
                        scale: 1.02,
                        background: isLight ? 'rgba(148, 163, 184, 0.25)' : isMidnight ? 'rgba(55, 65, 81, 0.35)' : 'rgba(42, 51, 73, 0.3)'
                      }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <PlusIcon className="w-4 h-4 mr-2 inline" />
                      Create New Connection
                    </motion.button>
                  </div>
                </>
              )}

              {/* Direct connection form */}
              {showDirectConnect && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-4"
                >
                  <div className="flex justify-between items-center">
                    <h4 className="text-lg font-semibold" style={{ color: textColor }}>Create Direct Connection</h4>
                    <motion.button
                      onClick={() => setShowDirectConnect(false)}
                      className="px-3 py-2 rounded-lg backdrop-blur-md text-sm"
                      style={{
                        background: isLight ? 'rgba(148, 163, 184, 0.15)' : isMidnight ? 'rgba(55, 65, 81, 0.25)' : 'rgba(42, 51, 73, 0.2)',
                        border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
                        color: textColor,
                      }}
                      whileHover={{ 
                        scale: 1.02,
                        background: isLight ? 'rgba(148, 163, 184, 0.25)' : isMidnight ? 'rgba(55, 65, 81, 0.35)' : 'rgba(42, 51, 73, 0.3)'
                      }}
                      whileTap={{ scale: 0.95 }}
                    >
                      ← Back to List
                    </motion.button>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-2" style={{ color: mutedTextColor }}>
                        Server Name
                      </label>
                      <motion.input
                        type="text"
                        placeholder="My MCP Server"
                        value={directName}
                        onChange={e => setDirectName(e.target.value)}
                        className="w-full p-3 rounded-xl backdrop-blur-md mcp-input"
                        style={{
                          background: isLight ? 'rgba(255, 255, 255, 0.9)' : isMidnight ? 'rgba(17, 24, 39, 0.8)' : 'rgba(35, 42, 61, 0.6)',
                          border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
                          color: textColor,
                        }}
                        whileFocus={{
                          scale: 1.01,
                          boxShadow: '0 0 20px rgba(99, 102, 241, 0.3)',
                        }}
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-2" style={{ color: mutedTextColor }}>
                        Host Address
                      </label>
                      <motion.input
                        type="text"
                        placeholder="192.168.1.100 or localhost"
                        value={directHost}
                        onChange={e => setDirectHost(e.target.value)}
                        className="w-full p-3 rounded-xl backdrop-blur-md mcp-input"
                        style={{
                          background: isLight ? 'rgba(255, 255, 255, 0.9)' : isMidnight ? 'rgba(17, 24, 39, 0.8)' : 'rgba(35, 42, 61, 0.6)',
                          border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
                          color: textColor,
                        }}
                        whileFocus={{
                          scale: 1.01,
                          boxShadow: '0 0 20px rgba(99, 102, 241, 0.3)',
                        }}
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-2" style={{ color: mutedTextColor }}>
                        Port
                      </label>
                      <motion.input
                        type="text"
                        placeholder={defaultMCPPort || '8080'}
                        value={directPort}
                        onChange={e => setDirectPort(e.target.value)}
                        className="w-full p-3 rounded-xl backdrop-blur-md mcp-input"
                        style={{
                          background: isLight ? 'rgba(255, 255, 255, 0.9)' : isMidnight ? 'rgba(17, 24, 39, 0.8)' : 'rgba(35, 42, 61, 0.6)',
                          border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
                          color: textColor,
                        }}
                        whileFocus={{
                          scale: 1.01,
                          boxShadow: '0 0 20px rgba(99, 102, 241, 0.3)',
                        }}
                      />
                    </div>

                    <motion.button
                      onClick={handleDirectConnect}
                      disabled={directConnecting || connecting || loading}
                      className="w-full py-4 rounded-xl backdrop-blur-md text-sm font-bold"
                      style={{
                        background: (directConnecting || connecting || loading)
                          ? 'rgba(148, 163, 184, 0.15)'
                          : 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        color: 'white',
                        opacity: (directConnecting || connecting || loading) ? 0.7 : 1,
                      }}
                      whileHover={!(directConnecting || connecting || loading) ? { 
                        scale: 1.02,
                        boxShadow: '0 0 25px rgba(99, 102, 241, 0.4)'
                      } : {}}
                      whileTap={!(directConnecting || connecting || loading) ? { scale: 0.98 } : {}}
                    >
                      {directConnecting || (connecting && !selectedServerId) ? (
                        <div className="flex items-center justify-center space-x-2">
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                          >
                            <ArrowPathIcon className="w-5 h-5" />
                          </motion.div>
                          <span>Testing & Connecting...</span>
                        </div>
                      ) : (
                        <div className="flex items-center justify-center space-x-2">
                          <ServerIcon className="w-5 h-5" />
                          <span>Test & Connect</span>
                        </div>
                      )}
                    </motion.button>

                    {/* Info box */}
                    <motion.div 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.3 }}
                      className="p-4 rounded-xl backdrop-blur-md"
                      style={{
                        background: isLight ? 'rgba(148, 163, 184, 0.15)' : isMidnight ? 'rgba(55, 65, 81, 0.25)' : 'rgba(42, 51, 73, 0.2)',
                        border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
                        color: textColor,
                      }}
                    >
                      <div className="flex items-start space-x-3">
                        <InformationCircleIcon className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: isLight ? '#4338ca' : isMidnight ? '#c7d2fe' : '#a78bfa' }} />
                        <div className="text-sm leading-relaxed" style={{ color: mutedTextColor }}>
                          <p className="font-medium mb-1">Quick Connect</p>
                          <p>This will test the server connection and automatically connect if successful. The server configuration will be saved for future use.</p>
                        </div>
                      </div>
                    </motion.div>
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default MCPServerSelector;

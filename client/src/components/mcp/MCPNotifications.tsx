import React, { useState, useEffect } from 'react';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import { useMCP } from '../../contexts/MCPContext';
import { useWebSocket } from '../../contexts/WebSocketContext';
import { 
  XMarkIcon, 
  BellIcon, 
  CheckCircleIcon,
  ExclamationCircleIcon,
  ArrowPathIcon,
  CpuChipIcon,
  ServerIcon,
  InformationCircleIcon,
  ExclamationTriangleIcon,
  WrenchIcon
} from '@heroicons/react/24/outline';

const MCPNotifications: React.FC = () => {
  const { 
    notificationResults, 
    clearNotifications,
    isAgentEnabled,
    reconnectToServer,
    hasClientIdIssue,
    attemptClientIdRecovery,
    pendingCommands,
    commandResults,
    isProcessing,
    isAnalyzing
  } = useMCPAgent();
  
  const { 
    mcpConnection, 
    isConnected, 
    defaultServer, 
    getClientId,
    availableTools 
  } = useMCP();
  
  const { connected: wsConnected, reconnect: reconnectWs } = useWebSocket();
  
  const [isOpen, setIsOpen] = useState(false);
  const [hasUnread, setHasUnread] = useState(false);
  
  // Track unread notifications
  useEffect(() => {
    if (notificationResults.length > 0) {
      setHasUnread(true);
    }
  }, [notificationResults]);
  
  // When opening the panel, mark as read
  const handleToggleOpen = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setHasUnread(false);
    }
  };
  
  // Determine if we have a client ID
  const clientId = getClientId();
  const clientIdMissing = mcpConnection.status === 'connected' && !clientId;
  
  // Handle WebSocket reconnect
  const handleReconnectWebSocket = () => {
    reconnectWs();
  };

  // Handle client ID recovery
  const handleFixClientId = async () => {
    await attemptClientIdRecovery();
    // Refresh notifications
    setHasUnread(true);
  };

  // Add page refresh function
  const handlePageRefresh = () => {
    window.location.reload();
  };
  
  // Filter out notification types we want to show - focus on connection status only
  const visibleNotifications = notificationResults.filter(
    notification => 
      // Only keep connection and client ID notifications
      notification.result?.text?.includes("MCP Agent initialized") ||
      notification.result?.text?.includes("Connected to MCP Server") ||
      notification.result?.text?.includes("Client ID") ||
      // Keep error notifications related to connection
      (notification.error && 
       (notification.error.includes("connection") || 
        notification.error.includes("connect") || 
        notification.error.includes("client") || 
        notification.error.includes("MCP server")))
  );
  
  // Add status indicator as first notification
  const statusNotification = {
    id: 'status-indicator',
    timestamp: Date.now(),
    success: mcpConnection.status === 'connected' && !clientIdMissing && wsConnected,
    error: mcpConnection.status === 'error' ? mcpConnection.error : null,
    commandId: '',
    result: {
      text: wsConnected 
        ? (mcpConnection.status === 'connected' 
            ? `Connected to MCP Server. You can now chat with AI as usual.` 
            : mcpConnection.status === 'connecting' 
              ? 'Connecting to MCP Server...'
              : 'MCP Agent Ready')
        : 'WebSocket Connection Lost. Please reconnect.',
      isStatus: true
    },
    needsAnalysis: false,
    isAnalyzed: true
  };
  
  // Get status indicator notification with MCP Agent Ready content
  const welcomeNotification = {
    id: 'welcome-message',
    timestamp: Date.now(),
    success: true,
    commandId: '',
    error: null,
    result: {
      text: `MCP Agent Ready\n\nConnected to ${defaultServer?.mcp_nickname || ''} at ${defaultServer?.mcp_host || ''}:${defaultServer?.mcp_port || ''}. You can ask me to perform tasks using ${availableTools?.length || 0} available tools.\n\nExamples of what you can ask:\n- List files in the current directory\n- Show the content of a file\n- Create a new file with specific content\n- Run a command in the terminal\n- Help me troubleshoot an issue with my code`,
      isWelcome: true
    },
    needsAnalysis: false,
    isAnalyzed: true
  };
  
  // Add the status and welcome notifications to the beginning of the array
  const allNotifications = [
    statusNotification,
    ...(isAgentEnabled && mcpConnection.status === 'connected' && defaultServer ? [welcomeNotification] : []),
    ...visibleNotifications
  ];
  
  // Get color class based on notification type
  const getNotificationColorClass = (notification) => {
    if (notification.id === 'status-indicator') {
      if (!wsConnected) return 'bg-red-50 dark:bg-red-900/20';
      if (mcpConnection.status === 'connected') {
        return clientIdMissing ? 'bg-yellow-50 dark:bg-yellow-900/20' : 'bg-green-50 dark:bg-green-900/20';
      }
      if (mcpConnection.status === 'error') return 'bg-red-50 dark:bg-red-900/20';
      if (mcpConnection.status === 'connecting') return 'bg-blue-50 dark:bg-blue-900/20';
      return 'bg-gray-50 dark:bg-gray-900/20';
    }
    
    if (notification.id === 'welcome-message') {
      return 'bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 dark:border-blue-400';
    }
    
    if (notification.error) {
      return 'bg-red-50 dark:bg-red-900/20';
    }
    
    return '';
  };
  
  return (
    <div className="relative">
      {/* Notification icon with badge */}
      <button 
        onClick={handleToggleOpen}
        className="relative p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        title="MCP Notifications"
      >
        <BellIcon className="h-5 w-5" />
        {hasUnread && (
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
        )}
      </button>
      
      {/* Notification panel */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-96 bg-white dark:bg-gray-800 rounded-md shadow-lg z-50 border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="flex justify-between items-center p-3 border-b border-gray-200 dark:border-gray-700">
            <h3 className="font-medium">MCP Notifications</h3>
            <div className="flex space-x-2">
              <button 
                onClick={clearNotifications}
                className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                Clear all
              </button>
              <button onClick={() => setIsOpen(false)}>
                <XMarkIcon className="h-4 w-4" />
              </button>
            </div>
          </div>
          
          <div className="max-h-96 overflow-y-auto">
            {allNotifications.length === 0 ? (
              <div className="p-4 text-center text-gray-500 dark:text-gray-400 text-sm">
                No notifications
              </div>
            ) : (
              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {allNotifications.map(notification => (
                  <div 
                    key={notification.id} 
                    className={`p-3 text-sm ${getNotificationColorClass(notification)}`}
                  >
                    {/* Status indicator with action buttons */}
                    {notification.id === 'status-indicator' && (
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          {!wsConnected ? (
                            <ExclamationCircleIcon className="h-4 w-4 mr-1.5 text-red-700 dark:text-red-400" />
                          ) : mcpConnection.status === 'connected' ? (
                            clientIdMissing ? (
                              <ExclamationTriangleIcon className="h-4 w-4 mr-1.5 text-yellow-700 dark:text-yellow-400" />
                            ) : (
                              <CheckCircleIcon className="h-4 w-4 mr-1.5 text-green-700 dark:text-green-400" />
                            )
                          ) : mcpConnection.status === 'error' ? (
                            <ExclamationCircleIcon className="h-4 w-4 mr-1.5 text-red-700 dark:text-red-400" />
                          ) : mcpConnection.status === 'connecting' ? (
                            <ArrowPathIcon className="h-4 w-4 mr-1.5 text-blue-700 dark:text-blue-400 animate-spin" />
                          ) : (
                            <CpuChipIcon className="h-4 w-4 mr-1.5 text-gray-700 dark:text-gray-400" />
                          )}
                          <span>{notification.result.text}</span>
                        </div>
                        <div className="flex space-x-2">
                          {/* WebSocket reconnect button */}
                          {!wsConnected && (
                            <button 
                              onClick={handleReconnectWebSocket}
                              className="text-xs py-0.5 px-1.5 rounded bg-red-100 dark:bg-red-900/30 hover:bg-red-200 dark:hover:bg-red-800/50"
                            >
                              Reconnect
                            </button>
                          )}
                          
                          {/* Fix ClientId button */}
                          {clientIdMissing && wsConnected && (
                            <button 
                              onClick={handleFixClientId}
                              className="text-xs py-0.5 px-1.5 rounded bg-yellow-100 dark:bg-yellow-900/30 hover:bg-yellow-200 dark:hover:bg-yellow-800/50"
                            >
                              Fix ClientId
                            </button>
                          )}
                          
                          {/* Reconnect button (when error) */}
                          {mcpConnection.status === 'error' && wsConnected && (
                            <button 
                              onClick={reconnectToServer}
                              className="text-xs py-0.5 px-1.5 rounded bg-red-100 dark:bg-red-900/30 hover:bg-red-200 dark:hover:bg-red-800/50"
                            >
                              Reconnect
                            </button>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Welcome message with formatted content */}
                    {notification.id === 'welcome-message' && (
                      <div className="whitespace-pre-line">
                        {notification.result.text}
                      </div>
                    )}
                    
                    {/* Regular notifications */}
                    {notification.id !== 'status-indicator' && notification.id !== 'welcome-message' && (
                      <>
                        {notification.error ? (
                          <div className="text-red-600 dark:text-red-400">{notification.error}</div>
                        ) : (
                          <div>{notification.result?.text}</div>
                        )}
                      </>
                    )}
                    
                    {/* Timestamp for all notifications */}
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {new Date(notification.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
          
          {/* Connection status */}
          <div className="p-2 border-t border-gray-200 dark:border-gray-700 text-xs flex items-center">
            <div 
              className={`w-2 h-2 rounded-full mr-2 ${
                mcpConnection.status === 'connected' ? 'bg-green-500' : 
                mcpConnection.status === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
              }`}
            ></div>
            <span>
              {mcpConnection.status === 'connected' 
                ? `Connected to MCP (ID: ${mcpConnection.clientId?.substring(0, 8)}...)` 
                : mcpConnection.status === 'connecting'
                ? 'Connecting to MCP...'
                : 'Disconnected from MCP'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MCPNotifications; 
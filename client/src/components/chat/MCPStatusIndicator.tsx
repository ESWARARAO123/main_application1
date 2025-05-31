import React from 'react';
import {
  CheckCircleIcon,
  ExclamationCircleIcon,
  ArrowPathIcon,
  CpuChipIcon,
  ServerIcon,
  InformationCircleIcon,
  ExclamationTriangleIcon,
  WrenchIcon
} from '@heroicons/react/24/outline';
import { useMCP } from '../../contexts/MCPContext';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import { useWebSocket } from '../../contexts/WebSocketContext';

interface MCPStatusIndicatorProps {
  className?: string;
}

const MCPStatusIndicator: React.FC<MCPStatusIndicatorProps> = ({ className = '' }) => {
  const { isConnected, defaultServer, mcpConnection, reconnect, getClientId } = useMCP();
  const { isAgentEnabled, reconnectToServer, hasClientIdIssue, attemptClientIdRecovery } = useMCPAgent();
  const { connected: wsConnected, isFullyReady, reconnect: reconnectWs } = useWebSocket();

  if (!isAgentEnabled) {
    return null;
  }

  // Determine if we have a client ID
  const clientId = getClientId();
  const clientIdMissing = mcpConnection.status === 'connected' && !clientId;
  const webSocketDown = !wsConnected;

  // Handle WebSocket reconnect
  const handleReconnectWebSocket = () => {
    reconnectWs();
  };

  // Handle client ID recovery
  const handleFixClientId = async () => {
    const recovered = await attemptClientIdRecovery();
    if (recovered) {
      // Success feedback
      const statusElement = document.querySelector('.mcp-status-indicator') as HTMLElement;
      if (statusElement) {
        const originalBg = statusElement.style.backgroundColor;
        statusElement.style.backgroundColor = 'var(--color-success-bg)';
        setTimeout(() => {
          if (statusElement) {
            statusElement.style.backgroundColor = originalBg || '';
          }
        }, 1500);
      }
    }
  };

  // Add page refresh function
  const handlePageRefresh = () => {
    window.location.reload();
  };

  // Determine status styles based on connection state
  const getStatusStyles = () => {
    // First check WebSocket connection status as it's most fundamental
    if (!wsConnected) {
      return {
        bgColor: 'bg-red-50 dark:bg-red-900/20',
        textColor: 'text-red-700 dark:text-red-400',
        borderColor: 'border-red-200 dark:border-red-800',
        icon: <ExclamationCircleIcon className="h-4 w-4 mr-1.5 text-red-700 dark:text-red-400" />
      };
    }
    
    // Then check MCP connection status
    if (mcpConnection.status === 'connected') {
      if (clientIdMissing) {
        // Connected but missing clientId - warning state
        return {
          bgColor: 'bg-yellow-50 dark:bg-yellow-900/20',
          textColor: 'text-yellow-700 dark:text-yellow-400',
          borderColor: 'border-yellow-200 dark:border-yellow-800',
          icon: <ExclamationTriangleIcon className="h-4 w-4 mr-1.5 text-yellow-700 dark:text-yellow-400" />
        };
      }
      return {
        bgColor: 'bg-green-50 dark:bg-green-900/20',
        textColor: 'text-green-700 dark:text-green-400',
        borderColor: 'border-green-200 dark:border-green-800',
        icon: <CheckCircleIcon className="h-4 w-4 mr-1.5 text-green-700 dark:text-green-400" />
      };
    } else if (mcpConnection.status === 'error') {
      return {
        bgColor: 'bg-red-50 dark:bg-red-900/20',
        textColor: 'text-red-700 dark:text-red-400',
        borderColor: 'border-red-200 dark:border-red-800',
        icon: <ExclamationCircleIcon className="h-4 w-4 mr-1.5 text-red-700 dark:text-red-400" />
      };
    } else if (mcpConnection.status === 'connecting') {
      return {
        bgColor: 'bg-blue-50 dark:bg-blue-900/20',
        textColor: 'text-blue-700 dark:text-blue-400',
        borderColor: 'border-blue-200 dark:border-blue-800',
        icon: <ArrowPathIcon className="h-4 w-4 mr-1.5 text-blue-700 dark:text-blue-400 animate-spin" />
      };
    } else {
      return {
        bgColor: 'bg-gray-50 dark:bg-gray-900/20',
        textColor: 'text-gray-700 dark:text-gray-400',
        borderColor: 'border-gray-200 dark:border-gray-800',
        icon: <CpuChipIcon className="h-4 w-4 mr-1.5 text-gray-700 dark:text-gray-400" />
      };
    }
  };

  const styles = getStatusStyles();

  // Render function for the component
  return (
    <div className={`mcp-status-indicator flex flex-col mb-4 ${className}`}>
      <div 
        className={`flex items-center py-1.5 px-3 rounded-md ${styles.bgColor} ${styles.textColor} border ${styles.borderColor}`}
      >
        {styles.icon}
        <span className="text-sm font-medium">
          {!wsConnected
            ? 'WebSocket Connection Lost'
            : mcpConnection.status === 'connected'
              ? clientIdMissing 
                ? 'Connected but Missing ClientId' 
                : 'Connected to MCP Server'
              : mcpConnection.status === 'error'
                ? 'MCP Connection Error'
                : mcpConnection.status === 'connecting'
                  ? 'Connecting to MCP Server...'
                  : 'MCP Agent Ready'}
        </span>
        
        {/* Server info (when connected) */}
        {isConnected && defaultServer && (
          <div className="ml-2 flex items-center text-xs opacity-80 pl-2 border-l border-current">
            <ServerIcon className="h-3 w-3 mr-1" />
            {defaultServer.mcp_nickname}
          </div>
        )}
        
        {/* Client ID info when available */}
        {clientId && (
          <div className="ml-2 flex items-center text-xs opacity-80 pl-2 border-l border-current">
            <span className="font-mono truncate max-w-[60px]" title={clientId}>
              {clientId.substring(0, 8)}...
            </span>
          </div>
        )}
        
        {/* Spacer */}
        <div className="flex-grow"></div>
        
        {/* WebSocket reconnect button */}
        {!wsConnected && (
          <button 
            onClick={handleReconnectWebSocket}
            className="ml-2 flex items-center text-xs py-0.5 px-1.5 rounded hover:bg-white hover:bg-opacity-20"
            title="Reconnect WebSocket"
          >
            <ArrowPathIcon className="h-3 w-3 mr-1" />
            Reconnect WebSocket
          </button>
        )}
        
        {/* Fix ClientId button */}
        {clientIdMissing && wsConnected && (
          <button 
            onClick={handleFixClientId}
            className="ml-2 flex items-center text-xs py-0.5 px-1.5 rounded hover:bg-white hover:bg-opacity-20"
            title="Attempt to recover the client ID"
          >
            <WrenchIcon className="h-3 w-3 mr-1" />
            Fix ClientId
          </button>
        )}
        
        {/* Reconnect button (when error) */}
        {mcpConnection.status === 'error' && wsConnected && (
          <button 
            onClick={reconnectToServer}
            className="ml-2 flex items-center text-xs py-0.5 px-1.5 rounded hover:bg-white hover:bg-opacity-20"
          >
            <ArrowPathIcon className="h-3 w-3 mr-1" />
            Reconnect
          </button>
        )}
      </div>
      
      {/* WebSocket warning */}
      {!wsConnected && (
        <div className="mt-1 text-xs flex items-start text-red-600 dark:text-red-400 font-medium justify-between">
          <div className="flex items-start">
            <ExclamationCircleIcon className="h-3.5 w-3.5 mr-1 mt-0.5 flex-shrink-0" />
            <span>
              WebSocket connection is down. Please click "Reconnect WebSocket" or refresh the page.
            </span>
          </div>
          <button 
            onClick={handlePageRefresh}
            className="ml-2 flex items-center text-xs px-2 py-0.5 rounded bg-red-100 dark:bg-red-900/30 hover:bg-red-200 dark:hover:bg-red-800/50"
          >
            Refresh Page
          </button>
        </div>
      )}
      
      {/* Error message or help text */}
      {mcpConnection.status === 'error' && mcpConnection.error && (
        <div className="mt-1 text-xs flex items-start text-red-600 dark:text-red-400 justify-between">
          <div className="flex items-start">
            <InformationCircleIcon className="h-3.5 w-3.5 mr-1 mt-0.5 flex-shrink-0" />
            <span>{mcpConnection.error}</span>
          </div>
          <button 
            onClick={handlePageRefresh}
            className="ml-2 flex items-center text-xs px-2 py-0.5 rounded bg-red-100 dark:bg-red-900/30 hover:bg-red-200 dark:hover:bg-red-800/50"
          >
            Refresh Page
          </button>
        </div>
      )}
      
      {/* Warning for client ID issue */}
      {clientIdMissing && (
        <div className="mt-1 text-xs flex items-start text-yellow-600 dark:text-yellow-400">
          <ExclamationTriangleIcon className="h-3.5 w-3.5 mr-1 mt-0.5 flex-shrink-0" />
          <span>
            Missing client ID. Tool invocation will not work. Click "Fix ClientId" to attempt recovery.
          </span>
        </div>
      )}
      
      {/* Status when everything is good */}
      {mcpConnection.status === 'connected' && !clientIdMissing && wsConnected && (
        <div className="mt-1 text-xs flex items-start text-gray-600 dark:text-gray-400">
          <InformationCircleIcon className="h-3.5 w-3.5 mr-1 mt-0.5 flex-shrink-0" />
          <span>Start typing to interact with the MCP Agent</span>
        </div>
      )}
    </div>
  );
};

export default MCPStatusIndicator; 
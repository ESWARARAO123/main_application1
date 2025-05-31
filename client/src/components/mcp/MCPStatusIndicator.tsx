import React from 'react';
import { useMCP } from '../../contexts/MCPContext';
import { useTheme } from '../../contexts/ThemeContext';
import { useMCPAgent } from '../../contexts/MCPAgentContext';

interface MCPStatusIndicatorProps {
  hideWhenDisconnected?: boolean;
  showServerInfo?: boolean;
  minimal?: boolean;
}

const MCPStatusIndicator: React.FC<MCPStatusIndicatorProps> = ({
  hideWhenDisconnected = false,
  showServerInfo = false,
  minimal = false
}) => {
  const {
    isConnected,
    mcpConnection,
    defaultServer,
    reconnect,
    error
  } = useMCP();
  
  // Add MCP Agent context to check for client ID issues
  const { hasClientIdIssue, attemptClientIdRecovery } = useMCPAgent();
  
  const { currentTheme } = useTheme();

  // Different statuses
  let backgroundColor = '';
  let textColor = '';
  let borderColor = '';
  let statusText = '';
  let statusIcon = null;

  // If not connected and should hide, return null
  if (!isConnected && hideWhenDisconnected) {
    return null;
  }

  // Handle page refresh 
  const handleRefreshPage = () => {
    window.location.reload();
  };

  // Determine status appearance
  if (isConnected && !hasClientIdIssue) {
    // Connected
    backgroundColor = currentTheme === 'dark' ? 'rgba(0, 150, 0, 0.2)' : 'rgba(0, 160, 0, 0.1)';
    textColor = currentTheme === 'dark' ? '#4ADE80' : '#16A34A'; // text-green-400 dark or text-green-600 light
    borderColor = currentTheme === 'dark' ? '#4ADE80' : '#16A34A';
    statusText = 'Connected';
    statusIcon = (
      <svg className="w-3 h-3 mr-1 fill-current" viewBox="0 0 24 24">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm4.59-12.42L10 14.17l-2.59-2.58L6 13l4 4 8-8z" />
      </svg>
    );
  } else if (isConnected && hasClientIdIssue) {
    // Connected but missing client ID
    backgroundColor = currentTheme === 'dark' ? 'rgba(150, 100, 0, 0.2)' : 'rgba(160, 100, 0, 0.1)';
    textColor = currentTheme === 'dark' ? '#FB923C' : '#EA580C'; // text-orange-400 dark or text-orange-600 light
    borderColor = currentTheme === 'dark' ? '#FB923C' : '#EA580C';
    statusText = 'Client ID Issue';
    statusIcon = (
      <svg className="w-3 h-3 mr-1 fill-current" viewBox="0 0 24 24">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z" />
      </svg>
    );
  } else if (mcpConnection.status === 'connecting') {
    // Connecting
    backgroundColor = currentTheme === 'dark' ? 'rgba(150, 150, 0, 0.2)' : 'rgba(160, 160, 0, 0.1)';
    textColor = currentTheme === 'dark' ? '#FACC15' : '#CA8A04'; // text-yellow-400 dark or text-yellow-600 light
    borderColor = currentTheme === 'dark' ? '#FACC15' : '#CA8A04';
    statusText = 'Connecting...';
    statusIcon = (
      <svg className="w-3 h-3 mr-1 fill-current animate-spin" viewBox="0 0 24 24">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z" />
        <path d="M12 4V2" stroke={textColor} strokeWidth="2" />
      </svg>
    );
  } else if (mcpConnection.status === 'error') {
    // Error
    backgroundColor = currentTheme === 'dark' ? 'rgba(150, 0, 0, 0.2)' : 'rgba(160, 0, 0, 0.1)';
    textColor = currentTheme === 'dark' ? '#F87171' : '#DC2626'; // text-red-400 dark or text-red-600 light
    borderColor = currentTheme === 'dark' ? '#F87171' : '#DC2626';
    statusText = 'Error';
    statusIcon = (
      <svg className="w-3 h-3 mr-1 fill-current" viewBox="0 0 24 24">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z" />
      </svg>
    );
  } else {
    // Disconnected
    backgroundColor = currentTheme === 'dark' ? 'rgba(100, 100, 100, 0.2)' : 'rgba(100, 100, 100, 0.1)';
    textColor = currentTheme === 'dark' ? '#9CA3AF' : '#6B7280'; // text-gray-400 dark or text-gray-500 light
    borderColor = currentTheme === 'dark' ? '#9CA3AF' : '#6B7280';
    statusText = 'Disconnected';
    statusIcon = (
      <svg className="w-3 h-3 mr-1 fill-current" viewBox="0 0 24 24">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm4-2.5l-1.41 1.41L12 16.34l-2.59 2.58L8 17.5l2.59-2.59L8 12.32l1.41-1.41L12 13.49l2.59-2.58L16 12.32l-2.59 2.59L16 17.5z" />
      </svg>
    );
  }

  // For minimal mode, just return a simple indicator
  if (minimal) {
    return (
      <div
        className="inline-flex items-center gap-1 px-1 py-0.5 rounded text-xs"
        style={{ backgroundColor, color: textColor, borderColor }}
      >
        <div className={`h-2 w-2 rounded-full ${isConnected && !hasClientIdIssue ? 'bg-green-500' : hasClientIdIssue ? 'bg-orange-500' : mcpConnection.status === 'connecting' ? 'bg-yellow-500' : mcpConnection.status === 'error' ? 'bg-red-500' : 'bg-gray-500'}`} 
          style={{ animation: mcpConnection.status === 'connecting' ? 'pulse 2s infinite' : 'none' }} />
      </div>
    );
  }

  // Normal full status indicator
  return (
    <div
      className="flex items-center py-1 px-2 rounded text-xs"
      style={{
        backgroundColor,
        color: textColor,
        border: `1px solid ${borderColor}`
      }}
    >
      <div className="flex items-center">
        {statusIcon}
        <span className="font-medium">{statusText}</span>
      </div>

      {isConnected && showServerInfo && defaultServer && (
        <span className="ml-1 opacity-75">
          ({defaultServer.mcp_nickname || `${defaultServer.mcp_host}:${defaultServer.mcp_port}`})
        </span>
      )}

      {mcpConnection.status === 'error' && (
        <div className="flex items-center ml-2 gap-1">
          <button
            className="text-xs px-1 rounded"
            style={{
              backgroundColor: currentTheme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
            }}
            onClick={reconnect}
            title={error || "Reconnect to MCP server"}
          >
            Reconnect
          </button>
          
          <button
            className="text-xs px-1 rounded"
            style={{
              backgroundColor: currentTheme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
            }}
            onClick={handleRefreshPage}
            title="Refresh the page to restart all connections"
          >
            Refresh Page
          </button>
        </div>
      )}
      
      {isConnected && hasClientIdIssue && (
        <div className="flex items-center ml-2 gap-1">
          <button
            className="text-xs px-1 rounded"
            style={{
              backgroundColor: currentTheme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
            }}
            onClick={() => attemptClientIdRecovery()}
            title="Attempt to recover the Client ID"
          >
            Recover ID
          </button>
          
          <button
            className="text-xs px-1 rounded"
            style={{
              backgroundColor: currentTheme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
            }}
            onClick={handleRefreshPage}
            title="Refresh the page to restart all connections"
          >
            Refresh Page
          </button>
        </div>
      )}
    </div>
  );
};

export default MCPStatusIndicator; 
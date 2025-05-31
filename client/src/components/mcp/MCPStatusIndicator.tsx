import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
  const [showTooltip, setShowTooltip] = useState(false);

  // Inject sophisticated animations
  useEffect(() => {
    const style = document.createElement('style');
    style.innerHTML = `
      @keyframes pulseGlow {
        0%, 100% { 
          box-shadow: 0 0 8px rgba(99, 102, 241, 0.4);
          transform: scale(1);
        }
        50% { 
          box-shadow: 0 0 20px rgba(99, 102, 241, 0.8);
          transform: scale(1.05);
        }
      }

      @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
      }

      .mcp-pulse-animation {
        animation: pulseGlow 2s infinite;
      }

      .mcp-shimmer {
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.3) 50%, transparent 70%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
      }
    `;
    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, []);

  // If not connected and should hide, return null
  if (!isConnected && hideWhenDisconnected) {
    return null;
  }

  // Handle page refresh 
  const handleRefreshPage = () => {
    window.location.reload();
  };

  // Get status configuration
  const getStatusConfig = () => {
    if (isConnected && !hasClientIdIssue) {
      return {
        status: 'connected',
        label: 'Connected',
        color: '#10B981', // emerald-500
        bgColor: 'rgba(16, 185, 129, 0.1)',
        borderColor: '#10B981',
        icon: (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', duration: 0.5 }}
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          </motion.div>
        )
      };
    } else if (isConnected && hasClientIdIssue) {
      return {
        status: 'warning',
        label: 'Client ID Issue',
        color: '#F59E0B', // amber-500
        bgColor: 'rgba(245, 158, 11, 0.1)',
        borderColor: '#F59E0B',
        icon: (
          <motion.div
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </motion.div>
        )
      };
    } else if (mcpConnection.status === 'connecting') {
      return {
        status: 'connecting',
        label: 'Connecting...',
        color: '#3B82F6', // blue-500
        bgColor: 'rgba(59, 130, 246, 0.1)',
        borderColor: '#3B82F6',
        icon: (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </motion.div>
        )
      };
    } else if (mcpConnection.status === 'error') {
      return {
        status: 'error',
        label: 'Error',
        color: '#EF4444', // red-500
        bgColor: 'rgba(239, 68, 68, 0.1)',
        borderColor: '#EF4444',
        icon: (
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </motion.div>
        )
      };
    } else {
      return {
        status: 'disconnected',
        label: 'Disconnected',
        color: '#6B7280', // gray-500
        bgColor: 'rgba(107, 114, 128, 0.1)',
        borderColor: '#6B7280',
        icon: (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M13.477 14.89A6 6 0 015.11 6.524l8.367 8.368zm1.414-1.414L6.524 5.11a6 6 0 018.367 8.367zM18 10a8 8 0 11-16 0 8 8 0 0116 0z" clipRule="evenodd" />
          </svg>
        )
      };
    }
  };

  const statusConfig = getStatusConfig();

  // For minimal mode, return elegant floating dot
  if (minimal) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0 }}
        animate={{ opacity: 1, scale: 1 }}
        className="relative"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        <motion.div
          className={`w-3 h-3 rounded-full relative overflow-hidden ${statusConfig.status === 'connecting' ? 'mcp-pulse-animation' : ''}`}
          style={{
            backgroundColor: statusConfig.color,
            boxShadow: `0 0 10px ${statusConfig.color}40`
          }}
          whileHover={{ scale: 1.2 }}
        >
          {statusConfig.status === 'connecting' && (
            <div className="absolute inset-0 mcp-shimmer rounded-full" />
          )}
        </motion.div>

        <AnimatePresence>
          {showTooltip && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.8 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.8 }}
              className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 z-50"
            >
              <div
                className="px-3 py-2 rounded-lg text-xs font-medium whitespace-nowrap"
                style={{
                  background: 'rgba(255, 255, 255, 0.9)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
                  color: '#1f2937'
                }}
              >
                MCP {statusConfig.label}
                <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-white/90" />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    );
  }

  // Full status indicator with glass morphism
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ type: 'spring', duration: 0.6 }}
      className="relative"
    >
      <motion.div
        className="flex items-center px-4 py-2 rounded-xl relative overflow-hidden backdrop-blur-md"
        style={{
          background: `linear-gradient(135deg, ${statusConfig.bgColor}, rgba(255, 255, 255, 0.05))`,
          border: `1px solid ${statusConfig.borderColor}40`,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
        }}
        whileHover={{ 
          scale: 1.02,
          boxShadow: `0 12px 40px rgba(0, 0, 0, 0.15), 0 0 20px ${statusConfig.color}30`
        }}
        whileTap={{ scale: 0.98 }}
      >
        {/* Animated background gradient */}
        <motion.div
          className="absolute inset-0 opacity-30"
          style={{
            background: `linear-gradient(45deg, ${statusConfig.color}20, transparent, ${statusConfig.color}20)`,
            backgroundSize: '200% 200%',
          }}
          animate={{
            backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'linear',
          }}
        />

        <div className="flex items-center space-x-3 relative z-10">
          {/* Status icon with glow effect */}
          <motion.div
            className="flex items-center justify-center w-8 h-8 rounded-full"
            style={{
              backgroundColor: `${statusConfig.color}20`,
              color: statusConfig.color,
              boxShadow: `0 0 15px ${statusConfig.color}40`,
            }}
            whileHover={{ rotate: 360 }}
            transition={{ duration: 0.5 }}
          >
            {statusConfig.icon}
          </motion.div>

          {/* Status text */}
          <div className="flex flex-col">
            <motion.span
              className="text-sm font-semibold"
              style={{ color: statusConfig.color }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              {statusConfig.label}
            </motion.span>

            {isConnected && showServerInfo && defaultServer && (
              <motion.span
                className="text-xs opacity-70"
                style={{ color: statusConfig.color }}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 0.7, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                {defaultServer.mcp_nickname || `${defaultServer.mcp_host}:${defaultServer.mcp_port}`}
              </motion.span>
            )}
          </div>

          {/* Action buttons for error states */}
          <AnimatePresence>
            {(mcpConnection.status === 'error' || hasClientIdIssue) && (
              <motion.div
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                className="flex space-x-2"
              >
                {hasClientIdIssue && (
                  <motion.button
                    className="px-3 py-1 rounded-lg text-xs font-medium backdrop-blur-md"
                    style={{
                      background: 'rgba(255, 255, 255, 0.2)',
                      border: '1px solid rgba(255, 255, 255, 0.3)',
                      color: statusConfig.color,
                    }}
                    whileHover={{ 
                      scale: 1.05,
                      background: 'rgba(255, 255, 255, 0.3)'
                    }}
                    whileTap={{ scale: 0.95 }}
                    onClick={attemptClientIdRecovery}
                  >
                    Fix ID
                  </motion.button>
                )}

                {mcpConnection.status === 'error' && (
                  <motion.button
                    className="px-3 py-1 rounded-lg text-xs font-medium backdrop-blur-md"
                    style={{
                      background: 'rgba(255, 255, 255, 0.2)',
                      border: '1px solid rgba(255, 255, 255, 0.3)',
                      color: statusConfig.color,
                    }}
                    whileHover={{ 
                      scale: 1.05,
                      background: 'rgba(255, 255, 255, 0.3)'
                    }}
                    whileTap={{ scale: 0.95 }}
                    onClick={reconnect}
                  >
                    Reconnect
                  </motion.button>
                )}

                <motion.button
                  className="px-3 py-1 rounded-lg text-xs font-medium backdrop-blur-md"
                  style={{
                    background: 'rgba(255, 255, 255, 0.2)',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    color: statusConfig.color,
                  }}
                  whileHover={{ 
                    scale: 1.05,
                    background: 'rgba(255, 255, 255, 0.3)'
                  }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleRefreshPage}
                >
                  Refresh
                </motion.button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Shine effect on hover */}
        <motion.div
          className="absolute inset-0 opacity-0 pointer-events-none"
          style={{
            background: 'linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%)',
            backgroundSize: '200% 100%',
          }}
          whileHover={{
            opacity: 1,
            backgroundPosition: ['200% 0', '-200% 0'],
            transition: { duration: 0.6 }
          }}
        />
      </motion.div>
    </motion.div>
  );
};

export default MCPStatusIndicator; 
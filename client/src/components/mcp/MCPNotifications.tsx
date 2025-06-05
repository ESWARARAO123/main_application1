import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import { useMCP } from '../../contexts/MCPContext';
import { useWebSocket } from '../../contexts/WebSocketContext';
import { useTheme } from '../../contexts/ThemeContext';
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
  WrenchIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';

const MCPNotifications: React.FC = () => {
  const { currentTheme } = useTheme();
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
  
  // Get theme colors
  const isDarkMode = currentTheme === 'dark';
  const isMidnight = currentTheme === 'midnight';
  const isLight = currentTheme === 'light';
  
  // Theme-aware colors using CSS variables for consistency
  const textColor = isLight ? '#1e293b' : isMidnight ? '#f9fafb' : '#f3f4f6';
  const mutedTextColor = isLight ? 'rgba(71, 85, 105, 0.8)' : isMidnight ? 'rgba(249, 250, 251, 0.7)' : 'rgba(156, 163, 175, 0.8)';
  const iconColor = isLight ? 'text-gray-600' : isMidnight ? 'text-gray-300' : 'text-gray-400';
  
  // Background and UI colors
  const panelBackground = isLight
    ? 'linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.9))'
    : isMidnight
      ? 'linear-gradient(135deg, rgba(17, 24, 39, 0.9), rgba(15, 23, 42, 0.8))'
      : 'linear-gradient(135deg, rgba(35, 42, 61, 0.9), rgba(21, 28, 44, 0.8))';
      
  const panelBorder = isLight
    ? '1px solid rgba(148, 163, 184, 0.3)'
    : isMidnight
      ? '1px solid rgba(55, 65, 81, 0.4)'
      : '1px solid rgba(42, 51, 73, 0.4)';
      
  const headerBackground = isLight
    ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.06), rgba(139, 92, 246, 0.06))'
    : isMidnight
      ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(236, 72, 153, 0.1))'
      : 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1))';
      
  const footerBackground = isLight
    ? 'linear-gradient(135deg, rgba(248, 250, 252, 0.9), rgba(241, 245, 249, 0.8))'
    : isMidnight
      ? 'linear-gradient(135deg, rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.2))'
      : 'linear-gradient(135deg, rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.1))';
  
  // Inject sophisticated animations and styles
  useEffect(() => {
    const style = document.createElement('style');
    style.innerHTML = `
      @keyframes bellRing {
        0%, 100% { transform: rotate(0deg); }
        25% { transform: rotate(10deg); }
        75% { transform: rotate(-10deg); }
      }

      @keyframes notificationPulse {
        0%, 100% { 
          box-shadow: 0 0 8px rgba(99, 102, 241, 0.3);
          transform: scale(1);
        }
        50% { 
          box-shadow: 0 0 25px rgba(99, 102, 241, 0.6);
          transform: scale(1.02);
        }
      }

      @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
      }

      .bell-ring { animation: bellRing 0.5s ease-in-out; }
      .notification-pulse { animation: notificationPulse 2s infinite; }
      .shimmer-bg {
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        background-size: 200% 100%;
        animation: shimmer 3s infinite;
      }

      .notification-panel-backdrop {
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
      }

      .notification-scrollbar::-webkit-scrollbar {
        width: 6px;
      }
      
      .notification-scrollbar::-webkit-scrollbar-track {
        background: ${isLight ? 'rgba(148, 163, 184, 0.2)' : isMidnight ? 'rgba(55, 65, 81, 0.3)' : 'rgba(42, 51, 73, 0.3)'};
        border-radius: 3px;
      }
      
      .notification-scrollbar::-webkit-scrollbar-thumb {
        background: ${isLight ? 'rgba(148, 163, 184, 0.5)' : isMidnight ? 'rgba(55, 65, 81, 0.6)' : 'rgba(42, 51, 73, 0.6)'};
        border-radius: 3px;
        transition: all 0.2s ease;
      }
      
      .notification-scrollbar::-webkit-scrollbar-thumb:hover {
        background: ${isLight ? 'rgba(148, 163, 184, 0.7)' : isMidnight ? 'rgba(55, 65, 81, 0.8)' : 'rgba(42, 51, 73, 0.8)'};
      }
    `;
    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, []);
  
  // Track unread notifications with animation trigger
  useEffect(() => {
    if (notificationResults.length > 0) {
      setHasUnread(true);
      // Trigger bell animation
      const bellElement = document.querySelector('.notification-bell');
      if (bellElement) {
        bellElement.classList.add('bell-ring');
        setTimeout(() => bellElement.classList.remove('bell-ring'), 500);
      }
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
    setHasUnread(true);
  };

  // Add page refresh function
  const handlePageRefresh = () => {
    window.location.reload();
  };
  
  // Filter notifications
  const visibleNotifications = notificationResults.filter(
    notification => 
      notification.result?.text?.includes("MCP Agent initialized") ||
      notification.result?.text?.includes("Connected to MCP Server") ||
      notification.result?.text?.includes("Client ID") ||
      (notification.error && 
       (notification.error.includes("connection") || 
        notification.error.includes("connect") || 
        notification.error.includes("client") || 
        notification.error.includes("MCP server")))
  );
  
  // Enhanced status notification
  const statusNotification = {
    id: 'status-indicator',
    timestamp: Date.now(),
    success: mcpConnection.status === 'connected' && !clientIdMissing && wsConnected,
    error: mcpConnection.status === 'error' ? mcpConnection.error : null,
    commandId: '',
    result: {
      text: wsConnected 
        ? (mcpConnection.status === 'connected' 
            ? `🚀 MCP Agent Ready - Connected to ${defaultServer?.mcp_nickname || 'Server'}. Ready to execute tasks with ${availableTools?.length || 0} tools available.` 
            : mcpConnection.status === 'connecting' 
              ? '⏳ Initializing MCP Connection...'
              : '✨ MCP Agent Standby')
        : '🔌 WebSocket Connection Lost - Please reconnect to restore functionality.',
      isStatus: true
    },
    needsAnalysis: false,
    isAnalyzed: true
  };
  
  // Welcome notification with enhanced content
  const welcomeNotification = {
    id: 'welcome-message',
    timestamp: Date.now(),
    success: true,
    commandId: '',
    error: null,
    result: {
      text: `🎯 MCP Agent Capabilities

🔗 Connected to: ${defaultServer?.mcp_nickname || 'MCP Server'} 
🏠 Host: ${defaultServer?.mcp_host || 'localhost'}:${defaultServer?.mcp_port || '8080'}
🛠️ Available Tools: ${availableTools?.length || 0}

📝 What you can do:
• List and manage files in directories
• Read and analyze file contents  
• Create, edit, and organize files
• Execute terminal commands safely
• Debug and troubleshoot code issues
• Get real-time system information

💡 Try asking me to:
"Show me files in the current directory"
"Read the contents of package.json"
"Create a new README file"
"Help debug this error message"`,
      isWelcome: true
    },
    needsAnalysis: false,
    isAnalyzed: true
  };
  
  // Combine all notifications
  const allNotifications = [
    statusNotification,
    ...(isAgentEnabled && mcpConnection.status === 'connected' && defaultServer ? [welcomeNotification] : []),
    ...visibleNotifications
  ];
  
  // Get notification styling
  const getNotificationStyle = (notification) => {
    if (notification.id === 'status-indicator') {
      if (!wsConnected) return {
        bg: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05))',
        border: '1px solid rgba(239, 68, 68, 0.3)',
        icon: <ExclamationCircleIcon className="w-5 h-5 text-red-400" />,
        color: '#EF4444'
      };
      if (mcpConnection.status === 'connected') {
        return clientIdMissing ? {
          bg: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05))',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          icon: <ExclamationTriangleIcon className="w-5 h-5 text-amber-400" />,
          color: '#F59E0B'
        } : {
          bg: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05))',
          border: '1px solid rgba(16, 185, 129, 0.3)',
          icon: <CheckCircleIcon className="w-5 h-5 text-emerald-400" />,
          color: '#10B981'
        };
      }
      if (mcpConnection.status === 'error') return {
        bg: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05))',
        border: '1px solid rgba(239, 68, 68, 0.3)',
        icon: <ExclamationCircleIcon className="w-5 h-5 text-red-400" />,
        color: '#EF4444'
      };
      if (mcpConnection.status === 'connecting') return {
        bg: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05))',
        border: '1px solid rgba(59, 130, 246, 0.3)',
        icon: <ArrowPathIcon className="w-5 h-5 text-blue-400 animate-spin" />,
        color: '#3B82F6'
      };
      return {
        bg: 'linear-gradient(135deg, rgba(107, 114, 128, 0.1), rgba(107, 114, 128, 0.05))',
        border: '1px solid rgba(107, 114, 128, 0.3)',
        icon: <CpuChipIcon className="w-5 h-5 text-gray-400" />,
        color: '#6B7280'
      };
    }
    
    if (notification.id === 'welcome-message') {
      return {
        bg: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1))',
        border: '1px solid rgba(99, 102, 241, 0.3)',
        icon: <SparklesIcon className="w-5 h-5 text-indigo-400" />,
        color: '#6366F1'
      };
    }
    
    if (notification.error) {
      return {
        bg: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05))',
        border: '1px solid rgba(239, 68, 68, 0.3)',
        icon: <ExclamationCircleIcon className="w-5 h-5 text-red-400" />,
        color: '#EF4444'
      };
    }
    
    return {
      bg: 'linear-gradient(135deg, rgba(107, 114, 128, 0.1), rgba(107, 114, 128, 0.05))',
      border: '1px solid rgba(107, 114, 128, 0.3)',
      icon: <InformationCircleIcon className="w-5 h-5 text-gray-400" />,
      color: '#6B7280'
    };
  };
  
  return (
    <div className="relative">
      {/* Notification Bell Button */}
      <motion.button 
        onClick={handleToggleOpen}
        className="notification-bell relative p-3 rounded-full backdrop-blur-md transition-all duration-300"
        style={{
          background: isLight ? 'rgba(148, 163, 184, 0.15)' : isMidnight ? 'rgba(55, 65, 81, 0.25)' : 'rgba(42, 51, 73, 0.2)',
          border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
        }}
        whileHover={{ 
          scale: 1.05,
          background: isLight ? 'rgba(148, 163, 184, 0.25)' : isMidnight ? 'rgba(55, 65, 81, 0.35)' : 'rgba(42, 51, 73, 0.3)',
          boxShadow: '0 12px 40px rgba(0, 0, 0, 0.15)',
        }}
        whileTap={{ scale: 0.95 }}
        title="MCP Notifications"
      >
        <motion.div
          animate={hasUnread ? { scale: [1, 1.2, 1] } : { scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          <BellIcon className={`h-5 w-5 ${iconColor}`} />
        </motion.div>

        {/* Unread Badge */}
        <AnimatePresence>
        {hasUnread && (
            <motion.div
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              className="absolute -top-1 -right-1 w-4 h-4 rounded-full flex items-center justify-center"
              style={{
                background: 'linear-gradient(135deg, #e11d48, #be185d)',
                boxShadow: '0 0 20px rgba(225, 29, 72, 0.5)',
              }}
            >
              <motion.span 
                className="text-xs font-bold"
                style={{ color: 'white' }}
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              >
                {allNotifications.length > 9 ? '9+' : allNotifications.length}
              </motion.span>
            </motion.div>
        )}
        </AnimatePresence>
      </motion.button>
      
      {/* Notification Panel */}
      <AnimatePresence>
      {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-40"
              onClick={() => setIsOpen(false)}
            />

            {/* Panel */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: -20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: -20 }}
              transition={{ type: 'spring', duration: 0.5 }}
              className="absolute right-0 mt-2 w-80 sm:w-96 max-w-[90vw] z-50"
            >
              <div
                className="rounded-2xl shadow-2xl overflow-hidden notification-panel-backdrop"
                style={{
                  background: panelBackground,
                  border: panelBorder,
                  boxShadow: isLight 
                    ? '0 25px 50px rgba(0, 0, 0, 0.08)'
                    : isMidnight
                      ? '0 25px 50px rgba(0, 0, 0, 0.4)'
                      : '0 25px 50px rgba(0, 0, 0, 0.2)',
                }}
              >
                {/* Header */}
                <div
                  className="p-4 border-b relative overflow-hidden"
                  style={{
                    background: headerBackground,
                    borderBottom: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.2)' : isMidnight ? 'rgba(55, 65, 81, 0.3)' : 'rgba(42, 51, 73, 0.3)'}`,
                  }}
                >
                  {/* Header background animation */}
                  <div className="absolute inset-0 shimmer-bg opacity-30" />
                  
                  <div className="flex justify-between items-center relative z-10">
                    <div className="flex items-center space-x-3">
                      <motion.div
                        className="w-8 h-8 rounded-full flex items-center justify-center"
                        style={{
                          background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                          boxShadow: '0 0 20px rgba(99, 102, 241, 0.4)',
                        }}
                        animate={{ rotate: 360 }}
                        transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                      >
                        <CpuChipIcon className="w-4 h-4 text-white" />
                      </motion.div>
                      <h3 className="text-lg font-bold" style={{ color: textColor }}>MCP Control Center</h3>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <motion.button 
                onClick={clearNotifications}
                        className="px-3 py-1 rounded-lg text-xs font-medium backdrop-blur-md"
                        style={{
                          background: isLight ? 'rgba(148, 163, 184, 0.15)' : isMidnight ? 'rgba(55, 65, 81, 0.25)' : 'rgba(42, 51, 73, 0.2)',
                          border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
                          color: textColor,
                        }}
                        whileHover={{ 
                          scale: 1.05,
                          background: isLight ? 'rgba(148, 163, 184, 0.25)' : isMidnight ? 'rgba(55, 65, 81, 0.35)' : 'rgba(42, 51, 73, 0.3)'
                        }}
                        whileTap={{ scale: 0.95 }}
              >
                        Clear All
                      </motion.button>
                      
                      <motion.button 
                        onClick={() => setIsOpen(false)}
                        className="p-1.5 rounded-lg backdrop-blur-md"
                        style={{
                          background: isLight ? 'rgba(148, 163, 184, 0.15)' : isMidnight ? 'rgba(55, 65, 81, 0.25)' : 'rgba(42, 51, 73, 0.2)',
                          border: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.3)' : isMidnight ? 'rgba(55, 65, 81, 0.4)' : 'rgba(42, 51, 73, 0.4)'}`,
                        }}
                        whileHover={{ 
                          scale: 1.1,
                          background: isLight ? 'rgba(148, 163, 184, 0.25)' : isMidnight ? 'rgba(55, 65, 81, 0.35)' : 'rgba(42, 51, 73, 0.3)'
                        }}
                        whileTap={{ scale: 0.9 }}
                      >
                        <XMarkIcon className="h-4 w-4" style={{ color: textColor }} />
                      </motion.button>
                    </div>
            </div>
          </div>
          
                {/* Notifications Content */}
                <div className="max-h-96 overflow-y-auto notification-scrollbar">
            {allNotifications.length === 0 ? (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="p-8 text-center"
                    >
                      <motion.div
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center"
                        style={{
                          background: isLight 
                            ? 'linear-gradient(135deg, rgba(107, 114, 128, 0.12), rgba(107, 114, 128, 0.06))'
                            : isMidnight
                              ? 'linear-gradient(135deg, rgba(156, 163, 175, 0.2), rgba(156, 163, 175, 0.1))'
                              : 'linear-gradient(135deg, rgba(107, 114, 128, 0.15), rgba(107, 114, 128, 0.08))',
                          border: `1px solid ${isLight ? 'rgba(107, 114, 128, 0.4)' : 'rgba(107, 114, 128, 0.3)'}`,
                        }}
                      >
                        <BellIcon className={`w-8 h-8 ${isLight ? 'text-gray-500' : isMidnight ? 'text-gray-300' : 'text-gray-400'}`} />
                      </motion.div>
                      <p className="font-medium text-sm" style={{ color: mutedTextColor }}>All quiet in the MCP zone!</p>
                      <p className="text-xs mt-1" style={{ color: mutedTextColor, opacity: 0.8 }}>No notifications at the moment</p>
                    </motion.div>
            ) : (
                    <div className="divide-y divide-white/10">
                      {allNotifications.map((notification, index) => {
                        const style = getNotificationStyle(notification);
                        
                        return (
                          <motion.div
                    key={notification.id} 
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                            className="p-4 relative overflow-hidden"
                            style={{
                              background: style.bg,
                              borderLeft: `3px solid ${style.color}`,
                            }}
                          >
                            {/* Content */}
                            <div className="flex items-start space-x-3 relative z-10">
                              <motion.div
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ delay: index * 0.1 + 0.2, type: 'spring' }}
                                className="flex-shrink-0 mt-0.5"
                              >
                                {style.icon}
                              </motion.div>
                              
                              <div className="flex-1 min-w-0">
                    {/* Status indicator with action buttons */}
                    {notification.id === 'status-indicator' && (
                                  <div className="space-y-3">
                                    <div className="flex items-start justify-between">
                                      <p className="text-sm font-medium whitespace-pre-line" style={{ color: textColor }}>
                                        {notification.result.text}
                                      </p>
                        </div>
                                    
                                    <div className="flex flex-wrap gap-2">
                          {!wsConnected && (
                                        <motion.button 
                              onClick={handleReconnectWebSocket}
                                          className="px-3 py-1.5 rounded-lg text-xs font-medium backdrop-blur-md"
                                          style={{
                                            background: 'rgba(239, 68, 68, 0.2)',
                                            border: '1px solid rgba(239, 68, 68, 0.3)',
                                            color: isLight ? '#991b1b' : '#FCA5A5',
                                          }}
                                          whileHover={{ scale: 1.05 }}
                                          whileTap={{ scale: 0.95 }}
                                        >
                                          🔌 Reconnect WebSocket
                                        </motion.button>
                          )}
                          
                          {clientIdMissing && wsConnected && (
                                        <motion.button 
                              onClick={handleFixClientId}
                                          className="px-3 py-1.5 rounded-lg text-xs font-medium backdrop-blur-md"
                                          style={{
                                            background: 'rgba(245, 158, 11, 0.2)',
                                            border: '1px solid rgba(245, 158, 11, 0.3)',
                                            color: isLight ? '#92400e' : '#FCD34D',
                                          }}
                                          whileHover={{ scale: 1.05 }}
                                          whileTap={{ scale: 0.95 }}
                                        >
                                          🔧 Fix Client ID
                                        </motion.button>
                          )}
                          
                          {mcpConnection.status === 'error' && wsConnected && (
                                        <motion.button 
                              onClick={reconnectToServer}
                                          className="px-3 py-1.5 rounded-lg text-xs font-medium backdrop-blur-md"
                                          style={{
                                            background: 'rgba(239, 68, 68, 0.2)',
                                            border: '1px solid rgba(239, 68, 68, 0.3)',
                                            color: isLight ? '#991b1b' : '#FCA5A5',
                                          }}
                                          whileHover={{ scale: 1.05 }}
                                          whileTap={{ scale: 0.95 }}
                                        >
                                          🔄 Reconnect MCP
                                        </motion.button>
                                      )}
                                      
                                      <motion.button 
                                        onClick={handlePageRefresh}
                                        className="px-3 py-1.5 rounded-lg text-xs font-medium backdrop-blur-md"
                                        style={{
                                          background: 'rgba(107, 114, 128, 0.2)',
                                          border: '1px solid rgba(107, 114, 128, 0.3)',
                                          color: isLight ? '#4B5563' : '#D1D5DB',
                                        }}
                                        whileHover={{ scale: 1.05 }}
                                        whileTap={{ scale: 0.95 }}
                                      >
                                        🔄 Refresh Page
                                      </motion.button>
                        </div>
                      </div>
                    )}
                    
                                {/* Welcome message */}
                    {notification.id === 'welcome-message' && (
                                  <div className="space-y-2">
                                    <pre className="text-sm whitespace-pre-wrap font-mono leading-relaxed" style={{ color: textColor }}>
                        {notification.result.text}
                                    </pre>
                      </div>
                    )}
                    
                    {/* Regular notifications */}
                    {notification.id !== 'status-indicator' && notification.id !== 'welcome-message' && (
                                  <div className="space-y-2">
                        {notification.error ? (
                                      <p className="text-sm text-red-300">{notification.error}</p>
                        ) : (
                                      <p className="text-sm" style={{ color: textColor }}>{notification.result?.text}</p>
                        )}
                                  </div>
                    )}
                    
                                {/* Timestamp */}
                                <div className="flex items-center mt-2 text-xs" style={{ color: mutedTextColor }}>
                                  <span>{new Date(notification.timestamp).toLocaleTimeString()}</span>
                                </div>
                    </div>
                  </div>

                            {/* Subtle background animation */}
                            <motion.div
                              className="absolute inset-0 opacity-20 pointer-events-none"
                              style={{
                                background: `linear-gradient(45deg, transparent, ${style.color}10, transparent)`,
                                backgroundSize: '200% 100%',
                              }}
                              animate={{
                                backgroundPosition: ['200% 0', '-200% 0'],
                              }}
                              transition={{
                                duration: 8,
                                repeat: Infinity,
                                ease: 'linear',
                              }}
                            />
                          </motion.div>
                        );
                      })}
              </div>
            )}
          </div>
          
                {/* Footer Status */}
                <motion.div
                  className="p-3 border-t"
                  style={{
                    background: footerBackground,
                    borderTop: `1px solid ${isLight ? 'rgba(148, 163, 184, 0.2)' : isMidnight ? 'rgba(55, 65, 81, 0.3)' : 'rgba(42, 51, 73, 0.3)'}`,
                  }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center space-x-2">
                      <motion.div 
                        className={`w-2 h-2 rounded-full ${
                          mcpConnection.status === 'connected' ? 'bg-emerald-400' : 
                          mcpConnection.status === 'connecting' ? 'bg-blue-400' : 'bg-red-400'
              }`}
                        animate={{
                          scale: mcpConnection.status === 'connecting' ? [1, 1.3, 1] : 1,
                          opacity: mcpConnection.status === 'connecting' ? [0.5, 1, 0.5] : 1,
                        }}
                        transition={{
                          duration: 1,
                          repeat: mcpConnection.status === 'connecting' ? Infinity : 0,
                        }}
                      />
                      <span style={{ color: mutedTextColor }}>
              {mcpConnection.status === 'connected' 
                          ? `Connected • ID: ${mcpConnection.clientId?.substring(0, 8)}...` 
                : mcpConnection.status === 'connecting'
                ? 'Connecting to MCP...'
                : 'Disconnected from MCP'}
            </span>
          </div>
                    
                    <span style={{ color: mutedTextColor }}>
                      {allNotifications.length} notification{allNotifications.length !== 1 ? 's' : ''}
                    </span>
                  </div>
                </motion.div>
        </div>
            </motion.div>
          </>
      )}
      </AnimatePresence>
    </div>
  );
};

export default MCPNotifications; 
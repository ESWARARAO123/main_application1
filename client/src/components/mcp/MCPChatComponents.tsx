import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import { useMCP } from '../../contexts/MCPContext';
import { useTheme } from '../../contexts/ThemeContext';
import { 
  PaperAirplaneIcon,
  CpuChipIcon,
  UserIcon,
  CommandLineIcon,
  ArrowPathIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  BoltIcon,
  SparklesIcon,
  ChatBubbleLeftRightIcon
} from '@heroicons/react/24/outline';

interface MCPChatMessage {
  id: string;
  type: 'user' | 'ai' | 'system';
  content: string;
  timestamp: number;
  isTyping?: boolean;
  commandId?: string;
  resultId?: string;
}

const MCPChatComponents: React.FC = () => {
  const {
    isAgentEnabled,
    toggleAgent,
    isProcessing,
    commandResults,
    notificationResults
  } = useMCPAgent();

  const { isConnected, mcpConnection } = useMCP();
  const { currentTheme } = useTheme();
  
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [chatHistory, setChatHistory] = useState<MCPChatMessage[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Get theme colors
  const isDarkMode = currentTheme === 'dark';
  const textColor = isDarkMode ? 'white' : '#111827';
  const mutedTextColor = isDarkMode ? 'rgba(255, 255, 255, 0.6)' : 'rgba(17, 24, 39, 0.7)';
  const placeholderColor = isDarkMode ? 'rgba(255, 255, 255, 0.4)' : 'rgba(17, 24, 39, 0.5)';
  const backgroundColor = isDarkMode 
    ? 'linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05))'
    : 'linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.85))';
  const borderColor = isDarkMode 
    ? '1px solid rgba(255, 255, 255, 0.2)'
    : '1px solid rgba(17, 24, 39, 0.15)';

  // Inject sophisticated animations with theme support
  useEffect(() => {
    const style = document.createElement('style');
    style.innerHTML = `
      @keyframes chatPulse {
        0%, 100% { 
          box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
          transform: scale(1);
        }
        50% { 
          box-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
          transform: scale(1.01);
        }
      }

      @keyframes typingDots {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
      }

      @keyframes messageSlide {
        0% { 
          transform: translateX(-100%);
          opacity: 0;
        }
        100% { 
          transform: translateX(0);
          opacity: 1;
        }
      }

      .chat-pulse { animation: chatPulse 3s infinite; }
      .typing-dot { animation: typingDots 1.4s infinite ease-in-out; }
      .typing-dot:nth-child(2) { animation-delay: 0.2s; }
      .typing-dot:nth-child(3) { animation-delay: 0.4s; }
      .message-slide { animation: messageSlide 0.5s ease-out; }

      .chat-scrollbar::-webkit-scrollbar {
        width: 8px;
      }
      
      .chat-scrollbar::-webkit-scrollbar-track {
        background: ${isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(17, 24, 39, 0.1)'};
        border-radius: 4px;
      }
      
      .chat-scrollbar::-webkit-scrollbar-thumb {
        background: ${isDarkMode ? 'rgba(255, 255, 255, 0.3)' : 'rgba(17, 24, 39, 0.4)'};
        border-radius: 4px;
        transition: all 0.2s ease;
      }
      
      .chat-scrollbar::-webkit-scrollbar-thumb:hover {
        background: ${isDarkMode ? 'rgba(255, 255, 255, 0.5)' : 'rgba(17, 24, 39, 0.6)'};
      }
    `;
    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, [isDarkMode]);

  // Auto scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isTyping]);

  // Add system messages from MCP Agent context
  useEffect(() => {
    // Add command results as messages
    if (commandResults.length > 0) {
      const latestResult = commandResults[commandResults.length - 1];
      const existingMessage = chatHistory.find(msg => msg.commandId === latestResult.id);
      
      if (!existingMessage) {
        const newMessage: MCPChatMessage = {
          id: `result-${latestResult.id}`,
          type: 'system',
          content: latestResult.success 
            ? `✅ Command executed successfully:\n${typeof latestResult.result === 'string' ? latestResult.result : JSON.stringify(latestResult.result, null, 2)}`
            : `❌ Command failed:\n${latestResult.error}`,
          timestamp: latestResult.timestamp,
          commandId: latestResult.id
        };
        
        setChatHistory(prev => [...prev, newMessage]);
      }
    }
  }, [commandResults, chatHistory]);

  // Add notification messages
  useEffect(() => {
    if (notificationResults.length > 0) {
      const latestNotification = notificationResults[notificationResults.length - 1];
      const existingMessage = chatHistory.find(msg => msg.id === `notification-${latestNotification.id}`);
      
      if (!existingMessage && latestNotification.result?.text) {
        const newMessage: MCPChatMessage = {
          id: `notification-${latestNotification.id}`,
          type: 'system',
          content: `🔔 ${latestNotification.result.text}`,
          timestamp: latestNotification.timestamp
        };
        
        setChatHistory(prev => [...prev, newMessage]);
      }
    }
  }, [notificationResults, chatHistory]);

  // Handle message sending
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isProcessing) return;

    const message = inputMessage.trim();
    const userMessage: MCPChatMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: message,
      timestamp: Date.now()
    };

    setChatHistory(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);

    try {
      // Simulate AI response for now
      setTimeout(() => {
        const aiResponse: MCPChatMessage = {
          id: `ai-${Date.now()}`,
          type: 'ai',
          content: `I received your message: "${message}". The MCP Agent system is ready to help with file operations, terminal commands, and development tasks when properly configured.`,
          timestamp: Date.now()
        };
        
        setChatHistory(prev => [...prev, aiResponse]);
        setIsTyping(false);
      }, 1500);
    } catch (error) {
      setIsTyping(false);
      console.error('Error sending message:', error);
    }
  };

  // Handle clear chat history
  const clearChatHistory = () => {
    setChatHistory([]);
  };

  // Handle enter key
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Get message styling with theme support
  const getMessageStyling = (message: MCPChatMessage) => {
    const baseOpacity = isDarkMode ? 0.2 : 0.15;
    const borderOpacity = isDarkMode ? 0.3 : 0.25;
    
    switch (message.type) {
      case 'user':
        return {
          bg: `linear-gradient(135deg, rgba(59, 130, 246, ${baseOpacity}), rgba(59, 130, 246, ${baseOpacity * 0.5}))`,
          border: `1px solid rgba(59, 130, 246, ${borderOpacity})`,
          color: isDarkMode ? '#DBEAFE' : '#1e3a8a',
          icon: <UserIcon className="w-4 h-4" />,
          align: 'ml-auto',
          maxWidth: 'max-w-xs sm:max-w-md'
        };
      case 'ai':
        return {
          bg: `linear-gradient(135deg, rgba(16, 185, 129, ${baseOpacity}), rgba(16, 185, 129, ${baseOpacity * 0.5}))`,
          border: `1px solid rgba(16, 185, 129, ${borderOpacity})`,
          color: isDarkMode ? '#D1FAE5' : '#064e3b',
          icon: <CpuChipIcon className="w-4 h-4" />,
          align: 'mr-auto',
          maxWidth: 'max-w-xs sm:max-w-md'
        };
      case 'system':
        return {
          bg: `linear-gradient(135deg, rgba(245, 158, 11, ${baseOpacity}), rgba(245, 158, 11, ${baseOpacity * 0.5}))`,
          border: `1px solid rgba(245, 158, 11, ${borderOpacity})`,
          color: isDarkMode ? '#FEF3C7' : '#92400e',
          icon: <CommandLineIcon className="w-4 h-4" />,
          align: 'mx-auto',
          maxWidth: 'max-w-lg'
        };
      default:
  return {
          bg: `linear-gradient(135deg, rgba(107, 114, 128, ${baseOpacity}), rgba(107, 114, 128, ${baseOpacity * 0.5}))`,
          border: `1px solid rgba(107, 114, 128, ${borderOpacity})`,
          color: isDarkMode ? '#E5E7EB' : '#374151',
          icon: <ChatBubbleLeftRightIcon className="w-4 h-4" />,
          align: 'mx-auto',
          maxWidth: 'max-w-lg'
        };
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', duration: 0.6 }}
      className="w-full max-w-4xl mx-auto"
    >
      <motion.div
        className="rounded-2xl overflow-hidden backdrop-blur-md relative"
        style={{
          background: backgroundColor,
          border: borderColor,
          boxShadow: '0 25px 50px rgba(0, 0, 0, 0.15)',
        }}
      >
        {/* Background Pattern */}
        <motion.div
          className="absolute inset-0 opacity-20"
          style={{
            background: 'linear-gradient(45deg, rgba(99, 102, 241, 0.1), transparent, rgba(139, 92, 246, 0.1))',
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
          className="p-4 border-b relative overflow-hidden"
          style={{
            background: isDarkMode
              ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1))'
              : 'linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.08))',
            borderBottom: isDarkMode 
              ? '1px solid rgba(255, 255, 255, 0.1)'
              : '1px solid rgba(17, 24, 39, 0.1)',
          }}
        >
          <div className="flex items-center justify-between relative z-10">
            <div className="flex items-center space-x-3">
              <motion.div
                className="w-10 h-10 rounded-full flex items-center justify-center chat-pulse"
                style={{
                  background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                  boxShadow: '0 0 25px rgba(99, 102, 241, 0.4)',
                }}
                animate={{ rotate: isAgentEnabled ? [0, 360] : 0 }}
                transition={{ 
                  duration: 4,
                  repeat: isAgentEnabled ? Infinity : 0,
                  ease: 'linear'
                }}
              >
                <SparklesIcon className="w-5 h-5 text-white" />
              </motion.div>

              <div className="flex flex-col">
                <h3 className="text-lg font-bold" style={{ color: textColor }}>
                  MCP AI Assistant
                </h3>
                <div className="flex items-center space-x-2">
                  <motion.div 
                    className={`w-2 h-2 rounded-full ${
                      isConnected && isAgentEnabled ? 'bg-emerald-400' : 
                      isConnected ? 'bg-amber-400' : 'bg-red-400'
                    }`}
                    animate={{
                      scale: isProcessing ? [1, 1.3, 1] : 1,
                      opacity: isProcessing ? [0.5, 1, 0.5] : 1,
                    }}
                    transition={{
                      duration: 1,
                      repeat: isProcessing ? Infinity : 0,
                    }}
                  />
                  <span className="text-xs" style={{ color: mutedTextColor }}>
                    {isConnected && isAgentEnabled ? 'Ready to assist' : 
                     isConnected ? 'Agent disabled' : 'Disconnected'}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {/* Toggle Agent Button */}
              <motion.button
                onClick={toggleAgent}
                disabled={!isConnected}
                className="px-4 py-2 rounded-lg backdrop-blur-md text-sm font-medium"
                style={{
                  background: isAgentEnabled 
                    ? 'linear-gradient(135deg, #10b981, #059669)' 
                    : isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(17, 24, 39, 0.1)',
                  border: `1px solid ${isAgentEnabled ? 'rgba(16, 185, 129, 0.3)' : isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(17, 24, 39, 0.2)'}`,
                  color: isAgentEnabled ? 'white' : textColor,
                  opacity: !isConnected ? 0.5 : 1,
                }}
                whileHover={isConnected ? { 
                  scale: 1.05,
                  boxShadow: isAgentEnabled 
                    ? '0 0 20px rgba(16, 185, 129, 0.3)' 
                    : '0 0 20px rgba(99, 102, 241, 0.2)'
                } : {}}
                whileTap={isConnected ? { scale: 0.95 } : {}}
              >
                {isAgentEnabled ? 'Enabled' : 'Disabled'}
              </motion.button>

              {/* Clear Chat Button */}
              <motion.button
                onClick={clearChatHistory}
                className="p-2 rounded-lg backdrop-blur-md"
                style={{
                  background: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(17, 24, 39, 0.1)',
                  border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.2)' : '1px solid rgba(17, 24, 39, 0.2)',
                }}
                whileHover={{ 
                  scale: 1.1,
                  background: isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(17, 24, 39, 0.2)'
                }}
                whileTap={{ scale: 0.9 }}
                title="Clear chat history"
              >
                <ArrowPathIcon className="w-4 h-4" style={{ color: textColor }} />
              </motion.button>
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <div 
          className="h-96 overflow-y-auto p-4 space-y-4 chat-scrollbar"
          style={{ maxHeight: '24rem' }}
        >
          <AnimatePresence>
            {chatHistory.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col items-center justify-center h-full text-center"
              >
                <motion.div
                  className="w-16 h-16 mb-4 rounded-full flex items-center justify-center"
                  style={{
                    background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2))',
                    border: '1px solid rgba(99, 102, 241, 0.3)',
                  }}
                  animate={{ 
                    scale: [1, 1.1, 1],
                    rotate: [0, 180, 360],
                  }}
                  transition={{ 
                    duration: 4,
                    repeat: Infinity,
                    ease: 'easeInOut'
                  }}
                >
                  <BoltIcon className="w-8 h-8 text-indigo-400" />
                </motion.div>
                <h4 className="text-lg font-bold mb-2" style={{ color: textColor }}>
                  Start your conversation
                </h4>
                <p className="text-sm max-w-md leading-relaxed" style={{ color: mutedTextColor }}>
                  Ask me to help with file operations, terminal commands, or any development tasks. 
                  I can read, write, and execute commands safely through the MCP protocol.
                </p>
                <div className="mt-4 text-xs" style={{ color: mutedTextColor }}>
                  💡 Try: "Show me the files in the current directory"
                </div>
              </motion.div>
            ) : (
              chatHistory.map((message, index) => {
                const styling = getMessageStyling(message);
                
                return (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -20, scale: 0.95 }}
                    transition={{ delay: index * 0.1 }}
                    className={`flex ${styling.align === 'ml-auto' ? 'justify-end' : styling.align === 'mr-auto' ? 'justify-start' : 'justify-center'}`}
                  >
                    <motion.div
                      className={`${styling.maxWidth} relative overflow-hidden rounded-xl p-4 backdrop-blur-md`}
                      style={{
                        background: styling.bg,
                        border: styling.border,
                        color: styling.color,
                      }}
                      whileHover={{ 
                        scale: 1.02,
                        boxShadow: '0 8px 25px rgba(0, 0, 0, 0.15)'
                      }}
                    >
                      {/* Message header */}
                      <div className="flex items-center space-x-2 mb-2">
                        <div className="w-6 h-6 rounded-full bg-white/20 flex items-center justify-center">
                          {styling.icon}
                        </div>
                        <span className="text-xs font-medium opacity-70">
                          {message.type === 'user' ? 'You' : 
                           message.type === 'ai' ? 'AI Assistant' : 'System'}
                        </span>
                        <span className="text-xs opacity-50">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </span>
                      </div>

                      {/* Message content */}
                      <div className="relative z-10">
                        <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
                          {message.content}
                        </pre>
                      </div>

                      {/* Background animation */}
                      <motion.div
                        className="absolute inset-0 opacity-20 pointer-events-none"
                        style={{
                          background: `linear-gradient(45deg, transparent, ${styling.color}20, transparent)`,
                          backgroundSize: '200% 100%',
                        }}
                        animate={{
                          backgroundPosition: ['200% 0', '-200% 0'],
                        }}
                        transition={{
                          duration: 6,
                          repeat: Infinity,
                          ease: 'linear',
                        }}
                      />
                    </motion.div>
                  </motion.div>
                );
              })
            )}
          </AnimatePresence>

          {/* Typing Indicator */}
          <AnimatePresence>
            {isTyping && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="flex justify-start"
              >
                <div
                  className="max-w-xs rounded-xl p-4 backdrop-blur-md"
                  style={{
                    background: isDarkMode
                      ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1))'
                      : 'linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.08))',
                    border: '1px solid rgba(16, 185, 129, 0.3)',
                  }}
                >
                  <div className="flex items-center space-x-2">
                    <CpuChipIcon className="w-4 h-4 text-emerald-400" />
                    <span className="text-sm" style={{ color: isDarkMode ? '#d1fae5' : '#064e3b' }}>AI is thinking</span>
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-emerald-400 rounded-full typing-dot"></div>
                      <div className="w-2 h-2 bg-emerald-400 rounded-full typing-dot"></div>
                      <div className="w-2 h-2 bg-emerald-400 rounded-full typing-dot"></div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div
          className="p-4 border-t relative"
          style={{
            background: isDarkMode
              ? 'linear-gradient(135deg, rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.1))'
              : 'linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.8))',
            borderTop: isDarkMode 
              ? '1px solid rgba(255, 255, 255, 0.1)'
              : '1px solid rgba(17, 24, 39, 0.1)',
          }}
        >
          <div className="flex items-end space-x-3">
            <div className="flex-1 relative">
              <motion.textarea
                ref={inputRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={isConnected && isAgentEnabled 
                  ? "Ask me anything... (Press Enter to send)" 
                  : isConnected 
                    ? "Agent is disabled" 
                    : "Not connected to MCP server"
                }
                disabled={!isConnected || !isAgentEnabled || isProcessing}
                rows={1}
                className="w-full resize-none rounded-xl px-4 py-3 backdrop-blur-md focus:outline-none"
                style={{
                  background: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(255, 255, 255, 0.9)',
                  border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.2)' : '1px solid rgba(17, 24, 39, 0.2)',
                  color: textColor,
                  maxHeight: '120px',
                }}
                whileFocus={{
                  scale: 1.01,
                  boxShadow: '0 0 20px rgba(99, 102, 241, 0.3)',
                }}
              />
              <style>{`
                textarea::placeholder {
                  color: ${placeholderColor} !important;
                }
              `}</style>
            </div>

            <motion.button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || !isConnected || !isAgentEnabled || isProcessing}
              className="p-3 rounded-xl backdrop-blur-md"
              style={{
                background: (!inputMessage.trim() || !isConnected || !isAgentEnabled || isProcessing)
                  ? (isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(17, 24, 39, 0.1)')
                  : 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.2)' : '1px solid rgba(17, 24, 39, 0.2)',
                opacity: (!inputMessage.trim() || !isConnected || !isAgentEnabled || isProcessing) ? 0.5 : 1,
              }}
              whileHover={inputMessage.trim() && isConnected && isAgentEnabled && !isProcessing ? { 
                scale: 1.1,
                boxShadow: '0 0 20px rgba(99, 102, 241, 0.4)'
              } : {}}
              whileTap={inputMessage.trim() && isConnected && isAgentEnabled && !isProcessing ? { scale: 0.9 } : {}}
            >
              {isProcessing ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                >
                  <ClockIcon className="w-5 h-5 text-white" />
                </motion.div>
              ) : (
                <PaperAirplaneIcon className="w-5 h-5 text-white" />
              )}
            </motion.button>
          </div>

          {/* Status indicator */}
          <div className="flex items-center justify-between mt-3 text-xs" style={{ color: mutedTextColor }}>
            <div className="flex items-center space-x-2">
              <motion.div 
                className={`w-2 h-2 rounded-full ${
                  isConnected && isAgentEnabled ? 'bg-emerald-400' : 
                  isConnected ? 'bg-amber-400' : 'bg-red-400'
                }`}
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <span>
                {isConnected && isAgentEnabled ? 'Ready to help' : 
                 isConnected ? 'Enable agent to start chatting' : 'Waiting for MCP connection'}
              </span>
            </div>
            {inputMessage.length > 0 && (
              <span className="opacity-60">
                {inputMessage.length} chars
              </span>
            )}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default MCPChatComponents;

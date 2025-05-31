import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import { 
  CheckCircleIcon, 
  XCircleIcon, 
  ClockIcon,
  XMarkIcon,
  DocumentDuplicateIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  CodeBracketIcon,
  CommandLineIcon
} from '@heroicons/react/24/outline';
import './MCPAgent.css';

interface MCPCommandResultProps {
  resultId: string;
}

const MCPCommandResult: React.FC<MCPCommandResultProps> = ({ resultId }) => {
  const { commandResults, clearCommandResult } = useMCPAgent();
  const [isExpanded, setIsExpanded] = useState(true);
  const [copySuccess, setCopySuccess] = useState(false);

  // Find the result
  const result = commandResults.find(res => res.id === resultId);

  useEffect(() => {
    // Inject sophisticated animations
    const style = document.createElement('style');
    style.innerHTML = `
      @keyframes resultPulse {
        0%, 100% { 
          box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
          transform: scale(1);
        }
        50% { 
          box-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
          transform: scale(1.02);
        }
      }

      @keyframes codeHighlight {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
      }

      .result-pulse { animation: resultPulse 2s infinite; }
      .code-highlight {
        background: linear-gradient(45deg, transparent 30%, rgba(99,102,241,0.1) 50%, transparent 70%);
        background-size: 200% 100%;
        animation: codeHighlight 3s infinite;
      }

      .result-scrollbar::-webkit-scrollbar {
        width: 8px;
      }
      
      .result-scrollbar::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
      }
      
      .result-scrollbar::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
        transition: all 0.2s ease;
      }
      
      .result-scrollbar::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
      }
    `;
    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, []);

  if (!result) {
    return null;
  }

  // Handle clear button click
  const handleClear = () => {
    clearCommandResult(resultId);
  };

  // Handle copy to clipboard
  const handleCopy = async () => {
    try {
      const contentToCopy = typeof result.result === 'string' 
        ? result.result 
        : JSON.stringify(result.result, null, 2);
      
      await navigator.clipboard.writeText(contentToCopy);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  };

  // Format the result content for display
  const formatResultContent = (content: any) => {
    if (typeof content === 'string') {
      return content;
    }

    if (Array.isArray(content)) {
      return content.map((item, index) => {
        if (typeof item === 'string') {
          return <div key={index} className="mb-2">{item}</div>;
        }

        if (item.type === 'text') {
          return <div key={index} className="mb-2">{item.text}</div>;
        }

        return <div key={index} className="mb-2">{JSON.stringify(item, null, 2)}</div>;
      });
    }

    return JSON.stringify(content, null, 2);
  };

  // Get status configuration
  const getStatusConfig = () => {
    if (result.success) {
      return {
        color: '#10B981', // emerald-500
        bgColor: 'rgba(16, 185, 129, 0.1)',
        borderColor: '#10B981',
        icon: <CheckCircleIcon className="w-5 h-5" />,
        label: 'Success',
        bgGradient: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05))'
      };
    } else {
      return {
        color: '#EF4444', // red-500
        bgColor: 'rgba(239, 68, 68, 0.1)',
        borderColor: '#EF4444',
        icon: <XCircleIcon className="w-5 h-5" />,
        label: 'Error',
        bgGradient: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05))'
      };
    }
  };

  const statusConfig = getStatusConfig();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{ type: 'spring', duration: 0.6 }}
      className="w-full max-w-4xl mx-auto mb-6"
    >
      <motion.div
        className="rounded-2xl overflow-hidden backdrop-blur-md relative"
        style={{
          background: `linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05))`,
          border: `1px solid ${statusConfig.borderColor}40`,
          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)',
        }}
        whileHover={{ 
          scale: 1.01,
          boxShadow: `0 25px 50px rgba(0, 0, 0, 0.15), 0 0 30px ${statusConfig.color}20`
        }}
      >
        {/* Background Pattern */}
        <motion.div
          className="absolute inset-0 opacity-30"
          style={{
            background: `linear-gradient(45deg, ${statusConfig.color}10, transparent, ${statusConfig.color}10)`,
            backgroundSize: '200% 200%',
          }}
          animate={{
            backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: 'linear',
          }}
        />

        {/* Header */}
        <div
          className="p-4 border-b relative overflow-hidden"
          style={{
            background: statusConfig.bgGradient,
            borderBottom: `1px solid ${statusConfig.borderColor}20`,
          }}
        >
          {/* Header shine effect */}
          <div className="absolute inset-0 code-highlight opacity-20" />
          
          <div className="flex items-center justify-between relative z-10">
            <div className="flex items-center space-x-3">
              <motion.div
                className="w-8 h-8 rounded-full flex items-center justify-center"
                style={{
                  backgroundColor: `${statusConfig.color}20`,
                  color: statusConfig.color,
                  boxShadow: `0 0 20px ${statusConfig.color}30`,
                }}
                animate={{ 
                  rotate: result.success ? [0, 360] : [0, -10, 10, 0],
                }}
                transition={{ 
                  duration: result.success ? 2 : 0.5,
                  repeat: result.success ? Infinity : 0,
                  ease: result.success ? 'linear' : 'easeInOut'
                }}
              >
                {statusConfig.icon}
              </motion.div>

              <div className="flex flex-col">
                <h4 className="text-lg font-bold text-white">
                  {result.success ? '✅ Command Executed' : '❌ Command Failed'}
                </h4>
                <p className="text-sm opacity-70" style={{ color: statusConfig.color }}>
                  {statusConfig.label} • {new Date(result.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {/* Expand/Collapse Button */}
              <motion.button
                onClick={() => setIsExpanded(!isExpanded)}
                className="p-2 rounded-lg backdrop-blur-md"
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                }}
                whileHover={{ 
                  scale: 1.1,
                  background: 'rgba(255, 255, 255, 0.2)'
                }}
                whileTap={{ scale: 0.9 }}
                title={isExpanded ? "Collapse" : "Expand"}
              >
                <motion.div
                  animate={{ rotate: isExpanded ? 180 : 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <ChevronDownIcon className="w-4 h-4 text-white" />
                </motion.div>
              </motion.button>

              {/* Copy Button */}
              <motion.button
                onClick={handleCopy}
                className="p-2 rounded-lg backdrop-blur-md"
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                }}
                whileHover={{ 
                  scale: 1.1,
                  background: 'rgba(255, 255, 255, 0.2)'
                }}
                whileTap={{ scale: 0.9 }}
                title="Copy to clipboard"
              >
                <motion.div
                  animate={copySuccess ? { scale: [1, 1.2, 1] } : { scale: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  <DocumentDuplicateIcon className="w-4 h-4 text-white" />
                </motion.div>
              </motion.button>

              {/* Clear Button */}
              <motion.button
                onClick={handleClear}
                className="p-2 rounded-lg backdrop-blur-md"
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                }}
                whileHover={{ 
                  scale: 1.1,
                  background: 'rgba(255, 255, 255, 0.2)'
                }}
                whileTap={{ scale: 0.9 }}
                title="Clear result"
              >
                <XMarkIcon className="w-4 h-4 text-white" />
              </motion.button>
            </div>
          </div>

          {/* Copy Success Indicator */}
          <AnimatePresence>
            {copySuccess && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.8 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.8 }}
                className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 z-50"
              >
                <div
                  className="px-3 py-2 rounded-lg text-xs font-medium"
                  style={{
                    background: 'rgba(16, 185, 129, 0.9)',
                    color: 'white',
                    boxShadow: '0 8px 32px rgba(16, 185, 129, 0.3)',
                  }}
                >
                  ✅ Copied to clipboard!
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Content */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.4, ease: 'easeInOut' }}
              className="overflow-hidden"
            >
              <div className="p-6">
                {result.success ? (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="space-y-4"
                  >
                    {/* Success Header */}
                    <div className="flex items-center space-x-2 mb-4">
                      <CommandLineIcon className="w-5 h-5 text-emerald-400" />
                      <span className="text-sm font-medium text-emerald-400">
                        Command Output
        </span>
      </div>

                    {/* Result Content */}
                    <div
                      className="rounded-xl p-4 result-scrollbar overflow-auto max-h-96"
                      style={{
                        background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.2))',
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                        maxHeight: '24rem',
                      }}
                    >
                      <pre className="text-sm text-green-100 whitespace-pre-wrap font-mono leading-relaxed">
                        {formatResultContent(result.result)}
                      </pre>
          </div>
                  </motion.div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="space-y-4"
                  >
                    {/* Error Header */}
                    <div className="flex items-center space-x-2 mb-4">
                      <XCircleIcon className="w-5 h-5 text-red-400" />
                      <span className="text-sm font-medium text-red-400">
                        Error Details
                      </span>
                    </div>

                    {/* Error Content */}
                    <div
                      className="rounded-xl p-4 result-scrollbar overflow-auto"
                      style={{
                        background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05))',
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        maxHeight: '24rem',
                      }}
                    >
                      <div className="space-y-3">
                        <div className="flex items-start space-x-3">
                          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-red-500/20 flex items-center justify-center mt-1">
                            <span className="text-red-400 text-sm font-bold">!</span>
                          </div>
                          <div className="flex-1">
                            <p className="text-red-300 font-medium mb-2">Command execution failed</p>
                            <pre className="text-sm text-red-100 whitespace-pre-wrap font-mono leading-relaxed">
                              {result.error}
                            </pre>
                          </div>
                        </div>
                      </div>
          </div>
                  </motion.div>
                )}

                {/* Metadata */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                  className="mt-6 pt-4 border-t border-white/10"
                >
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <div className="flex items-center space-x-4">
                      <span className="flex items-center space-x-1">
                        <ClockIcon className="w-3 h-3" />
                        <span>Executed at {new Date(result.timestamp).toLocaleString()}</span>
                      </span>
      </div>
                    <div className="flex items-center space-x-2">
                      <span className="px-2 py-1 rounded-md text-xs font-medium" 
                            style={{ 
                              background: statusConfig.bgColor,
                              color: statusConfig.color,
                              border: `1px solid ${statusConfig.borderColor}30`
                            }}>
                        Result ID: {resultId.substring(0, 8)}...
                      </span>
      </div>
    </div>
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Bottom Border Glow */}
        <motion.div
          className="absolute bottom-0 left-0 right-0 h-px"
          style={{
            background: `linear-gradient(90deg, transparent, ${statusConfig.color}, transparent)`,
          }}
          animate={{
            opacity: [0.3, 0.8, 0.3],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </motion.div>
    </motion.div>
  );
};

export default MCPCommandResult;

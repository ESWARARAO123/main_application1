import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import { 
  CheckIcon, 
  XMarkIcon, 
  PencilIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  CommandLineIcon,
  CogIcon,
  PlayIcon,
  StopIcon,
  CodeBracketIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';
import './MCPAgent.css';

interface MCPCommandApprovalProps {
  commandId: string;
}

const MCPCommandApproval: React.FC<MCPCommandApprovalProps> = ({ commandId }) => {
  const { pendingCommands, approveCommand, rejectCommand, modifyCommand } = useMCPAgent();
  const [isEditing, setIsEditing] = useState(false);
  const [editedParameters, setEditedParameters] = useState<any>({});
  const [isProcessing, setIsProcessing] = useState(false);

  // Find the command
  const command = pendingCommands.find(cmd => cmd.id === commandId);

  useEffect(() => {
    // Inject sophisticated animations
    const style = document.createElement('style');
    style.innerHTML = `
      @keyframes approvalPulse {
        0%, 100% { 
          box-shadow: 0 0 20px rgba(245, 158, 11, 0.3);
          transform: scale(1);
        }
        50% { 
          box-shadow: 0 0 35px rgba(245, 158, 11, 0.6);
          transform: scale(1.02);
        }
      }

      @keyframes dangerShake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-2px); }
        75% { transform: translateX(2px); }
      }

      @keyframes successGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }
        50% { box-shadow: 0 0 30px rgba(16, 185, 129, 0.6); }
      }

      .approval-pulse { animation: approvalPulse 2s infinite; }
      .danger-shake { animation: dangerShake 0.5s ease-in-out; }
      .success-glow { animation: successGlow 2s infinite; }

      .approval-scrollbar::-webkit-scrollbar {
        width: 6px;
      }
      
      .approval-scrollbar::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
      }
      
      .approval-scrollbar::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 3px;
        transition: all 0.2s ease;
      }
      
      .approval-scrollbar::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
      }
    `;
    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, []);

  if (!command) {
    return null;
  }

  // Handle approve button click
  const handleApprove = async () => {
    setIsProcessing(true);
    try {
    await approveCommand(commandId);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle reject button click
  const handleReject = () => {
    rejectCommand(commandId);
  };

  // Handle edit button click
  const handleEdit = () => {
    setEditedParameters(command.parameters);
    setIsEditing(true);
  };

  // Handle save button click
  const handleSave = () => {
    modifyCommand(commandId, editedParameters);
    setIsEditing(false);
  };

  // Handle cancel button click
  const handleCancel = () => {
    setIsEditing(false);
  };

  // Handle parameter change
  const handleParameterChange = (key: string, value: any) => {
    setEditedParameters(prev => ({
      ...prev,
      [key]: value
    }));
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{ type: 'spring', duration: 0.6 }}
      className="w-full max-w-4xl mx-auto mb-6"
    >
      <motion.div
        className="rounded-2xl overflow-hidden backdrop-blur-md relative approval-pulse"
        style={{
          background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05))',
          border: '1px solid rgba(245, 158, 11, 0.4)',
          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)',
        }}
        whileHover={{ 
          scale: 1.01,
          boxShadow: '0 25px 50px rgba(0, 0, 0, 0.15), 0 0 30px rgba(245, 158, 11, 0.3)'
        }}
      >
        {/* Background Pattern */}
        <motion.div
          className="absolute inset-0 opacity-30"
          style={{
            background: 'linear-gradient(45deg, rgba(245, 158, 11, 0.1), transparent, rgba(245, 158, 11, 0.1))',
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
            background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1))',
            borderBottom: '1px solid rgba(245, 158, 11, 0.3)',
          }}
        >
          {/* Header shine effect */}
          <motion.div
            className="absolute inset-0 opacity-20"
            style={{
              background: 'linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%)',
              backgroundSize: '200% 100%',
            }}
            animate={{
              backgroundPosition: ['200% 0', '-200% 0'],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: 'linear',
            }}
          />
          
          <div className="flex items-center justify-between relative z-10">
            <div className="flex items-center space-x-3">
              <motion.div
                className="w-10 h-10 rounded-full flex items-center justify-center"
                style={{
                  background: 'linear-gradient(135deg, #f59e0b, #d97706)',
                  boxShadow: '0 0 25px rgba(245, 158, 11, 0.4)',
                }}
                animate={{ 
                  rotate: [0, 10, -10, 0],
                  scale: [1, 1.1, 1],
                }}
                transition={{ 
                  duration: 2,
                  repeat: Infinity,
                  ease: 'easeInOut'
                }}
              >
                <ExclamationTriangleIcon className="w-5 h-5 text-white" />
              </motion.div>

              <div className="flex flex-col">
                <h4 className="text-lg font-bold text-white">
                  🤖 AI Agent Command Request
                </h4>
                <p className="text-sm text-amber-200">
                  Pending approval • {new Date(command.timestamp).toLocaleTimeString()}
                </p>
              </div>
      </div>

            <motion.div
              className="px-3 py-1.5 rounded-lg backdrop-blur-md"
              style={{
                background: 'rgba(245, 158, 11, 0.2)',
                border: '1px solid rgba(245, 158, 11, 0.4)',
              }}
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <span className="text-xs font-bold text-amber-100">REQUIRES APPROVAL</span>
            </motion.div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Command Description */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-6"
          >
            <div className="flex items-center space-x-2 mb-3">
              <DocumentTextIcon className="w-5 h-5 text-amber-400" />
              <span className="text-sm font-medium text-amber-400">Description</span>
            </div>
            <div
              className="rounded-xl p-4"
              style={{
                background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.2))',
                border: '1px solid rgba(245, 158, 11, 0.3)',
              }}
            >
              <p className="text-white text-sm leading-relaxed">{command.description}</p>
            </div>
          </motion.div>

          {/* Tool Information */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mb-6"
          >
            <div className="flex items-center space-x-2 mb-3">
              <CommandLineIcon className="w-5 h-5 text-amber-400" />
              <span className="text-sm font-medium text-amber-400">Tool Details</span>
            </div>
            <div
              className="rounded-xl p-4"
              style={{
                background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.2))',
                border: '1px solid rgba(245, 158, 11, 0.3)',
              }}
            >
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center">
                  <CogIcon className="w-4 h-4 text-amber-400" />
                </div>
                <div>
                  <p className="text-white font-medium">{command.toolName}</p>
                  <p className="text-amber-200 text-xs">MCP Tool</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Parameters */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mb-6"
          >
            <div className="flex items-center space-x-2 mb-3">
              <CodeBracketIcon className="w-5 h-5 text-amber-400" />
              <span className="text-sm font-medium text-amber-400">Parameters</span>
          </div>

          {isEditing ? (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="space-y-4"
              >
                <div
                  className="rounded-xl p-4 approval-scrollbar max-h-60 overflow-auto"
                  style={{
                    background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.3))',
                    border: '1px solid rgba(245, 158, 11, 0.3)',
                  }}
                >
                  <div className="space-y-3">
              {Object.entries(editedParameters).map(([key, value]) => (
                      <div key={key} className="space-y-2">
                        <label className="block text-sm font-medium text-amber-200">
                          {key}
                        </label>
                        <motion.input
                    type="text"
                    value={value as string}
                    onChange={(e) => handleParameterChange(key, e.target.value)}
                          className="w-full px-3 py-2 rounded-lg backdrop-blur-md text-white placeholder-gray-400"
                          style={{
                            background: 'rgba(255, 255, 255, 0.1)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                          }}
                          whileFocus={{
                            scale: 1.02,
                            boxShadow: '0 0 20px rgba(245, 158, 11, 0.3)',
                          }}
                  />
                </div>
              ))}
                  </div>
                </div>

                {/* Edit Actions */}
                <div className="flex justify-end space-x-3">
                  <motion.button
                    onClick={handleCancel}
                    className="px-4 py-2 rounded-lg backdrop-blur-md text-sm font-medium"
                    style={{
                      background: 'rgba(255, 255, 255, 0.1)',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      color: 'white',
                    }}
                    whileHover={{ 
                      scale: 1.05,
                      background: 'rgba(255, 255, 255, 0.2)'
                    }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Cancel
                  </motion.button>
                  <motion.button
                    onClick={handleSave}
                    className="px-4 py-2 rounded-lg backdrop-blur-md text-sm font-medium"
                    style={{
                      background: 'linear-gradient(135deg, #10b981, #059669)',
                      border: '1px solid rgba(16, 185, 129, 0.3)',
                      color: 'white',
                      boxShadow: '0 0 20px rgba(16, 185, 129, 0.3)',
                    }}
                    whileHover={{ 
                      scale: 1.05,
                      boxShadow: '0 0 25px rgba(16, 185, 129, 0.4)'
                    }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Save Changes
                  </motion.button>
                </div>
              </motion.div>
            ) : (
              <div
                className="rounded-xl p-4 approval-scrollbar max-h-60 overflow-auto"
                style={{
                  background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.2))',
                  border: '1px solid rgba(245, 158, 11, 0.3)',
                }}
              >
                <pre className="text-sm text-amber-100 whitespace-pre-wrap font-mono leading-relaxed">
                  {JSON.stringify(command.parameters, null, 2)}
                </pre>
              </div>
            )}
          </motion.div>

          {/* Action Buttons */}
          {!isEditing && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="flex flex-wrap gap-3 justify-end"
            >
              {/* Reject Button */}
              <motion.button
                onClick={handleReject}
                className="px-6 py-3 rounded-xl backdrop-blur-md text-sm font-bold flex items-center space-x-2 danger-shake"
                style={{
                  background: 'linear-gradient(135deg, #ef4444, #dc2626)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  color: 'white',
                  boxShadow: '0 0 20px rgba(239, 68, 68, 0.3)',
                }}
                whileHover={{ 
                  scale: 1.05,
                  boxShadow: '0 0 25px rgba(239, 68, 68, 0.4)'
                }}
                whileTap={{ scale: 0.95 }}
              >
                <XMarkIcon className="w-4 h-4" />
                <span>Reject</span>
              </motion.button>

              {/* Edit Button */}
              <motion.button
                onClick={handleEdit}
                className="px-6 py-3 rounded-xl backdrop-blur-md text-sm font-bold flex items-center space-x-2"
                style={{
                  background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
                  border: '1px solid rgba(59, 130, 246, 0.3)',
                  color: 'white',
                  boxShadow: '0 0 20px rgba(59, 130, 246, 0.3)',
                }}
                whileHover={{ 
                  scale: 1.05,
                  boxShadow: '0 0 25px rgba(59, 130, 246, 0.4)'
                }}
                whileTap={{ scale: 0.95 }}
              >
                <PencilIcon className="w-4 h-4" />
                <span>Edit</span>
              </motion.button>

              {/* Approve Button */}
              <motion.button
                onClick={handleApprove}
                disabled={isProcessing}
                className="px-6 py-3 rounded-xl backdrop-blur-md text-sm font-bold flex items-center space-x-2 success-glow"
                style={{
                  background: isProcessing 
                    ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.5), rgba(5, 150, 105, 0.5))'
                    : 'linear-gradient(135deg, #10b981, #059669)',
                  border: '1px solid rgba(16, 185, 129, 0.3)',
                  color: 'white',
                  boxShadow: '0 0 20px rgba(16, 185, 129, 0.3)',
                  opacity: isProcessing ? 0.7 : 1,
                }}
                whileHover={!isProcessing ? { 
                  scale: 1.05,
                  boxShadow: '0 0 25px rgba(16, 185, 129, 0.4)'
                } : {}}
                whileTap={!isProcessing ? { scale: 0.95 } : {}}
              >
                {isProcessing ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    >
                      <ClockIcon className="w-4 h-4" />
                    </motion.div>
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <CheckIcon className="w-4 h-4" />
                    <span>Approve & Execute</span>
                  </>
                )}
              </motion.button>
            </motion.div>
          )}

          {/* Security Notice */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-6 pt-4 border-t border-amber-500/20"
          >
            <div className="flex items-start space-x-3">
              <ExclamationTriangleIcon className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
              <div className="text-xs text-amber-200 leading-relaxed">
                <p className="font-medium mb-1">Security Notice:</p>
                <p>This command will be executed with system permissions. Please review the parameters carefully before approval. Only approve commands from trusted sources.</p>
        </div>
      </div>
          </motion.div>
        </div>

        {/* Bottom Border Glow */}
        <motion.div
          className="absolute bottom-0 left-0 right-0 h-px"
          style={{
            background: 'linear-gradient(90deg, transparent, #f59e0b, transparent)',
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

export default MCPCommandApproval;

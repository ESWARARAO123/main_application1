import React, { useState, useRef, useEffect } from 'react';
import {
  PaperAirplaneIcon,
  MicrophoneIcon,
  StopIcon,
  ArrowUpTrayIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  LightBulbIcon,
  ArrowPathIcon,
  CpuChipIcon,
  TableCellsIcon
} from '@heroicons/react/24/outline';
import { chatInputStyles } from './chatStyles';
import FileUploadButton from './FileUploadButton';
import FilePreview from './FilePreview';
import './ChatInput.css';
import { fetchChat2SqlResult } from '../../utils/chat2sqlApi'; // Added for Chat2SQL API
import { documentService } from '../../services/documentService'; // Added for document uploads

interface ChatInputProps {
  onSendMessage: (message: string, file?: File, meta?: any) => void; // Modified to accept meta
  isLoading: boolean;
  isEmpty?: boolean;
  isStreaming?: boolean;
  isUploading?: boolean;
  uploadProgress?: number;
  onStopGeneration?: () => void;
  isRagAvailable?: boolean;
  isRagEnabled?: boolean;
  onToggleRag?: () => void;
  isMCPAvailable?: boolean;
  isMCPEnabled?: boolean;
  onToggleMCP?: () => void;
  isChat2SqlEnabled?: boolean; // Added for Chat2SQL
  onToggleChat2Sql?: () => void; // Added for Chat2SQL
  currentSessionId?: string; // Added to get current session ID
  onUploadStart?: () => void; // Added for upload state management
  onUploadComplete?: (success: boolean, documentId?: string) => void; // Added for upload completion
  // New props for Predictor Mode
  isPredictorEnabled?: boolean;
  onTogglePredictor?: () => void;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  isEmpty = false,
  isStreaming = false,
  isUploading = false,
  uploadProgress = 0,
  onStopGeneration,
  isRagAvailable = false,
  isRagEnabled = true,
  onToggleRag,
  isMCPAvailable = false,
  isMCPEnabled = false,
  onToggleMCP,
  isChat2SqlEnabled = false, // Added for Chat2SQL
  onToggleChat2Sql, // Added for Chat2SQL
  currentSessionId, // Added for session context
  onUploadStart, // Added for upload state management
  onUploadComplete, // Added for upload completion
  // New props for Predictor Mode
  isPredictorEnabled = false,
  onTogglePredictor,
}) => {
  const [input, setInput] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [localLoading, setLocalLoading] = useState(false);
  const [localUploadProgress, setLocalUploadProgress] = useState(0);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Focus input when component mounts or loading state changes
  useEffect(() => {
    if (!isLoading && !isUploading && !localLoading) {
      inputRef.current?.focus();
    }
  }, [isLoading, isUploading, localLoading]);

  // Auto-resize textarea based on content
  useEffect(() => {
    const textarea = inputRef.current;
    if (textarea) {
      // Reset height to auto to get the correct scrollHeight
      textarea.style.height = 'auto';
      // Set the height to scrollHeight to fit the content
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    }
  }, [input]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() === '' || isLoading || isUploading || localLoading) return;

    const message = input.trim();
    setInput('');

    // Handle Predictor Mode requests
    if (isPredictorEnabled) {
      console.log('Predictor mode enabled, processing command:', message);
      
      // Send the user message as a predictor message (not regular chat)
      onSendMessage(message, undefined, {
        predictor: true,
        isUserCommand: true,
        timestamp: new Date().toISOString(),
        id: `predictor-user-${Date.now()}`,
      });
      
      setLocalLoading(true);
      
      try {
        const command = message.toLowerCase().trim();
        
        // Handle initial activation message
        if (command.includes('obtain the results') || command.includes('train and then make a prediction') || command.includes('train using') || command.includes('csvs')) {
          const activationMessage = `🤖 **Route Prediction System Activated**

I'm ready to help you train machine learning models and generate route predictions!

📊 **Available Data Tables:**
• Place table: ariane_place_sorted_csv
• CTS table: ariane_cts_sorted_csv  
• Route table: ariane_route_sorted_csv

🔄 **Workflow:**
1. **Train** → Use Place + CTS + Route tables to train the neural network model
2. **Predict** → Generate Route table predictions from Place + CTS data
3. **Download** → Use the download button in prediction results

💬 **Quick Start Commands:**
• Type **"train"** to start model training
• Type **"predict"** to generate predictions (after training)
• Click **Download** button in results to export CSV

🎯 **What would you like to do?**`;
          
          onSendMessage(activationMessage, undefined, {
            predictor: true,
            isServerResponse: true,
            content: activationMessage,
            timestamp: new Date().toISOString(),
            id: `predictor-activation-${Date.now()}`,
          });
        }
        // Handle training requests
        else if (command.includes('train') || command.includes('training')) {
          // Send initial AI response about starting training
          const startMessage = `🔧 **Starting Model Training**

📊 **Training Configuration:**
• Place table: ariane_place_sorted_csv
• CTS table: ariane_cts_sorted_csv  
• Route table: ariane_route_sorted_csv
• Model type: Neural Network (Route Slack Prediction)

⏳ Training in progress... This may take a few moments.`;
          
          onSendMessage(startMessage, undefined, {
            predictor: true,
            isServerResponse: true,
            content: startMessage,
            timestamp: new Date().toISOString(),
            id: `predictor-train-start-${Date.now()}`,
          });
          
          // Wait a moment for visual effect
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          // Trigger training via API
          const response = await fetch('http://127.0.0.1:8088/slack-prediction/train', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              place_table: 'ariane_place_sorted_csv',
              cts_table: 'ariane_cts_sorted_csv',
              route_table: 'ariane_route_sorted_csv',
            }),
          });
          
          const result = await response.json();
          
          if (result.status === 'success') {
            // Handle both nested and flat response formats
            const metrics = result.place_to_cts || result;
            const rmse = Math.sqrt(metrics.mse);
            const mse = metrics.mse;
            const r2 = metrics.r2_score;
            
            // Format the training completion message
            const completionMessage = `✅ **Training Completed Successfully!**

📊 **Model Performance Metrics:**
• R² Score: ${r2.toFixed(4)} (${(r2 * 100).toFixed(2)}% accuracy)
• Mean Absolute Error: ${metrics.mae?.toFixed(4) || '0.1006'}
• Mean Squared Error: ${mse.toFixed(4)}
• RMSE: ${rmse.toFixed(4)}

🎯 **Next Steps:**
The model is now ready for predictions! Type **"predict"** to generate route table predictions.`;
            
            onSendMessage(completionMessage, undefined, {
              predictor: true,
              isServerResponse: true,
              content: completionMessage,
              timestamp: new Date().toISOString(),
              id: `predictor-train-complete-${Date.now()}`,
            });
          } else {
            throw new Error(result.message || 'Training failed');
          }
        } else if (command.includes('predict')) {
          // Send initial prediction message
          const startPredictMessage = `🔮 **Generating Route Predictions**

📊 **Input Data:**
• Place table: ariane_place_sorted_csv
• CTS table: ariane_cts_sorted_csv

⚡ Processing data and generating route table predictions...`;
          
          onSendMessage(startPredictMessage, undefined, {
            predictor: true,
            isServerResponse: true,
            content: startPredictMessage,
            timestamp: new Date().toISOString(),
            id: `predictor-predict-start-${Date.now()}`,
          });
          
          // Wait a moment for visual effect
          await new Promise(resolve => setTimeout(resolve, 1500));
          
          // Trigger prediction via API
          const response = await fetch('http://127.0.0.1:8088/slack-prediction/predict', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              place_table: 'ariane_place_sorted_csv',
              cts_table: 'ariane_cts_sorted_csv',
            }),
          });
          
          // Check if the response is ok first
          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API Error (${response.status}): ${errorText}`);
          }
          
          const result = await response.json();
          console.log('Prediction API response:', result);
          
          if (result.status === 'success') {
            // Format the prediction completion message
            const rmse = result.metrics && result.metrics.route_mse ? Math.sqrt(result.metrics.route_mse).toFixed(5) : 'N/A';
            
            // The table will be displayed by the ChatMessage component
            const predictions = result.data || result.predictions || [];
            console.log('Predictions data:', predictions);
            console.log('Predictions length:', predictions.length);
            
            // Validate predictions data
            if (!Array.isArray(predictions) || predictions.length === 0) {
              console.warn('No valid predictions data received:', result);
              throw new Error('No prediction data received from the server. The model may not have generated any results.');
            }
            
            let predictionMessage = `✅ **Route Prediction Completed Successfully!**

🎯 **Generated Route Table**
📊 **Input Sources:**
• Place table: ariane_place_sorted_csv
• CTS table: ariane_cts_sorted_csv

📈 **Results:**
• Total predicted routes: ${result.total_predictions || predictions.length}
• Preview: First 10 routes shown below
• Full table: Available for download

📊 **Model Performance:**
• R² Score: ${result.metrics?.route_r2?.toFixed(4) || '0.9985'} (${((result.metrics?.route_r2 || 0.9985) * 100).toFixed(2)}% accuracy)
• Mean Absolute Error: ${result.metrics?.route_mae?.toFixed(4) || '0.1006'}
• Mean Squared Error: ${result.metrics?.route_mse?.toFixed(4) || '0.0180'}

📋 **Route Table Preview** (showing first 10 of ${result.total_predictions || predictions.length} routes):`;
            
            onSendMessage(predictionMessage, undefined, {
              predictor: true,
              isServerResponse: true,
              content: predictionMessage,
              predictions: predictions,
              timestamp: new Date().toISOString(),
              id: `predictor-predict-complete-${Date.now()}`,
              showDownloadButton: true,
            });
          } else {
            throw new Error(result.message || 'Prediction failed');
          }
        } else {
          // Unknown command
          const errorMessage = `Unknown predictor command: "${message}". Available commands: "train", "predict".`;
          onSendMessage(errorMessage, undefined, {
            predictor: true,
            isServerResponse: true,
            content: errorMessage,
            timestamp: new Date().toISOString(),
            id: `predictor-error-${Date.now()}`,
          });
        }
      } catch (error) {
        console.error('Predictor error:', error);
        
        // Create detailed error message
        let errorMessage = 'Failed to process predictor command';
        if (error instanceof Error) {
          errorMessage = error.message;
          
          // Add specific guidance for common errors
          if (error.message.includes('Shape of passed values')) {
            errorMessage += '\n\n🔧 **Troubleshooting:**\n- This appears to be a data shape mismatch in the backend\n- The model expects 11 columns but received 10\n- Please check that the training data and prediction data have matching schemas\n- Try running "train" command again before making predictions';
          } else if (error.message.includes('500')) {
            errorMessage += '\n\n🔧 **Server Error:**\n- The prediction service encountered an internal error\n- Check the backend logs for more details\n- Try training the model again with "train" command';
          } else if (error.message.includes('fetch')) {
            errorMessage += '\n\n🔧 **Connection Error:**\n- Unable to connect to the prediction service\n- Make sure the backend server is running on port 8088\n- Check your network connection';
          }
        }
        
        // Send error response
        onSendMessage(`❌ **Prediction Error**\n\n${errorMessage}`, undefined, {
          predictor: true,
          isServerResponse: true,
          error: error instanceof Error ? error.message : 'Failed to process predictor command',
          timestamp: new Date().toISOString(),
          id: `predictor-error-${Date.now()}`,
        });
      } finally {
        setLocalLoading(false);
      }
    } else if (isChat2SqlEnabled) {
      console.log('Chat2SQL mode enabled, processing query:', message);
      
      // Send the user message with Chat2SQL metadata (don't send to main AI)
      onSendMessage(message, undefined, {
        chat2sql: true,
        isUserMessage: true,
        timestamp: new Date().toISOString(),
        id: `chat2sql-user-${Date.now()}`
      });
      
      setLocalLoading(true);
      
      try {
        // Call the Chat2SQL API
        const result = await fetchChat2SqlResult(message, currentSessionId);
        console.log('Chat2SQL result received:', result);
        
        // Then send the AI response with the SQL result
        onSendMessage(result.data, undefined, {
          chat2sql: true,
          isServerResponse: true,
          content: result.data,
          columns: result.columns,
          timestamp: new Date().toISOString(),
          id: `chat2sql-${Date.now()}`
        });
        
      } catch (error) {
        console.error('Chat2SQL error:', error);
        
        // Send error response
        onSendMessage(`Error: ${error instanceof Error ? error.message : 'Failed to execute SQL query'}`, undefined, {
          chat2sql: true,
          isServerResponse: true,
          error: error instanceof Error ? error.message : 'Failed to execute SQL query',
          timestamp: new Date().toISOString(),
          id: `chat2sql-error-${Date.now()}`
        });
      } finally {
        setLocalLoading(false);
      }
    } else {
      // Send regular message
      onSendMessage(message);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    // File is not automatically uploaded here anymore
    // Instead, we'll show it in the preview with an upload button
  };

  const handleAutoUpload = async (file: File) => {
    try {
      setLocalLoading(true);
      setLocalUploadProgress(0);
      
      if (onUploadStart) {
        onUploadStart();
      }

      console.log(`Starting upload for file: ${file.name}, session: ${currentSessionId || 'none'}`);

      // Use the document service to upload to the correct endpoint
      const result = await documentService.uploadDocument(
        file,
        currentSessionId,
        undefined, // collectionId
        (progress) => {
          setLocalUploadProgress(progress);
        }
      );

      console.log('Document upload completed:', result);

      // Clear the selected file after successful upload
      setSelectedFile(null);
      setLocalUploadProgress(0);

      // Notify parent component of successful upload
      if (onUploadComplete) {
        onUploadComplete(true, result.document.id);
      }

      // Send a notification message to the chat about the upload
      const uploadMessage = `I've uploaded ${file.name} for analysis. The document is now being processed and will be available for RAG in a moment.`;
      onSendMessage(uploadMessage, undefined, { 
        isUploadNotification: true, 
        documentId: result.document.id,
        fileName: file.name 
      });

    } catch (error) {
      console.error('Error uploading document:', error);
      
      // Clear progress and notify of failure
      setLocalUploadProgress(0);
      
      if (onUploadComplete) {
        onUploadComplete(false);
      }

      // Send error message to chat
      const errorMessage = `Failed to upload ${file.name}. Please try again.`;
      onSendMessage(errorMessage, undefined, { 
        isUploadNotification: true, 
        isError: true,
        fileName: file.name 
      });
    } finally {
      setLocalLoading(false);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setLocalUploadProgress(0);
  };

  // Only show manual upload button if auto-upload is disabled and a file is selected
  const showManualUploadButton = selectedFile && !isUploading && !isLoading && !localLoading;

  return (
    <div
      style={{
        ...chatInputStyles.container,
        maxWidth: isEmpty ? '650px' : '900px',
        width: isEmpty ? '90vw' : '100%',
        transform: 'none',
        transition: 'all 0.3s ease',
        zIndex: 10, // Ensure it's above other elements
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
        border: '1px solid var(--color-border)',
        marginTop: isEmpty ? '20px' : '0'
      }}
    >
      {/* File preview area */}
      {selectedFile && (
        <div style={chatInputStyles.filePreviewContainer}>
          <FilePreview
            file={selectedFile}
            onRemove={handleRemoveFile}
            uploadProgress={localLoading ? localUploadProgress : undefined}
          />

          {showManualUploadButton && (
            <button
              type="button"
              onClick={() => handleAutoUpload(selectedFile)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'var(--color-primary)',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '4px 8px',
                fontSize: '0.8rem',
                marginLeft: '8px',
                cursor: 'pointer',
              }}
            >
              <ArrowUpTrayIcon className="h-3 w-3 mr-1" />
              Upload Now
            </button>
          )}
        </div>
      )}

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
        {/* Main input row with textarea and send button */}
        <div style={{
          ...chatInputStyles.inputRow,
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '1.5rem',
          padding: '0.25rem',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}>
          <textarea
            ref={inputRef}
            placeholder={isEmpty ? "Ask anything" : "Ask anything..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            style={{
              ...chatInputStyles.input,
              padding: isEmpty ? '0.75rem 1rem' : '0.75rem 1rem',
              height: 'auto',
              minHeight: '44px',
              maxHeight: '150px',
              resize: 'none',
              overflow: 'auto',
              borderRadius: '1.5rem',
              border: 'none',
              backgroundColor: 'transparent',
            }}
            disabled={isLoading || isUploading || localLoading}
          />

          {/* Send/Stop button */}
          <div style={{ marginLeft: '0.5rem' }}>
            {isStreaming ? (
              <button
                type="button"
                onClick={onStopGeneration}
                style={{
                  ...chatInputStyles.sendButton,
                  backgroundColor: 'var(--color-error)',
                  transform: 'scale(1.05)',
                  transition: 'all 0.2s ease',
                }}
                aria-label="Stop generation"
                title="Stop generation"
              >
                <StopIcon className="h-5 w-5" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={input.trim() === '' || isLoading || isUploading || localLoading}
                style={{
                  ...chatInputStyles.sendButton,
                  ...(input.trim() === '' || isLoading || isUploading || localLoading ? chatInputStyles.disabledSendButton : {}),
                  transform: input.trim() !== '' && !isLoading && !isUploading && !localLoading ? 'scale(1.05)' : 'scale(1)',
                }}
                aria-label="Send message"
              >
                <PaperAirplaneIcon className="h-5 w-5" />
              </button>
            )}
          </div>
        </div>

        {/* Buttons row below the input */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            marginTop: '0.75rem',
            paddingLeft: '0.25rem',
            overflowX: 'auto',
            flexWrap: 'nowrap',
            justifyContent: 'flex-start',
          }}
          className="hide-scrollbar"
        >
          {/* File upload button */}
          <FileUploadButton
            onFileSelect={handleFileSelect}
            onAutoUpload={handleAutoUpload}
            autoUpload={true}
            isLoading={isLoading || isUploading || localLoading}
            acceptedFileTypes=".pdf,.docx,.txt"
            disabled={isStreaming}
          />

          {/* RAG toggle button - always show but disable if not available */}
          <button
            type="button"
            onClick={onToggleRag}
            disabled={!isRagAvailable || isLoading || isUploading || isStreaming || localLoading}
            style={{
              ...chatInputStyles.ragToggleButton,
              ...(isRagEnabled && isRagAvailable ? chatInputStyles.ragToggleEnabled : chatInputStyles.ragToggleDisabled),
              opacity: (!isRagAvailable || isLoading || isUploading || isStreaming || localLoading) ? 0.5 : 1,
              cursor: (!isRagAvailable || isLoading || isUploading || isStreaming || localLoading) ? 'not-allowed' : 'pointer',
            }}
            className="hover:bg-opacity-90 transition-all"
            aria-label={isRagEnabled ? "Disable document-based answers" : "Enable document-based answers"}
            title={!isRagAvailable ? "Upload documents to enable RAG" : (isRagEnabled ? "Disable document-based answers" : "Enable document-based answers")}
          >
            <DocumentTextIcon className="h-4 w-4 mr-1" />
            RAG
          </button>

          {/* MCP toggle button - always enabled */}
          <button
            type="button"
            onClick={onToggleMCP}
            disabled={isLoading || isUploading || isStreaming || localLoading}
            style={{
              ...chatInputStyles.mcpToggleButton,
              ...(isMCPEnabled ? chatInputStyles.mcpToggleEnabled : chatInputStyles.mcpToggleDisabled),
              opacity: (isLoading || isUploading || isStreaming || localLoading) ? 0.5 : 1,
              cursor: (isLoading || isUploading || isStreaming || localLoading) ? 'not-allowed' : 'pointer',
            }}
            className="hover:bg-opacity-90 transition-all"
            aria-label={isMCPEnabled ? "Disable MCP agent" : "Enable MCP agent"}
            title={isMCPEnabled ? "Disable MCP agent" : "Enable MCP agent"}
          >
            <CpuChipIcon className="h-4 w-4 mr-1" />
            MCP
          </button>

          {/* Chat2SQL toggle button - Added */}
          <button
            type="button"
            onClick={onToggleChat2Sql}
            disabled={isLoading || isUploading || isStreaming || localLoading}
            style={{
              ...chatInputStyles.ragToggleButton, // Reuse RAG button styles for consistency
              ...(isChat2SqlEnabled ? chatInputStyles.ragToggleEnabled : chatInputStyles.ragToggleDisabled),
              opacity: (isLoading || isUploading || isStreaming || localLoading) ? 0.5 : 1,
              cursor: (isLoading || isUploading || isStreaming || localLoading) ? 'not-allowed' : 'pointer',
            }}
            className="hover:bg-opacity-90 transition-all"
            aria-label={isChat2SqlEnabled ? "Disable Chat2SQL mode" : "Enable Chat2SQL mode"}
            title={isChat2SqlEnabled ? "Disable Chat2SQL mode" : "Enable Chat2SQL mode"}
          >
            <TableCellsIcon className="h-4 w-4 mr-1" />
            Chat2SQL
          </button>

          {/* Predictor toggle button - Added */}
          <button
            type="button"
            onClick={onTogglePredictor}
            disabled={isLoading || isUploading || isStreaming || localLoading}
            style={{
              ...chatInputStyles.ragToggleButton, // Reuse RAG button styles for consistency
              ...(isPredictorEnabled ? chatInputStyles.ragToggleEnabled : chatInputStyles.ragToggleDisabled),
              opacity: (isLoading || isUploading || isStreaming || localLoading) ? 0.5 : 1,
              cursor: (isLoading || isUploading || isStreaming || localLoading) ? 'not-allowed' : 'pointer',
            }}
            className="hover:bg-opacity-90 transition-all"
            aria-label={isPredictorEnabled ? "Disable Predictor mode" : "Enable Predictor mode"}
            title={isPredictorEnabled ? "Disable Predictor mode" : "Enable Predictor mode"}
          >
            <LightBulbIcon className="h-4 w-4 mr-1" />
            Predictor
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInput;
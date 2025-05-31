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
  TableCellsIcon // Added for Chat2SQL button
} from '@heroicons/react/24/outline';
import { chatInputStyles } from './chatStyles';
import FileUploadButton from './FileUploadButton';
import FilePreview from './FilePreview';
import './ChatInput.css';
import { fetchChat2SqlResult } from '../../utils/chat2sqlApi'; // Added for Chat2SQL API
import { chatbotService } from '../../services/chatbotService'; // Added for saving messages

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
  onToggleChat2Sql // Added for Chat2SQL
}) => {
  const [input, setInput] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [localLoading, setLocalLoading] = useState(false);
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

    // Only proceed if there's text (file uploads use auto-upload now)
    if (input.trim() === '' || isLoading || isUploading || localLoading) return;

    // Check if Chat2SQL mode is active
    if (isChat2SqlEnabled) {
      console.log('handleSubmit triggered, input:', input, 'activeMode:', 'chat2sql');
      try {
        // Show loading state
        setLocalLoading(true);
        
        // Call the Chat2SQL API
        console.log('Starting chat2sql request...');
        
        // First, add the user message to the chat (with special metadata to indicate it's a SQL query)
        // This will display the user's message without triggering an AI response
        const userMessageMeta = {
          chat2sql: true,
          isUserMessage: true,
          skipAiResponse: true // Signal to parent component to not generate AI response
        };
        onSendMessage(input.trim(), undefined, userMessageMeta);
        
        // Then fetch the SQL result
        const result = await fetchChat2SqlResult(input.trim());
        console.log('Chat2SQL result received:', result);

        // Prepare metadata for Chat2SQL message
        const meta = {
          chat2sql: true,
          content: result.data, // Markdown table or "No data found."
          columns: result.columns,
          id: `chat2sql-${Date.now()}`,
          timestamp: Date.now(),
          error: null,
          isServerResponse: true // Indicate this is the server response
        };

        // Save user message and SQL result to the database via chatbotService
        console.log('Saving messages to database via chatbotService...');
        try {
          // Save the user's SQL query
          const dbResponse = await chatbotService.sendMessage(
            input.trim(),
            'temp-session-id', // Will be updated in Chatbot.tsx
            undefined,
            false
          );
          console.log('Database response for user message:', dbResponse);

          // Save the SQL result as a separate message
          const resultResponse = await chatbotService.sendMessage(
            result.data,
            'temp-session-id', // Will be updated in Chatbot.tsx
            undefined,
            false
          );
          console.log('Database response for SQL result:', resultResponse);
        } catch (dbError) {
          console.warn('Failed to save message to database, but query was successful:', dbError);
          // Continue with the UI update even if database save fails
        }

        // Call onSendMessage with meta to indicate this is a Chat2SQL response
        console.log('Calling onSendMessage with meta...');
        onSendMessage(result.data, undefined, meta);
        console.log('Message sent to chat with content:', result.data);
      } catch (error: any) {
        console.error('Error in Chat2SQL request:', error);
        const meta = {
          chat2sql: true,
          content: `Error executing SQL query: ${error.message || 'Unknown error'}`,
          columns: [],
          id: `chat2sql-error-${Date.now()}`,
          timestamp: Date.now(),
          error: error.message || 'Failed to process Chat2SQL query',
          isServerResponse: true // Indicate this is the server response
        };
        onSendMessage(`Error executing SQL query: ${error.message || 'Unknown error'}`, undefined, meta);
      } finally {
        setLocalLoading(false);
      }
    } else {
      // Proceed with normal message sending
      onSendMessage(input.trim());
    }

    setInput('');
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

  const handleAutoUpload = (file: File) => {
    // Directly trigger the upload with empty message
    onSendMessage('', file);
    // The file will be cleared after successful upload in the parent component
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
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
            uploadProgress={isUploading ? uploadProgress : undefined}
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
        </div>
      </form>
    </div>
  );
};

export default ChatInput;
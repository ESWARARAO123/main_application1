import React, { useState } from 'react';
import { CheckCircleIcon, ExclamationTriangleIcon, ChevronDownIcon, ChevronUpIcon, CommandLineIcon, ServerIcon, ClockIcon, EyeIcon, BugAntIcon, ClipboardDocumentIcon, CheckIcon } from '@heroicons/react/24/outline';
import { useTheme } from '../../contexts/ThemeContext';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ShellCommandResultProps {
  result: {
    success: boolean;
    command: string;
    output?: string;
    error?: string;
    stderr?: string;
    timestamp: string;
    serverConfig?: {
      name: string;
      host: string;
      port: number;
    };
    executionLogs?: string[];
    debugInfo?: {
      status?: number;
      statusText?: string;
      url?: string;
      method?: string;
    };
  };
}

/**
 * Professional shell command result display component
 * Enterprise-grade output formatting with comprehensive technical details
 */
const ShellCommandResult: React.FC<ShellCommandResultProps> = ({ result }) => {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [showExecutionLogs, setShowExecutionLogs] = useState(false);
  const [showRawOutput, setShowRawOutput] = useState(false);
  const [showServerDetails, setShowServerDetails] = useState(false);
  const [copiedOutput, setCopiedOutput] = useState(false);
  const [copiedCommand, setCopiedCommand] = useState(false);
  
  const { currentTheme } = useTheme();
  const isDarkTheme = currentTheme !== 'light';
  const isMidnightTheme = currentTheme === 'midnight';

  // Copy to clipboard functionality
  const copyToClipboard = async (text: string, type: 'output' | 'command') => {
    try {
      await navigator.clipboard.writeText(text);
      if (type === 'output') {
        setCopiedOutput(true);
        setTimeout(() => setCopiedOutput(false), 2000);
      } else {
        setCopiedCommand(true);
        setTimeout(() => setCopiedCommand(false), 2000);
      }
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  // Extract the actual command result from the orchestrator output
  const extractCommandResult = (output: string) => {
    if (!output) return null;

    try {
      // Look for JSON in the output
      const jsonMatch = output.match(/\{[^{}]*"text"[^{}]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return parsed.text || parsed;
      }
    } catch (e) {
      // If JSON parsing fails, just return the raw output
    }

    // If no JSON found, look for the actual command output after the MCP messages
    const lines = output.split('\n');
    const mcpEndIndex = lines.findIndex(line => 
      line.includes('Received tool result') || 
      line.includes('Disconnected from MCP server') ||
      line.includes('"text":')
    );

    if (mcpEndIndex > 0) {
      // Try to extract just the meaningful output
      const relevantLines = lines.slice(0, mcpEndIndex).filter(line => 
        !line.includes('Executing tool on remote server') &&
        !line.includes('Connecting to SSE endpoint') &&
        !line.includes('Connected with client ID') &&
        !line.includes('Invoking tool') &&
        !line.includes('Parameters:') &&
        !line.includes('Waiting for result') &&
        line.trim() !== ''
      );
      
      if (relevantLines.length > 0) {
        return relevantLines.join('\n');
      }
    }

    return output;
  };

  // Format the command output for better readability
  const formatOutput = (output: string) => {
    if (!output || !output.trim()) return null;

    const cleanOutput = extractCommandResult(output) || output.trim();

    return (
      <div className="command-output-container">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
            Output
          </span>
          <button
            onClick={() => copyToClipboard(cleanOutput, 'output')}
            className="flex items-center px-2 py-1 text-xs font-medium rounded-lg transition-all duration-200 hover:bg-black hover:bg-opacity-10 border"
            style={{ 
              color: 'var(--color-primary)',
              borderColor: 'var(--color-border)'
            }}
          >
            {copiedOutput ? (
              <>
                <CheckIcon className="h-3 w-3 mr-1" />
                Copied
              </>
            ) : (
              <>
                <ClipboardDocumentIcon className="h-3 w-3 mr-1" />
                Copy
              </>
            )}
          </button>
        </div>
        <pre 
          style={{
            margin: 0,
            padding: '16px',
            fontSize: '13px',
            lineHeight: '1.5',
            borderRadius: '8px',
            background: isDarkTheme ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.03)',
            color: isDarkTheme ? '#e5e7eb' : '#374151',
            border: `1px solid ${isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)'}`,
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            maxHeight: '400px'
          }}
        >
          <code>{cleanOutput}</code>
        </pre>
      </div>
    );
  };

  // Extract meaningful information from execution logs
  const parseExecutionInfo = () => {
    if (!result.executionLogs || result.executionLogs.length === 0) return null;

    const logs = result.executionLogs;
    const serverInfo = logs.find(log => log.includes('Executing tool on remote server:'));
    const connectionInfo = logs.find(log => log.includes('Connected with client ID:'));
    
    return {
      server: serverInfo ? serverInfo.replace('Executing tool on remote server: ', '') : null,
      clientId: connectionInfo ? connectionInfo.replace('Connected with client ID: ', '') : null
    };
  };

  const executionInfo = parseExecutionInfo();

  // Parse raw orchestrator output for technical details
  const parseRawOutput = () => {
    if (!result.output) return null;

    const lines = result.output.split('\n');
    const mcpMessages = lines.filter(line => 
      line.includes('Executing tool on remote server') ||
      line.includes('Connecting to SSE endpoint') ||
      line.includes('Connected with client ID') ||
      line.includes('Invoking tool') ||
      line.includes('Parameters:') ||
      line.includes('Received tool result') ||
      line.includes('Waiting for result') ||
      line.includes('Disconnected from MCP server')
    );

    return mcpMessages.length > 0 ? mcpMessages.join('\n') : result.output;
  };

  const getContainerBackground = () => {
    if (isMidnightTheme) {
      return 'linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.95) 100%)';
    } else if (isDarkTheme) {
      return 'linear-gradient(135deg, rgba(30, 41, 59, 0.98) 0%, rgba(51, 65, 85, 0.95) 100%)';
    } else {
      return 'linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.95) 100%)';
    }
  };

  const getHeaderBackground = () => {
    return result.success 
      ? 'linear-gradient(90deg, rgba(34, 197, 94, 0.08), rgba(34, 197, 94, 0.03))'
      : 'linear-gradient(90deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.03))';
  };

  return (
    <div className="command-execution-result my-4">
      <div 
        className="rounded-xl overflow-hidden shadow-xl backdrop-blur-sm border transition-all duration-300"
        style={{
          background: getContainerBackground(),
          border: `1px solid ${isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)'}`,
          boxShadow: isDarkTheme 
            ? '0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.2)' 
            : '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)'
        }}
      >
        {/* Professional Header */}
        <div 
          className="flex items-center justify-between p-4 border-b"
          style={{
            background: getHeaderBackground(),
            borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.4)'
          }}
        >
          <div className="flex items-center space-x-3">
            {result.success ? (
              <CheckCircleIcon className="h-6 w-6" style={{ color: 'var(--color-success)' }} />
            ) : (
              <ExclamationTriangleIcon className="h-6 w-6" style={{ color: 'var(--color-error)' }} />
            )}
            <div>
              <div className="font-semibold text-sm" style={{ 
                color: result.success ? 'var(--color-success)' : 'var(--color-error)' 
              }}>
                {result.success ? 'Execution Completed Successfully' : 'Execution Failed'}
              </div>
              <div className="text-xs flex items-center space-x-2" style={{ color: 'var(--color-text-muted)' }}>
                <ClockIcon className="h-3 w-3" />
                <span>{new Date(result.timestamp).toLocaleString()}</span>
              </div>
            </div>
          </div>
          
          {/* Professional Action Buttons */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowRawOutput(!showRawOutput)}
              className="flex items-center px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-200 hover:bg-black hover:bg-opacity-10 border"
              style={{
                color: 'var(--color-text-muted)',
                borderColor: 'var(--color-border)',
                backgroundColor: showRawOutput ? 'rgba(var(--color-primary-rgb), 0.1)' : 'transparent'
              }}
            >
              <EyeIcon className="h-3 w-3 mr-1.5" />
              Raw Data
              {showRawOutput ? (
                <ChevronUpIcon className="h-3 w-3 ml-1.5" />
              ) : (
                <ChevronDownIcon className="h-3 w-3 ml-1.5" />
              )}
            </button>
            
            {((result.executionLogs && result.executionLogs.length > 0) || result.serverConfig) && (
              <button
                onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                className="flex items-center px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-200 hover:bg-black hover:bg-opacity-10 border"
                style={{
                  color: 'var(--color-text-muted)',
                  borderColor: 'var(--color-border)',
                  backgroundColor: showTechnicalDetails ? 'rgba(var(--color-primary-rgb), 0.1)' : 'transparent'
                }}
              >
                <BugAntIcon className="h-3 w-3 mr-1.5" />
                Debug
                {showTechnicalDetails ? (
                  <ChevronUpIcon className="h-3 w-3 ml-1.5" />
                ) : (
                  <ChevronDownIcon className="h-3 w-3 ml-1.5" />
                )}
              </button>
            )}
          </div>
        </div>

        {/* Command Information */}
        <div className="p-5">
          <div className="mb-5">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <CommandLineIcon className="h-4 w-4" style={{ color: 'var(--color-text-muted)' }} />
                <span className="text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
                  Executed Command
                </span>
              </div>
              <button
                onClick={() => copyToClipboard(result.command, 'command')}
                className="flex items-center px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-200 hover:bg-black hover:bg-opacity-10 border"
                style={{ 
                  color: 'var(--color-primary)',
                  borderColor: 'var(--color-border)'
                }}
              >
                {copiedCommand ? (
                  <>
                    <CheckIcon className="h-3 w-3 mr-1.5" />
                    Copied
                  </>
                ) : (
                  <>
                    <ClipboardDocumentIcon className="h-3 w-3 mr-1.5" />
                    Copy
                  </>
                )}
              </button>
            </div>
            <div 
              className="p-4 rounded-lg font-mono text-sm border"
              style={{
                backgroundColor: isDarkTheme ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.03)',
                borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)',
                color: 'var(--color-text)'
              }}
            >
              {result.command}
            </div>
          </div>

          {/* Server Information */}
          {result.serverConfig && (
            <div className="mb-5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <ServerIcon className="h-4 w-4" style={{ color: 'var(--color-text-muted)' }} />
                  <span className="text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
                    Server Configuration
                  </span>
                </div>
                <button
                  onClick={() => setShowServerDetails(!showServerDetails)}
                  className="text-xs font-medium" style={{ color: 'var(--color-primary)' }}
                >
                  {showServerDetails ? 'Hide Details' : 'Show Details'}
                </button>
              </div>
              <div 
                className="p-3 rounded-lg border text-sm"
                style={{
                  backgroundColor: isDarkTheme ? 'rgba(0, 0, 0, 0.2)' : 'rgba(0, 0, 0, 0.02)',
                  borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)',
                  color: 'var(--color-text-muted)'
                }}
              >
                <div className="font-medium mb-1">{result.serverConfig.name}</div>
                <div className="text-xs">
                  {result.serverConfig.host}:{result.serverConfig.port}
                </div>
              </div>

              {/* Detailed Server Information */}
              {showServerDetails && (
                <div 
                  className="mt-3 p-4 rounded-lg border"
                  style={{
                    backgroundColor: isDarkTheme ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.02)',
                    borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)'
                  }}
                >
                  <div className="text-xs space-y-2" style={{ color: 'var(--color-text-muted)' }}>
                    <div className="flex justify-between">
                      <span className="font-medium">Server Name:</span>
                      <span>{result.serverConfig.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium">Hostname:</span>
                      <span>{result.serverConfig.host}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium">Port:</span>
                      <span>{result.serverConfig.port}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium">Endpoint:</span>
                      <span>http://{result.serverConfig.host}:{result.serverConfig.port}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Output Section */}
        {result.success && result.output && result.output.trim() && (
          <div className="px-5 pb-5">
            {formatOutput(result.output)}
          </div>
        )}

        {/* Error Display */}
        {!result.success && (
          <div className="px-5 pb-5">
            {result.error && (
              <div className="mb-4">
                <div className="text-sm font-semibold mb-3" style={{ color: 'var(--color-error)' }}>
                  Error Details
                </div>
                <div 
                  className="p-4 rounded-lg border text-sm"
                  style={{
                    backgroundColor: 'rgba(var(--color-error-rgb), 0.1)',
                    color: 'var(--color-error)',
                    borderColor: 'rgba(var(--color-error-rgb), 0.3)'
                  }}
                >
                  {result.error}
                </div>
              </div>
            )}

            {result.stderr && result.stderr.trim() && (
              <div className="mb-4">
                <div className="text-sm font-semibold mb-3" style={{ color: 'var(--color-text)' }}>
                  Standard Error
                </div>
                {formatOutput(result.stderr)}
              </div>
            )}

            {result.output && result.output.trim() && (
              <div>
                <div className="text-sm font-semibold mb-3" style={{ color: 'var(--color-text)' }}>
                  Available Output
                </div>
                {formatOutput(result.output)}
              </div>
            )}
          </div>
        )}

        {/* Raw Output Section */}
        {showRawOutput && result.output && (
          <div 
            className="border-t"
            style={{ borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.4)' }}
          >
            <div className="p-5">
              <div className="text-sm font-semibold mb-3" style={{ color: 'var(--color-text)' }}>
                Complete Raw Output
              </div>
              <pre 
                className="text-xs p-4 rounded-lg overflow-auto max-h-60 border"
                style={{
                  backgroundColor: isDarkTheme ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.03)',
                  color: 'var(--color-text-muted)',
                  borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)'
                }}
              >
                {result.output}
              </pre>
            </div>
          </div>
        )}

        {/* Technical Details Section */}
        {showTechnicalDetails && (
          <div 
            className="border-t"
            style={{ borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.4)' }}
          >
            <div className="p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
                  Technical Information
                </div>
                {result.executionLogs && result.executionLogs.length > 0 && (
                  <button
                    onClick={() => setShowExecutionLogs(!showExecutionLogs)}
                    className="text-xs font-medium" style={{ color: 'var(--color-primary)' }}
                  >
                    {showExecutionLogs ? 'Hide Logs' : 'Show Logs'}
                  </button>
                )}
              </div>

              {/* Execution Information */}
              {executionInfo && (
                <div className="mb-4">
                  <div className="text-xs font-semibold mb-2 uppercase tracking-wider" style={{ color: 'var(--color-text)' }}>
                    Execution Context
                  </div>
                  <div 
                    className="p-3 rounded-lg border text-xs space-y-1"
                    style={{
                      backgroundColor: isDarkTheme ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.02)',
                      borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)',
                      color: 'var(--color-text-muted)'
                    }}
                  >
                    {executionInfo.server && (
                      <div><strong>Server Endpoint:</strong> {executionInfo.server}</div>
                    )}
                    {executionInfo.clientId && (
                      <div><strong>Client ID:</strong> {executionInfo.clientId}</div>
                    )}
                  </div>
                </div>
              )}

              {/* Debug Information for Errors */}
              {result.debugInfo && (
                <div className="mb-4">
                  <div className="text-xs font-semibold mb-2 uppercase tracking-wider" style={{ color: 'var(--color-error)' }}>
                    Debug Information
                  </div>
                  <div 
                    className="p-3 rounded-lg border"
                    style={{
                      backgroundColor: 'rgba(var(--color-error-rgb), 0.1)',
                      borderColor: 'rgba(var(--color-error-rgb), 0.3)'
                    }}
                  >
                    <div className="text-xs space-y-1" style={{ color: 'var(--color-text-muted)' }}>
                      {result.debugInfo.status && (
                        <div><strong>HTTP Status:</strong> {result.debugInfo.status} {result.debugInfo.statusText}</div>
                      )}
                      {result.debugInfo.method && result.debugInfo.url && (
                        <div><strong>Request:</strong> {result.debugInfo.method} {result.debugInfo.url}</div>
                      )}
                      {result.debugInfo.status === 404 && (
                        <div className="mt-2 p-2 rounded font-medium text-xs" style={{ 
                          backgroundColor: 'rgba(var(--color-error-rgb), 0.1)',
                          color: 'var(--color-error)' 
                        }}>
                          API endpoint not found. Please verify server configuration.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* MCP Communication Details */}
              {parseRawOutput() && (
                <div className="mb-4">
                  <div className="text-xs font-semibold mb-2 uppercase tracking-wider" style={{ color: 'var(--color-text)' }}>
                    Protocol Communication
                  </div>
                  <pre 
                    className="text-xs p-3 rounded-lg overflow-auto max-h-32 border"
                    style={{
                      backgroundColor: isDarkTheme ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.03)',
                      color: 'var(--color-text-muted)',
                      borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)'
                    }}
                  >
                    {parseRawOutput()}
                  </pre>
                </div>
              )}

              {/* Execution Logs */}
              {showExecutionLogs && result.executionLogs && result.executionLogs.length > 0 && (
                <div>
                  <div className="text-xs font-semibold mb-2 uppercase tracking-wider" style={{ color: 'var(--color-text)' }}>
                    Frontend Execution Log
                  </div>
                  <pre 
                    className="text-xs p-3 rounded-lg overflow-auto max-h-40 border"
                    style={{
                      backgroundColor: isDarkTheme ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.03)',
                      color: 'var(--color-text-muted)',
                      borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)'
                    }}
                  >
                    {result.executionLogs.join('\n')}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ShellCommandResult; 
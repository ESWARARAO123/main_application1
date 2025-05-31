import React, { useState } from 'react';
import { CheckCircleIcon, ExclamationTriangleIcon, ChevronDownIcon, ChevronUpIcon, CommandLineIcon, ServerIcon, ClockIcon, EyeIcon, BugAntIcon } from '@heroicons/react/24/outline';
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
 * Shows clean, formatted output with collapsible technical details
 */
const ShellCommandResult: React.FC<ShellCommandResultProps> = ({ result }) => {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [showExecutionLogs, setShowExecutionLogs] = useState(false);
  const [showRawOutput, setShowRawOutput] = useState(false);
  const [showServerDetails, setShowServerDetails] = useState(false);
  const { currentTheme } = useTheme();
  const isDarkTheme = currentTheme !== 'light';

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
        <pre 
          style={{
            margin: 0,
            padding: '12px',
            fontSize: '13px',
            lineHeight: '1.4',
            borderRadius: '6px',
            background: isDarkTheme ? '#1e1e1e' : '#f8f9fa',
            color: isDarkTheme ? '#d4d4d4' : '#24292e',
            border: `1px solid ${isDarkTheme ? '#333' : '#e1e4e8'}`,
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            maxHeight: '300px'
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

  return (
    <div className="shell-command-result my-4">
      <div className="flex flex-col rounded-lg overflow-hidden" style={{
        backgroundColor: 'var(--color-surface-accent)',
        border: '1px solid var(--color-border-accent)',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)'
      }}>
        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b" style={{
          borderColor: 'var(--color-border-accent)',
          backgroundColor: result.success 
            ? 'rgba(var(--color-success-rgb), 0.1)' 
            : 'rgba(var(--color-error-rgb), 0.1)'
        }}>
          <div className="flex items-center">
            {result.success ? (
              <CheckCircleIcon className="h-5 w-5 mr-2" style={{ color: 'var(--color-success)' }} />
            ) : (
              <ExclamationTriangleIcon className="h-5 w-5 mr-2" style={{ color: 'var(--color-error)' }} />
            )}
            <div>
              <div className="font-medium text-sm" style={{ 
                color: result.success ? 'var(--color-success)' : 'var(--color-error)' 
              }}>
                {result.success ? 'Command Executed Successfully' : 'Command Execution Failed'}
              </div>
              <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                {new Date(result.timestamp).toLocaleString()}
              </div>
            </div>
          </div>
          
          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => setShowRawOutput(!showRawOutput)}
              className="text-xs px-2 py-1 rounded border transition-colors flex items-center"
              style={{
                color: 'var(--color-text-muted)',
                borderColor: 'var(--color-border)',
                backgroundColor: showRawOutput ? 'rgba(var(--color-primary-rgb), 0.1)' : 'transparent'
              }}
            >
              <EyeIcon className="h-3 w-3 mr-1" />
              Raw Output
              {showRawOutput ? (
                <ChevronUpIcon className="h-3 w-3 ml-1" />
              ) : (
                <ChevronDownIcon className="h-3 w-3 ml-1" />
              )}
            </button>
            
            {((result.executionLogs && result.executionLogs.length > 0) || result.serverConfig) && (
              <button
                onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                className="text-xs px-2 py-1 rounded border transition-colors flex items-center"
                style={{
                  color: 'var(--color-text-muted)',
                  borderColor: 'var(--color-border)',
                  backgroundColor: showTechnicalDetails ? 'rgba(var(--color-primary-rgb), 0.1)' : 'transparent'
                }}
              >
                <BugAntIcon className="h-3 w-3 mr-1" />
                Debug Info
                {showTechnicalDetails ? (
                  <ChevronUpIcon className="h-3 w-3 ml-1" />
                ) : (
                  <ChevronDownIcon className="h-3 w-3 ml-1" />
                )}
              </button>
            )}
          </div>
        </div>

        {/* Command and Basic Info */}
        <div className="p-3">
          <div className="flex items-center mb-2">
            <CommandLineIcon className="h-4 w-4 mr-2" style={{ color: 'var(--color-text-muted)' }} />
            <span className="text-sm font-medium" style={{ color: 'var(--color-text)' }}>Command:</span>
          </div>
          <code className="px-2 py-1 rounded text-sm block mb-3" style={{
            backgroundColor: isDarkTheme ? '#2d2d2d' : '#f6f8fa',
            color: 'var(--color-text)',
            border: `1px solid ${isDarkTheme ? '#444' : '#e1e4e8'}`
          }}>
            {result.command}
          </code>

          {result.serverConfig && (
            <div className="flex items-center mb-3">
              <ServerIcon className="h-4 w-4 mr-2" style={{ color: 'var(--color-text-muted)' }} />
              <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Executed on {result.serverConfig.name} ({result.serverConfig.host}:{result.serverConfig.port})
              </span>
              <button
                onClick={() => setShowServerDetails(!showServerDetails)}
                className="ml-2 text-xs text-blue-500 hover:text-blue-600"
              >
                {showServerDetails ? 'Hide' : 'Show'} Details
              </button>
            </div>
          )}

          {/* Server Details (Collapsible) */}
          {showServerDetails && result.serverConfig && (
            <div className="mb-3 p-2 rounded" style={{
              backgroundColor: isDarkTheme ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.5)',
              border: `1px solid ${isDarkTheme ? '#444' : '#e1e4e8'}`
            }}>
              <div className="text-xs space-y-1" style={{ color: 'var(--color-text-muted)' }}>
                <div><strong>Server Name:</strong> {result.serverConfig.name}</div>
                <div><strong>Host:</strong> {result.serverConfig.host}</div>
                <div><strong>Port:</strong> {result.serverConfig.port}</div>
                <div><strong>URL:</strong> http://{result.serverConfig.host}:{result.serverConfig.port}</div>
              </div>
            </div>
          )}
        </div>

        {/* Command Output */}
        {result.success && result.output && result.output.trim() && (
          <div className="px-3 pb-3">
            <div className="text-sm font-medium mb-2" style={{ color: 'var(--color-text)' }}>
              Output:
            </div>
            {formatOutput(result.output)}
          </div>
        )}

        {/* Error Display */}
        {!result.success && (
          <div className="px-3 pb-3">
            {result.error && (
              <div className="mb-3">
                <div className="text-sm font-medium mb-2" style={{ color: 'var(--color-error)' }}>
                  Error:
                </div>
                <div className="p-2 rounded text-sm" style={{
                  backgroundColor: 'rgba(var(--color-error-rgb), 0.1)',
                  color: 'var(--color-error)',
                  border: '1px solid rgba(var(--color-error-rgb), 0.2)'
                }}>
                  {result.error}
                </div>
              </div>
            )}

            {result.stderr && result.stderr.trim() && (
              <div className="mb-3">
                <div className="text-sm font-medium mb-2" style={{ color: 'var(--color-text)' }}>
                  Error Details:
                </div>
                {formatOutput(result.stderr)}
              </div>
            )}

            {result.output && result.output.trim() && (
              <div>
                <div className="text-sm font-medium mb-2" style={{ color: 'var(--color-text)' }}>
                  Available Output:
                </div>
                {formatOutput(result.output)}
              </div>
            )}
          </div>
        )}

        {/* Raw Output (Collapsible) */}
        {showRawOutput && result.output && (
          <div className="border-t" style={{ borderColor: 'var(--color-border-accent)' }}>
            <div className="p-3">
              <div className="text-sm font-medium mb-2" style={{ color: 'var(--color-text)' }}>
                Complete Orchestrator Output
              </div>
              <pre className="text-xs p-2 rounded overflow-auto max-h-40" style={{
                backgroundColor: isDarkTheme ? '#1e1e1e' : '#f8f9fa',
                color: 'var(--color-text-muted)',
                border: `1px solid ${isDarkTheme ? '#333' : '#e1e4e8'}`
              }}>
                {result.output}
              </pre>
            </div>
          </div>
        )}

        {/* Technical Details (Collapsible) */}
        {showTechnicalDetails && (
          <div className="border-t" style={{ borderColor: 'var(--color-border-accent)' }}>
            <div className="p-3">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm font-medium" style={{ color: 'var(--color-text)' }}>
                  Debug Information
                </div>
                {result.executionLogs && result.executionLogs.length > 0 && (
                  <button
                    onClick={() => setShowExecutionLogs(!showExecutionLogs)}
                    className="text-xs text-blue-500 hover:text-blue-600"
                  >
                    {showExecutionLogs ? 'Hide' : 'Show'} Execution Logs
                  </button>
                )}
              </div>

              {/* Parsed execution info */}
              {executionInfo && (
                <div className="space-y-1 text-xs mb-3" style={{ color: 'var(--color-text-muted)' }}>
                  {executionInfo.server && (
                    <div><strong>Server Endpoint:</strong> {executionInfo.server}</div>
                  )}
                  {executionInfo.clientId && (
                    <div><strong>Client ID:</strong> {executionInfo.clientId}</div>
                  )}
                </div>
              )}

              {/* Debug Information for errors */}
              {result.debugInfo && (
                <div className="mb-3 p-2 rounded" style={{
                  backgroundColor: 'rgba(var(--color-error-rgb), 0.1)',
                  border: '1px solid rgba(var(--color-error-rgb), 0.2)'
                }}>
                  <div className="text-xs font-medium mb-1" style={{ color: 'var(--color-error)' }}>
                    HTTP Debug Information:
                  </div>
                  <div className="text-xs space-y-1" style={{ color: 'var(--color-text-muted)' }}>
                    {result.debugInfo.status && (
                      <div><strong>Status:</strong> {result.debugInfo.status} {result.debugInfo.statusText}</div>
                    )}
                    {result.debugInfo.method && result.debugInfo.url && (
                      <div><strong>Request:</strong> {result.debugInfo.method} {result.debugInfo.url}</div>
                    )}
                    {result.debugInfo.status === 404 && (
                      <div style={{ color: 'var(--color-error)', fontWeight: 'bold' }}>
                        ⚠️ API endpoint not found. Check if the server routes are properly configured.
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* MCP Communication Details */}
              {parseRawOutput() && (
                <div className="mb-3">
                  <div className="text-xs font-medium mb-1" style={{ color: 'var(--color-text)' }}>
                    MCP Communication:
                  </div>
                  <pre className="text-xs p-2 rounded overflow-auto max-h-24" style={{
                    backgroundColor: isDarkTheme ? '#1e1e1e' : '#f8f9fa',
                    color: 'var(--color-text-muted)',
                    border: `1px solid ${isDarkTheme ? '#333' : '#e1e4e8'}`
                  }}>
                    {parseRawOutput()}
                  </pre>
                </div>
              )}

              {/* Raw execution logs */}
              {showExecutionLogs && result.executionLogs && result.executionLogs.length > 0 && (
                <div>
                  <div className="text-xs font-medium mb-1" style={{ color: 'var(--color-text)' }}>
                    Frontend Execution Logs:
                  </div>
                  <pre className="text-xs p-2 rounded overflow-auto max-h-32" style={{
                    backgroundColor: isDarkTheme ? '#1e1e1e' : '#f8f9fa',
                    color: 'var(--color-text-muted)',
                    border: `1px solid ${isDarkTheme ? '#333' : '#e1e4e8'}`
                  }}>
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
import React from 'react';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import {
  CheckCircleIcon,
  XCircleIcon,
  XMarkIcon,
  CommandLineIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

interface MCPCommandResultProps {
  resultId: string;
}

const MCPCommandResult: React.FC<MCPCommandResultProps> = ({ resultId }) => {
  const { commandResults, clearCommandResult } = useMCPAgent();

  // Find the result
  const result = commandResults.find(res => res.id === resultId);

  if (!result) {
    return null;
  }

  // Handle clear button click
  const handleClear = () => {
    clearCommandResult(resultId);
  };

  // Format the result content for display
  const formatResultContent = (content: any) => {
    if (typeof content === 'string') {
      return content;
    }

    if (content && content.type === 'text') {
      return content.text;
    }

    if (Array.isArray(content)) {
      return content.map((item, index) => {
        if (typeof item === 'string') {
          return <div key={index}>{item}</div>;
        }

        if (item && item.type === 'text') {
          return <div key={index}>{item.text}</div>;
        }

        return <div key={index}>{JSON.stringify(item)}</div>;
      });
    }

    try {
      return JSON.stringify(content, null, 2);
    } catch (error) {
      return String(content);
    }
  };

  return (
    <div 
      className="rounded-lg border p-4 my-3 transition-all" 
      style={{ 
        backgroundColor: 'var(--color-surface-light)', 
        borderColor: result.success ? 'var(--color-success-light)' : 'var(--color-error-light)',
        borderLeft: `3px solid ${result.success ? 'var(--color-success)' : 'var(--color-error)'}`
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center">
          <CommandLineIcon className="h-5 w-5 mr-2" style={{ 
            color: result.success ? 'var(--color-success)' : 'var(--color-error)' 
          }} />
          <h4 className="font-medium" style={{ color: 'var(--color-text)' }}>
            {result.success ? 'Command Result' : 'Command Error'}
          </h4>
        </div>
        
        <div className="flex items-center">
          <div className="flex items-center mr-3">
            <ClockIcon className="h-4 w-4 mr-1" style={{ color: 'var(--color-text-muted)' }} />
            <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              {new Date(result.timestamp).toLocaleTimeString()}
            </span>
          </div>
          
          <button
            onClick={handleClear}
            className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700"
            title="Clear result"
          >
            <XMarkIcon className="h-4 w-4" style={{ color: 'var(--color-text-muted)' }} />
          </button>
        </div>
      </div>
      
      <div className="mb-2">
        {result.success ? (
          <div 
            className="p-3 rounded font-mono text-sm overflow-auto max-h-60"
            style={{
              backgroundColor: 'var(--color-surface-dark)',
              color: 'var(--color-text)'
            }}
          >
            <pre className="whitespace-pre-wrap break-words"><code>{formatResultContent(result.result)}</code></pre>
          </div>
        ) : (
          <div 
            className="p-3 rounded text-sm flex items-start"
            style={{
              backgroundColor: 'var(--color-error-bg)',
              color: 'var(--color-error)'
            }}
          >
            <XCircleIcon className="h-5 w-5 mr-2 mt-0.5 flex-shrink-0" />
            <pre className="whitespace-pre-wrap break-words"><code>{result.error}</code></pre>
          </div>
        )}
      </div>
      
      <div className="text-xs flex items-center" style={{ color: 'var(--color-text-muted)' }}>
        {result.success ? (
          <div className="flex items-center">
            <CheckCircleIcon className="h-4 w-4 mr-1" style={{ color: 'var(--color-success)' }} />
            <span>Command executed successfully</span>
          </div>
        ) : (
          <div className="flex items-center">
            <XCircleIcon className="h-4 w-4 mr-1" style={{ color: 'var(--color-error)' }} />
            <span>Command execution failed</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default MCPCommandResult; 
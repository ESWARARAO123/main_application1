import React from 'react';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import './MCPAgent.css';

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

    if (Array.isArray(content)) {
      return content.map((item, index) => {
        if (typeof item === 'string') {
          return <div key={index}>{item}</div>;
        }

        if (item.type === 'text') {
          return <div key={index}>{item.text}</div>;
        }

        return <div key={index}>{JSON.stringify(item)}</div>;
      });
    }

    return JSON.stringify(content, null, 2);
  };

  return (
    <div className={`mcp-command-result ${result.success ? 'success' : 'error'}`}>
      <div className="mcp-result-header">
        <h4>{result.success ? 'Command Result' : 'Command Error'}</h4>
        <span className="mcp-result-timestamp">
          {new Date(result.timestamp).toLocaleTimeString()}
        </span>
      </div>

      <div className="mcp-result-content">
        {result.success ? (
          <div className="mcp-result-success">
            <pre>{formatResultContent(result.result)}</pre>
          </div>
        ) : (
          <div className="mcp-result-error">
            <p className="error-message">{result.error}</p>
          </div>
        )}
      </div>

      <div className="mcp-result-actions">
        <button onClick={handleClear} className="mcp-clear-btn">Clear</button>
      </div>
    </div>
  );
};

export default MCPCommandResult;

import React, { useEffect, useState } from 'react';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import { useMCP } from '../../contexts/MCPContext';
import MCPCommandResult from '../mcp/MCPCommandResult';
import axios from 'axios';

// Add interface for AI models
interface AIModel {
  id: string;
  name: string;
  model_id: string;
  ollama_model_id: string;
  is_active: boolean;
  description: string;
}

const MCPAgentCommands: React.FC = () => {
  const { 
    pendingCommands, 
    commandResults, 
    isAgentEnabled,
    isProcessing,
    sseStatus,
    addCommandResult,
    isAnalyzing,
    isStopped
  } = useMCPAgent();
  
  const { 
    availableTools,
    mcpConnection
  } = useMCP();

  // Add state for active models
  const [activeModels, setActiveModels] = useState<AIModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState<boolean>(false);
  const [modelsError, setModelsError] = useState<string | null>(null);

  // Fetch active models when component mounts
  useEffect(() => {
    const fetchActiveModels = async () => {
      try {
        setModelsLoading(true);
        setModelsError(null);
        const response = await axios.get('/api/ollama/models/active');
        if (response.data && response.data.models) {
          setActiveModels(response.data.models);
        }
      } catch (error) {
        console.error('Error fetching active models:', error);
        setModelsError('Failed to fetch active AI models. Some features may be limited.');
      } finally {
        setModelsLoading(false);
      }
    };

    if (isAgentEnabled) {
      fetchActiveModels();
    }
  }, [isAgentEnabled]);

  // Don't render anything if the agent is not enabled
  if (!isAgentEnabled) {
    return null;
  }
  
  // Determine if there's any active processing happening
  const isActive = isProcessing || isAnalyzing;

  return (
    <div className="mcp-agent-commands">
      {/* Model error warning */}
      {modelsError && (
        <div className="mb-4 p-3 rounded-lg border border-warning-300"
          style={{ 
            backgroundColor: 'var(--color-warning-bg)',
            color: 'var(--color-warning)'
          }}>
          <p className="text-sm">{modelsError}</p>
        </div>
      )}

      {/* Show tool results */}
      <div className="command-results">
        {commandResults.map(result => (
          <MCPCommandResult
            key={result.id}
            resultId={result.id}
          />
        ))}
      </div>
    </div>
  );
};

export default MCPAgentCommands;

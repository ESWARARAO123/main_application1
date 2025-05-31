import React, { useState } from 'react';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import './MCPAgent.css';

interface MCPCommandApprovalProps {
  commandId: string;
}

const MCPCommandApproval: React.FC<MCPCommandApprovalProps> = ({ commandId }) => {
  const { pendingCommands, approveCommand, rejectCommand, modifyCommand } = useMCPAgent();
  const [isEditing, setIsEditing] = useState(false);
  const [editedParameters, setEditedParameters] = useState<any>({});

  // Find the command
  const command = pendingCommands.find(cmd => cmd.id === commandId);

  if (!command) {
    return null;
  }

  // Handle approve button click
  const handleApprove = async () => {
    await approveCommand(commandId);
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
    <div className="mcp-command-approval">
      <div className="mcp-command-header">
        <h4>AI Agent Suggested Command</h4>
        <span className="mcp-command-timestamp">
          {new Date(command.timestamp).toLocaleTimeString()}
        </span>
      </div>

      <div className="mcp-command-content">
        <div className="mcp-command-description">
          <p>{command.description}</p>
        </div>

        <div className="mcp-command-details">
          <div className="mcp-command-tool">
            <strong>Tool:</strong> {command.toolName}
          </div>

          {isEditing ? (
            <div className="mcp-command-parameters-edit">
              <h5>Edit Parameters:</h5>
              {Object.entries(editedParameters).map(([key, value]) => (
                <div key={key} className="mcp-parameter-edit">
                  <label htmlFor={`param-${key}`}>{key}:</label>
                  <input
                    id={`param-${key}`}
                    type="text"
                    value={value as string}
                    onChange={(e) => handleParameterChange(key, e.target.value)}
                  />
                </div>
              ))}
              <div className="mcp-edit-actions">
                <button onClick={handleSave} className="mcp-save-btn">Save</button>
                <button onClick={handleCancel} className="mcp-cancel-btn">Cancel</button>
              </div>
            </div>
          ) : (
            <div className="mcp-command-parameters">
              <h5>Parameters:</h5>
              <pre>{JSON.stringify(command.parameters, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>

      {!isEditing && (
        <div className="mcp-command-actions">
          <button onClick={handleApprove} className="mcp-approve-btn">Approve</button>
          <button onClick={handleEdit} className="mcp-edit-btn">Edit</button>
          <button onClick={handleReject} className="mcp-reject-btn">Reject</button>
        </div>
      )}
    </div>
  );
};

export default MCPCommandApproval;

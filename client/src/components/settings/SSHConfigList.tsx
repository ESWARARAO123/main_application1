import React, { useState } from 'react';
import {
  ServerIcon,
  PencilIcon,
  TrashIcon,
  CheckCircleIcon,
  XCircleIcon,
  PlusIcon
} from '@heroicons/react/24/outline';
import SSHConfigForm from './SSHConfigForm';

interface SSHConfiguration {
  id: string;
  machine_nickname: string;
  ssh_host: string;
  ssh_port: number;
  ssh_user: string;
  ssh_auth_method: string;
  last_ssh_connection_status?: string;
  last_ssh_error_message?: string;
}

interface SSHFormData extends SSHConfiguration {
  ssh_password?: string;
  ssh_key_path?: string;
}

interface SSHConfigListProps {
  configurations: SSHConfiguration[];
  onSave: (config: SSHFormData) => Promise<void>;
  onDelete: (configId: string) => Promise<void>;
  onTest?: (config: SSHFormData) => Promise<void>;
}

const SSHConfigList: React.FC<SSHConfigListProps> = ({
  configurations,
  onSave,
  onDelete,
  onTest
}) => {
  const [editingConfig, setEditingConfig] = useState<SSHConfiguration | null>(null);
  const [showAddForm, setShowAddForm] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Handle saving an SSH configuration
  const handleSaveConfig = async (config: SSHConfiguration) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await onSave(config as SSHFormData);
      setEditingConfig(null);
      setShowAddForm(false);
      showSuccess(config.id ? 'Configuration updated successfully' : 'Configuration saved successfully');
    } catch (err: any) {
      console.error('Error saving SSH configuration:', err);
      setError(err.message || 'Failed to save SSH configuration');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle deleting an SSH configuration
  const handleDeleteConfig = async (configId: string) => {
    if (!confirm('Are you sure you want to delete this SSH configuration?')) {
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      await onDelete(configId);
      showSuccess('Configuration deleted successfully');
    } catch (err: any) {
      console.error('Error deleting SSH configuration:', err);
      setError(err.message || 'Failed to delete SSH configuration');
    } finally {
      setIsLoading(false);
    }
  };

  // Show success message with timeout
  const showSuccess = (message: string) => {
    setSuccess(message);
    setError(null);
    setTimeout(() => setSuccess(null), 5000);
  };

  // Show error message with timeout
  const showError = (message: string) => {
    setError(message);
    setSuccess(null);
    setTimeout(() => setError(null), 5000);
  };

  return (
    <div className="space-y-6">
      {/* Success/Error Messages */}
      {success && (
        <div className="mb-4 p-3 bg-green-900/30 border border-green-800 rounded-md text-green-400 flex items-start">
          <CheckCircleIcon className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" />
          <span>{success}</span>
        </div>
      )}
      
      {error && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-800 rounded-md text-red-400 flex items-start">
          <XCircleIcon className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* SSH Configurations List */}
      <div>
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium" style={{ color: 'var(--color-text)' }}>
            SSH Configurations
          </h3>
          
          <button
            onClick={() => setShowAddForm(true)}
            disabled={showAddForm || isLoading}
            className="px-3 py-1.5 rounded transition-colors flex items-center text-sm"
            style={{
              backgroundColor: 'var(--color-primary)',
              color: 'white',
              opacity: (showAddForm || isLoading) ? 0.7 : 1
            }}
          >
            <PlusIcon className="h-4 w-4 mr-1" />
            Add Configuration
          </button>
        </div>
        
        {configurations.length === 0 && !showAddForm && (
          <div className="p-4 rounded-lg text-center" style={{ backgroundColor: 'var(--color-surface-dark)', color: 'var(--color-text-secondary)' }}>
            <p>No SSH configurations found. Add one to get started.</p>
          </div>
        )}
        
        {configurations.length > 0 && (
          <div className="space-y-3">
            {configurations.map(config => (
              editingConfig?.id === config.id ? (
                <SSHConfigForm
                  key={`edit-${config.id}`}
                  existingConfig={config}
                  onSave={handleSaveConfig}
                  onCancel={() => setEditingConfig(null)}
                  onTest={onTest}
                />
              ) : (
                <div 
                  key={config.id}
                  className="p-4 rounded-lg border flex justify-between items-center"
                  style={{
                    backgroundColor: 'var(--color-surface-light)',
                    borderColor: 'var(--color-border)'
                  }}
                >
                  <div className="flex-1">
                    <div className="flex items-center">
                      <ServerIcon className="h-5 w-5 mr-2" style={{ 
                        color: config.last_ssh_connection_status === 'successful' ? 'var(--color-success)' : 
                              config.last_ssh_connection_status === 'failed' ? 'var(--color-error)' : 
                              'var(--color-text-muted)' 
                      }} />
                      <h4 className="font-medium" style={{ color: 'var(--color-text)' }}>
                        {config.machine_nickname}
                      </h4>
                    </div>
                    <p className="text-sm mt-1" style={{ color: 'var(--color-text-secondary)' }}>
                      {config.ssh_user}@{config.ssh_host}:{config.ssh_port} • 
                      <span style={{ 
                        color: config.last_ssh_connection_status === 'successful' ? 'var(--color-success)' : 
                              config.last_ssh_connection_status === 'failed' ? 'var(--color-error)' : 
                              'var(--color-text-muted)' 
                      }}>
                        {" "}{config.last_ssh_connection_status === 'successful' ? 'Connected' : 
                              config.last_ssh_connection_status === 'failed' ? 'Failed' : 
                              'Unknown'}
                      </span>
                    </p>
                    {config.last_ssh_error_message && (
                      <p className="text-xs mt-1" style={{ color: 'var(--color-error)' }}>
                        Error: {config.last_ssh_error_message}
                      </p>
                    )}
                  </div>
                  
                  <div className="flex space-x-2">
                    <button 
                      className="p-2 rounded-full transition-colors"
                      style={{ 
                        backgroundColor: 'var(--color-surface)', 
                        color: 'var(--color-text-secondary)'
                      }}
                      onClick={() => setEditingConfig(config)}
                      title="Edit configuration"
                    >
                      <PencilIcon className="h-4 w-4" />
                    </button>
                    
                    <button 
                      className="p-2 rounded-full transition-colors hover:bg-red-900/20"
                      style={{ 
                        backgroundColor: 'var(--color-surface)', 
                        color: 'var(--color-error)'
                      }}
                      onClick={() => handleDeleteConfig(config.id)}
                      title="Delete configuration"
                    >
                      <TrashIcon className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )
            ))}
          </div>
        )}
        
        {/* Add New Configuration Form */}
        {showAddForm && (
          <div className="mt-4">
            <SSHConfigForm
              onSave={handleSaveConfig}
              onCancel={() => setShowAddForm(false)}
              onTest={onTest}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default SSHConfigList; 
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import SSHConfigList from '../components/settings/SSHConfigList';
import axios from 'axios';
import {
  CheckCircleIcon,
  XCircleIcon,
  InformationCircleIcon,
  ArrowLeftIcon
} from '@heroicons/react/24/outline';
import { api } from '../services/api';

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

const MCPSSHSetup: React.FC = () => {
  const [configurations, setConfigurations] = useState<SSHConfiguration[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchSSHConfigs();
  }, []);

  const fetchSSHConfigs = async () => {
    setLoading(true);
    try {
      const response = await api.get('/mcp/ssh/config');
      setConfigurations(response.data.configurations || []);
      setError(null);
    } catch (err: any) {
      setError('Failed to load SSH configurations');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveConfig = async (config: SSHFormData) => {
    setLoading(true);
    try {
      let response;
      if (config.id) {
        // Update existing config
        response = await api.put(`/mcp/ssh/config/${config.id}`, config);
        setConfigurations(prev => 
          prev.map(c => c.id === config.id ? response.data : c)
        );
      } else {
        // Create new config
        response = await api.post('/mcp/ssh/config', config);
        setConfigurations(prev => [...prev, response.data]);
      }
      
      setSuccess('SSH configuration saved successfully');
      
      // Refresh the list to ensure we have the latest data
      fetchSSHConfigs();
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to save SSH configuration');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteConfig = async (configId: string) => {
    setLoading(true);
    try {
      await api.delete(`/mcp/ssh/config/${configId}`);
      
      // Remove from local state
      setConfigurations(prev => prev.filter(c => c.id !== configId));
      
      setSuccess('SSH configuration deleted successfully');
    } catch (err: any) {
      setError('Failed to delete SSH configuration');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleTestConnection = async (config: SSHFormData): Promise<void> => {
    try {
      const response = await api.post('/mcp/ssh/test', config);
      
      // Update the local state to show connection status
      if (config.id) {
        setConfigurations(prev => 
          prev.map(c => c.id === config.id ? 
            {...c, 
              last_ssh_connection_status: response.data.success ? 'successful' : 'failed',
              last_ssh_error_message: response.data.error || null
            } : c
          )
        );
      }
      
      if (response.data.success) {
        setSuccess('SSH connection test successful');
      } else {
        setError(`SSH connection test failed: ${response.data.error}`);
      }
    } catch (err: any) {
      console.error('Error testing SSH connection:', err);
      setError(err.response?.data?.error || 'Failed to test SSH connection');
    }
  };

  return (
    <div className="py-6 px-4 md:px-8 max-w-[1400px] mx-auto" style={{ backgroundColor: 'var(--color-bg)' }}>
      <div className="flex items-center mb-6">
        <button
          onClick={() => navigate('/settings')}
          className="mr-4 flex items-center text-sm"
          style={{ color: 'var(--color-primary)' }}
        >
          <ArrowLeftIcon className="h-4 w-4 mr-1" />
          Back to Settings
        </button>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--color-text)' }}>
          SSH Setup for MCP Installation
        </h1>
      </div>

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

      <div className="space-y-6">
        <div className="p-4 rounded-lg" style={{ backgroundColor: 'var(--color-surface-dark)', borderColor: 'var(--color-border)' }}>
          <div className="flex items-start">
            <InformationCircleIcon className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" style={{ color: 'var(--color-primary)' }} />
            <p style={{ color: 'var(--color-text-secondary)' }}>
              Configure SSH connections to install MCP on remote machines. These connections will be used to securely install
              and configure the MCP server on your target machines.
            </p>
          </div>
        </div>

        {loading ? (
          <div className="p-4 text-center" style={{ color: 'var(--color-text-muted)' }}>
            Loading SSH configurations...
          </div>
        ) : (
          <SSHConfigList
            configurations={configurations}
            onSave={handleSaveConfig}
            onDelete={handleDeleteConfig}
            onTest={handleTestConnection}
          />
        )}

        {configurations.length > 0 && (
          <div className="mt-8">
            <button
              onClick={() => navigate('/settings/mcp/install-mcp')}
              className="px-4 py-2 rounded"
              style={{
                backgroundColor: 'var(--color-primary)',
                color: 'white'
              }}
            >
              Continue to MCP Installation
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default MCPSSHSetup; 
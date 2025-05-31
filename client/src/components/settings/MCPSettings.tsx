import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  CpuChipIcon,
  ServerIcon,
  ArrowPathIcon,
  PlusIcon,
  TrashIcon,
  ComputerDesktopIcon,
  CheckCircleIcon,
  XCircleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

// Types for MCP integration
interface MCPServer {
  id: string;
  mcp_nickname: string;
  mcp_host: string;
  mcp_port: number;
  mcp_connection_status: string;
  mcp_last_error_message?: string;
  mcp_server_version?: string;
  is_default: boolean;
  ssh_configuration_id?: string;
  tools_count?: number;
}

interface SSHConfiguration {
  id: string;
  machine_nickname: string;
  ssh_host: string;
  ssh_port: number;
  ssh_user: string;
  ssh_auth_method: string;
  last_ssh_connection_status: string;
}

const MCPSettings: React.FC<{ isAdmin: boolean }> = ({ isAdmin }) => {
  // State for MCP servers and SSH configurations
  const [mcpServers, setMcpServers] = useState<MCPServer[]>([]);
  const [sshConfigs, setSshConfigs] = useState<SSHConfiguration[]>([]);
  const [activeView, setActiveView] = useState<'connect' | 'install'>('connect');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const { user } = useAuth();
  const navigate = useNavigate();

  // MCP connection form state
  const [connectionForm, setConnectionForm] = useState({
    mcp_nickname: '',
    mcp_host: 'localhost',
    mcp_port: 3000,
    is_default: false
  });

  // Load MCP servers and SSH configurations on component mount
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Get MCP server configurations
        const [mcpResponse, sshResponse] = await Promise.all([
          axios.get('/api/mcp/server/config'),
          axios.get('/api/mcp/ssh/config')
        ]);

        // Map the server configurations to the expected format
        const servers = mcpResponse.data.configurations || [];
        const mappedServers = servers.map((server: any) => ({
          id: server.id,
          mcp_nickname: server.server_name,
          mcp_host: server.mcp_host,
          mcp_port: server.mcp_port,
          mcp_connection_status: server.last_connection_status || 'unknown',
          mcp_last_error_message: server.last_error_message,
          is_default: server.is_default
        }));

        setMcpServers(mappedServers);
        setSshConfigs(sshResponse.data.configurations || []);
        setError(null);

        // Check connection status for all servers
        setTimeout(() => {
          mappedServers.forEach(server => {
            checkServerConnection(server);
          });
        }, 500); // Small delay to ensure UI is rendered first
      } catch (err: any) {
        console.error('Error loading MCP data:', err);
        setError(err.response?.data?.error || 'Failed to load MCP configuration');

        // For development without backend
        // Remove this when backend is implemented
        setMcpServers([]);
        setSshConfigs([]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Check server connection status without UI updates
  const checkServerConnection = async (server: MCPServer) => {
    try {
      // Call the /info endpoint directly to check if the server is running
      const response = await axios.get(`http://${server.mcp_host}:${server.mcp_port}/info`, {
        timeout: 5000,
        // Use direct httpAgent to avoid proxy issues
        proxy: false
      });

      // If we get a response, the server is running
      if (response.data) {
        // Update the server status in the UI
        setMcpServers(prevServers =>
          prevServers.map(s =>
            s.id === server.id
              ? {
                  ...s,
                  mcp_connection_status: 'connected',
                  mcp_server_version: response.data.version || 'unknown',
                  tools_count: response.data.toolCount || 0
                }
              : s
          )
        );

        // Also update the status in the database
        await axios.post(`/api/mcp/server/config/${server.id}/status`, {
          status: 'connected',
          errorMessage: null
        });
      }
    } catch (err) {
      // Silently fail - we don't want to show errors for automatic checks
      console.log(`Server ${server.mcp_nickname} is not available: ${err}`);
    }
  };

  // Handle connection form changes
  const handleConnectionFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target as HTMLInputElement;
    setConnectionForm({
      ...connectionForm,
      [name]: type === 'checkbox' ? (e.target as HTMLInputElement).checked : type === 'number' ? parseInt(value) : value
    });
  };

  // Test MCP connection
  const handleTestConnection = async () => {
    setLoading(true);
    try {
      // This will be implemented in Phase 2 (Backend)
      const response = await axios.post('/api/mcp/test-connection', {
        mcp_host: connectionForm.mcp_host,
        mcp_port: connectionForm.mcp_port
      });

      if (response.data.success) {
        showSuccess(`Successfully connected to MCP server at ${connectionForm.mcp_host}:${connectionForm.mcp_port}`);
      } else {
        setError(response.data.error || 'Connection test failed');
      }
    } catch (err: any) {
      console.error('Error testing MCP connection:', err);
      setError(err.response?.data?.error || 'Failed to test MCP connection');
    } finally {
      setLoading(false);
    }
  };

  // Save MCP connection
  const handleSaveConnection = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      // Create the server configuration with the correct field names
      const serverConfig = {
        server_name: connectionForm.mcp_nickname,  // Use server_name for the API
        mcp_host: connectionForm.mcp_host,
        mcp_port: connectionForm.mcp_port,
        is_default: connectionForm.is_default
      };

      // Use the correct API endpoint for creating a server configuration
      const response = await axios.post('/api/mcp/server/config', serverConfig);

      // The response should contain the created server configuration
      if (response.data && response.data.id) {
        // Add the new server to the list
        setMcpServers([...mcpServers, {
          id: response.data.id,
          mcp_nickname: response.data.server_name,
          mcp_host: response.data.mcp_host,
          mcp_port: response.data.mcp_port,
          mcp_connection_status: response.data.last_connection_status || 'unknown',
          is_default: response.data.is_default
        }]);

        // Reset the form
        setConnectionForm({
          mcp_nickname: '',
          mcp_host: 'localhost',
          mcp_port: 3000,
          is_default: false
        });

        showSuccess('MCP server connection saved successfully');
      } else {
        setError('Failed to save connection: Invalid response from server');
      }
    } catch (err: any) {
      console.error('Error saving MCP connection:', err);
      setError(err.response?.data?.error || 'Failed to save MCP connection');
    } finally {
      setLoading(false);
    }
  };

  // Check MCP server connection status
  const handleCheckConnection = async (server: MCPServer) => {
    try {
      // Show loading state for this server
      setMcpServers(prevServers =>
        prevServers.map(s =>
          s.id === server.id
            ? { ...s, mcp_connection_status: 'checking' }
            : s
        )
      );

      // Call the /info endpoint directly to check if the server is running
      const response = await axios.get(`http://${server.mcp_host}:${server.mcp_port}/info`, {
        timeout: 5000,
        // Use direct httpAgent to avoid proxy issues
        proxy: false
      });

      // If we get a response, the server is running
      if (response.data) {
        // Update the server status in the UI
        setMcpServers(prevServers =>
          prevServers.map(s =>
            s.id === server.id
              ? {
                  ...s,
                  mcp_connection_status: 'connected',
                  mcp_server_version: response.data.version || 'unknown',
                  tools_count: response.data.toolCount || 0
                }
              : s
          )
        );

        // Also update the status in the database
        await axios.post(`/api/mcp/server/config/${server.id}/status`, {
          status: 'connected',
          errorMessage: null
        });

        showSuccess(`Successfully connected to MCP server at ${server.mcp_host}:${server.mcp_port}`);
      } else {
        throw new Error('Invalid response from MCP server');
      }
    } catch (err: any) {
      console.error(`Error checking MCP connection for ${server.mcp_nickname}:`, err);

      // Update the server status in the UI
      setMcpServers(prevServers =>
        prevServers.map(s =>
          s.id === server.id
            ? {
                ...s,
                mcp_connection_status: 'error',
                mcp_last_error_message: err.message || 'Failed to connect to MCP server'
              }
            : s
        )
      );

      // Also update the status in the database
      try {
        await axios.post(`/api/mcp/server/config/${server.id}/status`, {
          status: 'error',
          errorMessage: err.message || 'Failed to connect to MCP server'
        });
      } catch (updateErr) {
        console.error('Failed to update server status in database:', updateErr);
      }

      setError(`Failed to connect to MCP server at ${server.mcp_host}:${server.mcp_port}: ${err.message || 'Unknown error'}`);
    }
  };

  // Delete MCP server configuration
  const handleDeleteServer = async (serverId: string) => {
    if (!confirm('Are you sure you want to delete this MCP server configuration?')) {
      return;
    }

    setLoading(true);
    try {
      // Use the correct API endpoint for deleting a server configuration
      const response = await axios.delete(`/api/mcp/server/config/${serverId}`);

      // Check if the deletion was successful
      if (response.data.success) {
        // Remove the deleted server from the list
        setMcpServers(mcpServers.filter(server => server.id !== serverId));
        showSuccess('MCP server configuration deleted successfully');
      } else {
        setError(response.data.error || 'Failed to delete server configuration');
      }
    } catch (err: any) {
      console.error('Error deleting MCP server:', err);
      setError(err.response?.data?.error || 'Failed to delete MCP server configuration');
    } finally {
      setLoading(false);
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

      {/* Navigation tabs */}
      <div className="flex border-b border-solid mb-6" style={{ borderColor: 'var(--color-border)' }}>
        <button
          onClick={() => setActiveView('connect')}
          className={`px-4 py-2 font-medium ${activeView === 'connect' ? 'border-b-2 -mb-px' : 'text-opacity-70'}`}
          style={{
            borderColor: activeView === 'connect' ? 'var(--color-primary)' : 'transparent',
            color: 'var(--color-text)'
          }}
        >
          <div className="flex items-center">
            <ServerIcon className="h-5 w-5 mr-2" />
            <span>Connect to MCP</span>
          </div>
        </button>

        <button
          onClick={() => setActiveView('install')}
          className={`px-4 py-2 font-medium ${activeView === 'install' ? 'border-b-2 -mb-px' : 'text-opacity-70'}`}
          style={{
            borderColor: activeView === 'install' ? 'var(--color-primary)' : 'transparent',
            color: 'var(--color-text)'
          }}
        >
          <div className="flex items-center">
            <ComputerDesktopIcon className="h-5 w-5 mr-2" />
            <span>Installation</span>
          </div>
        </button>
      </div>

      {/* Connect to MCP View */}
      {activeView === 'connect' && (
        <div className="space-y-6">
          <div className="p-4 rounded-lg" style={{ backgroundColor: 'var(--color-surface-dark)', borderColor: 'var(--color-border)' }}>
            <div className="flex items-start mb-4">
              <InformationCircleIcon className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" style={{ color: 'var(--color-primary)' }} />
              <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                Connect to a Model Context Protocol (MCP) server to allow the AI to interact with your system.
                MCP allows the AI to execute commands and access files in a controlled and secure manner.
              </p>
            </div>
          </div>

          {/* Existing MCP Servers List */}
          <div>
            <h3 className="text-lg font-medium mb-3" style={{ color: 'var(--color-text)' }}>Your MCP Servers</h3>

            {loading && <p className="text-sm italic" style={{ color: 'var(--color-text-muted)' }}>Loading servers...</p>}

            {!loading && mcpServers.length === 0 && (
              <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                No MCP servers configured. Add a new connection below.
              </p>
            )}

            {!loading && mcpServers.length > 0 && (
              <div className="space-y-3">
                {mcpServers.map(server => (
                  <div
                    key={server.id}
                    className="p-4 rounded-lg border flex justify-between items-center"
                    style={{
                      backgroundColor: 'var(--color-surface-light)',
                      borderColor: server.is_default ? 'var(--color-primary)' : 'var(--color-border)'
                    }}
                  >
                    <div className="flex-1">
                      <div className="flex items-center">
                        <ServerIcon
                          className={`h-5 w-5 mr-2 ${server.mcp_connection_status === 'checking' ? 'animate-pulse' : ''}`}
                          style={{
                            color: server.mcp_connection_status === 'connected'
                              ? 'var(--color-success)'
                              : server.mcp_connection_status === 'checking'
                                ? 'var(--color-warning)'
                                : 'var(--color-error)'
                          }}
                        />
                        <h4 className="font-medium" style={{ color: 'var(--color-text)' }}>{server.mcp_nickname}</h4>
                        {server.is_default && (
                          <span className="ml-2 px-2 py-0.5 text-xs rounded-full" style={{ backgroundColor: 'var(--color-primary-light)', color: 'var(--color-primary)' }}>
                            Default
                          </span>
                        )}
                      </div>
                      <p className="text-sm mt-1" style={{ color: 'var(--color-text-secondary)' }}>
                        {server.mcp_host}:{server.mcp_port} •
                        <span style={{
                          color: server.mcp_connection_status === 'connected'
                            ? 'var(--color-success)'
                            : server.mcp_connection_status === 'checking'
                              ? 'var(--color-warning)'
                              : 'var(--color-error)'
                        }}>
                          {" "}
                          {server.mcp_connection_status === 'checking'
                            ? 'checking...'
                            : server.mcp_connection_status}
                        </span>
                        {server.tools_count && ` • ${server.tools_count} tools available`}
                        {server.mcp_server_version && ` • v${server.mcp_server_version}`}
                      </p>
                    </div>

                    <div className="flex space-x-2">
                      <button
                        className="p-2 rounded-full transition-colors"
                        style={{
                          backgroundColor: 'var(--color-surface)',
                          color: 'var(--color-text-secondary)'
                        }}
                        onClick={() => handleCheckConnection(server)}
                        title="Check connection status"
                      >
                        <ArrowPathIcon className="h-4 w-4" />
                      </button>

                      <button
                        className="p-2 rounded-full transition-colors hover:bg-red-900/20"
                        style={{
                          backgroundColor: 'var(--color-surface)',
                          color: 'var(--color-error)'
                        }}
                        onClick={() => handleDeleteServer(server.id)}
                        title="Delete server"
                      >
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Add New Connection Form */}
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-3" style={{ color: 'var(--color-text)' }}>Add New MCP Connection</h3>

            <form onSubmit={handleSaveConnection} className="space-y-4 p-4 rounded-lg border" style={{ backgroundColor: 'var(--color-surface-light)', borderColor: 'var(--color-border)' }}>
              <div>
                <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
                  Connection Name
                </label>
                <input
                  type="text"
                  name="mcp_nickname"
                  value={connectionForm.mcp_nickname}
                  onChange={handleConnectionFormChange}
                  placeholder="My MCP Server"
                  className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                  style={{
                    backgroundColor: 'var(--color-surface-dark)',
                    borderColor: 'var(--color-border)',
                    color: 'var(--color-text)'
                  }}
                  required
                />
              </div>

              <div className="flex gap-4">
                <div className="flex-1">
                  <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
                    Host / IP Address
                  </label>
                  <input
                    type="text"
                    name="mcp_host"
                    value={connectionForm.mcp_host}
                    onChange={handleConnectionFormChange}
                    placeholder="localhost or IP address"
                    className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                    style={{
                      backgroundColor: 'var(--color-surface-dark)',
                      borderColor: 'var(--color-border)',
                      color: 'var(--color-text)'
                    }}
                    required
                  />
                </div>

                <div className="w-1/4">
                  <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
                    Port
                  </label>
                  <input
                    type="number"
                    name="mcp_port"
                    value={connectionForm.mcp_port}
                    onChange={handleConnectionFormChange}
                    placeholder="3000"
                    className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                    style={{
                      backgroundColor: 'var(--color-surface-dark)',
                      borderColor: 'var(--color-border)',
                      color: 'var(--color-text)'
                    }}
                    required
                  />
                </div>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="is_default"
                  name="is_default"
                  checked={connectionForm.is_default}
                  onChange={handleConnectionFormChange}
                  className="mr-2 rounded"
                  style={{ accentColor: 'var(--color-primary)' }}
                />
                <label htmlFor="is_default" className="text-sm" style={{ color: 'var(--color-text)' }}>
                  Set as default MCP server
                </label>
              </div>

              <div className="flex justify-between pt-2">
                <button
                  type="button"
                  onClick={handleTestConnection}
                  disabled={loading}
                  className="px-4 py-2 rounded transition-colors flex items-center"
                  style={{
                    backgroundColor: 'var(--color-surface)',
                    color: 'var(--color-text)',
                    opacity: loading ? 0.7 : 1
                  }}
                >
                  <ArrowPathIcon className="h-4 w-4 mr-2" />
                  Test Connection
                </button>

                <button
                  type="submit"
                  disabled={loading}
                  className="px-4 py-2 rounded transition-colors flex items-center"
                  style={{
                    backgroundColor: 'var(--color-primary)',
                    color: 'white',
                    opacity: loading ? 0.7 : 1
                  }}
                >
                  <PlusIcon className="h-4 w-4 mr-2" />
                  Save Connection
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Installation View */}
      {activeView === 'install' && (
        <div className="space-y-6">
          <div className="p-4 rounded-lg" style={{ backgroundColor: 'var(--color-surface-dark)', borderColor: 'var(--color-border)' }}>
            <div className="flex items-start mb-4">
              <InformationCircleIcon className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" style={{ color: 'var(--color-primary)' }} />
              <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                MCP Server needs to be installed on the target machine where you want the AI to perform actions.
                You can either install it via SSH (if the target is remote) or manually using the provided instructions.
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-medium" style={{ color: 'var(--color-text)' }}>Installation Options</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Option A: SSH Installation */}
              <div
                className="p-4 rounded-lg border flex flex-col h-full"
                style={{
                  backgroundColor: 'var(--color-surface-light)',
                  borderColor: 'var(--color-border)'
                }}
              >
                <h4 className="font-medium mb-2" style={{ color: 'var(--color-text)' }}>
                  Option A: SSH Installation
                </h4>
                <p className="text-sm mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                  Let the application install MCP server on a remote machine via SSH.
                </p>
                <div className="mt-auto">
                  <button
                    onClick={() => navigate('/settings/mcp/ssh-setup')}
                    className="w-full px-4 py-2 rounded transition-colors"
                    style={{
                      backgroundColor: 'var(--color-primary)',
                      color: 'white'
                    }}
                  >
                    Setup SSH Installation
                  </button>
                </div>
              </div>

              {/* Option B: Manual Installation */}
              <div
                className="p-4 rounded-lg border flex flex-col h-full"
                style={{
                  backgroundColor: 'var(--color-surface-light)',
                  borderColor: 'var(--color-border)'
                }}
              >
                <h4 className="font-medium mb-2" style={{ color: 'var(--color-text)' }}>
                  Option B: Manual Installation
                </h4>
                <p className="text-sm mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                  Follow instructions to manually install the MCP server on your machine.
                </p>
                <div className="mt-auto">
                  <button
                    onClick={() => navigate('/settings/mcp/manual-install')}
                    className="w-full px-4 py-2 rounded transition-colors"
                    style={{
                      backgroundColor: 'var(--color-primary-light)',
                      color: 'var(--color-primary)'
                    }}
                  >
                    View Installation Instructions
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Installation Command (for reference) */}
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-3" style={{ color: 'var(--color-text)' }}>Quick Install Command</h3>
            <div
              className="p-4 rounded-lg border"
              style={{
                backgroundColor: 'var(--color-surface-dark)',
                borderColor: 'var(--color-border)'
              }}
            >
              <p className="text-sm mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                Run this command in your terminal to install MCP server:
              </p>
              <div
                className="p-3 rounded-md font-mono text-sm overflow-x-auto"
                style={{ backgroundColor: 'var(--color-bg-dark)' }}
              >
                <code style={{ color: 'var(--color-text)' }}>
                  wget -qO- https://mcp-server.example.com/install.sh | bash
                </code>
              </div>
              <p className="text-xs mt-2 italic" style={{ color: 'var(--color-text-muted)' }}>
                This will install the MCP server on the local machine. You will need sudo privileges.
              </p>
              <button
                onClick={() => {
                  navigator.clipboard.writeText('wget -qO- https://mcp-server.example.com/install.sh | bash');
                  showSuccess('Command copied to clipboard');
                }}
                className="mt-2 px-3 py-1 rounded text-sm transition-colors"
                style={{
                  backgroundColor: 'var(--color-surface)',
                  color: 'var(--color-text-secondary)'
                }}
              >
                Copy to Clipboard
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MCPSettings;
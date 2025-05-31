import React, { useState } from 'react';
import axios from 'axios';
import {
  ServerIcon,
  KeyIcon,
  ArrowPathIcon,
  PlusIcon,
  CheckCircleIcon,
  XCircleIcon,
  LockClosedIcon
} from '@heroicons/react/24/outline';

interface SSHConfigFormProps {
  existingConfig?: SSHConfiguration;
  onSave: (config: SSHFormData) => void;
  onCancel?: () => void;
  onTest?: (config: SSHFormData) => void;
}

interface SSHConfiguration {
  id?: string;
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
  use_default_port?: boolean;
}

const SSHConfigForm: React.FC<SSHConfigFormProps> = ({
  existingConfig,
  onSave,
  onCancel,
  onTest
}) => {
  // Initialize form state with existing config or defaults
  const [formData, setFormData] = useState<SSHFormData>({
    machine_nickname: existingConfig?.machine_nickname || '',
    ssh_host: existingConfig?.ssh_host || '',
    ssh_port: existingConfig?.ssh_port || 22,
    ssh_user: existingConfig?.ssh_user || '',
    ssh_auth_method: existingConfig?.ssh_auth_method || 'password',
    ssh_password: '',
    ssh_key_path: '',
    use_default_port: false
  });

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'failed'>('idle');

  // Handle form input changes
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type, checked } = e.target as HTMLInputElement;

    if (type === 'checkbox') {
      setFormData({
        ...formData,
        [name]: checked
      });
    } else {
      setFormData({
        ...formData,
        [name]: type === 'number' ? parseInt(value, 10) : value
      });
    }
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // If this is an update to an existing config, include the ID
      const configToSave: SSHFormData = {
        ...formData,
        id: existingConfig?.id
      };

      // Include the necessary auth data based on the selected auth method
      if (configToSave.ssh_auth_method === 'password') {
        // Only include password for password auth
        configToSave.ssh_password = formData.ssh_password;
        // Clear key path if using password auth
        delete configToSave.ssh_key_path;
      } else if (configToSave.ssh_auth_method === 'key') {
        // Only include key path for key auth
        configToSave.ssh_key_path = formData.ssh_key_path;
        // Clear password if using key auth
        delete configToSave.ssh_password;
      }

      // Remove the use_default_port flag before saving
      // as it's only used for the UI and testing
      delete configToSave.use_default_port;

      onSave(configToSave);

      // Reset form if it's a new config
      if (!existingConfig) {
        setFormData({
          machine_nickname: '',
          ssh_host: '',
          ssh_port: 22,
          ssh_user: '',
          ssh_auth_method: 'password',
          ssh_password: '',
          ssh_key_path: ''
        });
      }

      showSuccess('SSH configuration saved successfully');
    } catch (err: any) {
      console.error('Error saving SSH configuration:', err);
      setError(err.message || 'Failed to save SSH configuration');
    } finally {
      setLoading(false);
    }
  };

  // Test SSH connection
  const handleTestConnection = async () => {
    if (!formData.ssh_host || !formData.ssh_user) {
      setError('Host and username are required');
      return;
    }

    if (formData.ssh_auth_method === 'password' && !formData.ssh_password) {
      setError('Password is required');
      return;
    }

    if (formData.ssh_auth_method === 'key' && !formData.ssh_key_path) {
      setError('Key path is required');
      return;
    }

    setTestStatus('testing');
    setError(null);

    try {
      // Prepare the test configuration
      const testConfig: any = {
        ssh_host: formData.ssh_host,
        ssh_user: formData.ssh_user,
        ssh_auth_method: formData.ssh_auth_method,
        ...(formData.ssh_auth_method === 'password'
          ? { ssh_password: formData.ssh_password }
          : { ssh_key_path: formData.ssh_key_path })
      };

      // Only include port if not using default port
      // This is important because we want to completely omit the port parameter
      // rather than setting it to null or undefined
      if (!formData.use_default_port) {
        testConfig.ssh_port = formData.ssh_port;
      }

      if (onTest) {
        onTest(formData);
      } else {
        // Default implementation if onTest isn't provided
        // This would typically be handled by the parent component
        await axios.post('/api/mcp/ssh/test', testConfig);
      }

      setTestStatus('success');
      showSuccess('SSH connection successful');
    } catch (err: any) {
      console.error('Error testing SSH connection:', err);
      setTestStatus('failed');
      setError(err.response?.data?.error || 'Failed to connect via SSH');
    }
  };

  // Show success message with timeout
  const showSuccess = (message: string) => {
    setSuccess(message);
    setError(null);
    setTimeout(() => setSuccess(null), 5000);
  };

  return (
    <div className="rounded-lg border p-5" style={{
      backgroundColor: 'var(--color-surface-light)',
      borderColor: 'var(--color-border)'
    }}>
      <h3 className="text-lg font-medium mb-4" style={{ color: 'var(--color-text)' }}>
        {existingConfig ? 'Edit SSH Configuration' : 'New SSH Configuration'}
      </h3>

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

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Machine Nickname */}
        <div>
          <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
            Machine Nickname
          </label>
          <input
            type="text"
            name="machine_nickname"
            value={formData.machine_nickname}
            onChange={handleChange}
            placeholder="e.g., Development Server"
            className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
            style={{
              backgroundColor: 'var(--color-surface-dark)',
              borderColor: 'var(--color-border)',
              color: 'var(--color-text)'
            }}
            required
          />
        </div>

        {/* Host and Port */}
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2">
            <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
              SSH Host / IP Address
            </label>
            <input
              type="text"
              name="ssh_host"
              value={formData.ssh_host}
              onChange={handleChange}
              placeholder="e.g., 192.168.1.100"
              className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
              style={{
                backgroundColor: 'var(--color-surface-dark)',
                borderColor: 'var(--color-border)',
                color: 'var(--color-text)'
              }}
              required
            />
          </div>

          <div>
            <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
              SSH Port
            </label>
            <div className="space-y-2">
              <input
                type="number"
                name="ssh_port"
                value={formData.ssh_port}
                onChange={handleChange}
                placeholder="22"
                className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                style={{
                  backgroundColor: 'var(--color-surface-dark)',
                  borderColor: 'var(--color-border)',
                  color: 'var(--color-text)'
                }}
                required
              />
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="use_default_port"
                  name="use_default_port"
                  checked={formData.use_default_port}
                  onChange={handleChange}
                  className="mr-2"
                />
                <label htmlFor="use_default_port" className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                  Omit port when testing connection (use default SSH port 22)
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* Username */}
        <div>
          <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
            SSH Username
          </label>
          <input
            type="text"
            name="ssh_user"
            value={formData.ssh_user}
            onChange={handleChange}
            placeholder="e.g., admin"
            className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
            style={{
              backgroundColor: 'var(--color-surface-dark)',
              borderColor: 'var(--color-border)',
              color: 'var(--color-text)'
            }}
            required
          />
        </div>

        {/* Authentication Method */}
        <div>
          <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
            Authentication Method
          </label>
          <select
            name="ssh_auth_method"
            value={formData.ssh_auth_method}
            onChange={handleChange}
            className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
            style={{
              backgroundColor: 'var(--color-surface-dark)',
              borderColor: 'var(--color-border)',
              color: 'var(--color-text)'
            }}
          >
            <option value="password">Password</option>
            <option value="key">Private Key</option>
          </select>
        </div>

        {/* Auth Details - Password */}
        {formData.ssh_auth_method === 'password' && (
          <div>
            <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
              SSH Password
            </label>
            <div className="relative">
              <input
                type="password"
                name="ssh_password"
                value={formData.ssh_password}
                onChange={handleChange}
                placeholder="Enter your SSH password"
                className="w-full rounded pl-9 pr-3 py-2 focus:outline-none focus:ring-1"
                style={{
                  backgroundColor: 'var(--color-surface-dark)',
                  borderColor: 'var(--color-border)',
                  color: 'var(--color-text)'
                }}
              />
              <LockClosedIcon className="h-5 w-5 absolute left-2 top-2" style={{ color: 'var(--color-text-muted)' }} />
            </div>
            <p className="text-xs mt-1 italic" style={{ color: 'var(--color-text-muted)' }}>
              Your password will be encrypted and stored securely.
            </p>
          </div>
        )}

        {/* Auth Details - Private Key */}
        {formData.ssh_auth_method === 'key' && (
          <div>
            <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
              Private Key Path (on the backend server)
            </label>
            <div className="relative">
              <input
                type="text"
                name="ssh_key_path"
                value={formData.ssh_key_path}
                onChange={handleChange}
                placeholder="/path/to/private_key"
                className="w-full rounded pl-9 pr-3 py-2 focus:outline-none focus:ring-1"
                style={{
                  backgroundColor: 'var(--color-surface-dark)',
                  borderColor: 'var(--color-border)',
                  color: 'var(--color-text)'
                }}
              />
              <KeyIcon className="h-5 w-5 absolute left-2 top-2" style={{ color: 'var(--color-text-muted)' }} />
            </div>
            <p className="text-xs mt-1 italic" style={{ color: 'var(--color-text-muted)' }}>
              Enter the path to your private key file on the server where this application is running.
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-between pt-2">
          <div className="flex space-x-3">
            <button
              type="button"
              onClick={handleTestConnection}
              disabled={loading || testStatus === 'testing'}
              className="px-4 py-2 rounded transition-colors flex items-center"
              style={{
                backgroundColor: testStatus === 'success' ? 'var(--color-success-bg)' :
                                testStatus === 'failed' ? 'var(--color-error-bg)' :
                                'var(--color-surface)',
                color: testStatus === 'success' ? 'var(--color-success)' :
                       testStatus === 'failed' ? 'var(--color-error)' :
                       'var(--color-text)',
                opacity: (loading || testStatus === 'testing') ? 0.7 : 1
              }}
            >
              {testStatus === 'testing' ? (
                <span className="flex items-center">
                  <ArrowPathIcon className="h-4 w-4 mr-2 animate-spin" />
                  Testing...
                </span>
              ) : testStatus === 'success' ? (
                <span className="flex items-center">
                  <CheckCircleIcon className="h-4 w-4 mr-2" />
                  Connected
                </span>
              ) : testStatus === 'failed' ? (
                <span className="flex items-center">
                  <XCircleIcon className="h-4 w-4 mr-2" />
                  Failed
                </span>
              ) : (
                <span className="flex items-center">
                  <ServerIcon className="h-4 w-4 mr-2" />
                  Test Connection
                </span>
              )}
            </button>

            {onCancel && (
              <button
                type="button"
                onClick={onCancel}
                disabled={loading}
                className="px-4 py-2 rounded transition-colors"
                style={{
                  backgroundColor: 'var(--color-surface)',
                  color: 'var(--color-text)',
                  opacity: loading ? 0.7 : 1
                }}
              >
                Cancel
              </button>
            )}
          </div>

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
            {existingConfig ? 'Update' : 'Save Configuration'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default SSHConfigForm;
import React, { useState } from 'react';
import {
  LockClosedIcon,
  ExclamationCircleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';

interface SSHPasswordPromptProps {
  sshConfigId: string;
  sshConfigName: string;
  onAuthenticate: (password: string) => Promise<boolean>;
  onCancel: () => void;
  isConfirmation?: boolean;
}

const SSHPasswordPrompt: React.FC<SSHPasswordPromptProps> = ({
  sshConfigId,
  sshConfigName,
  onAuthenticate,
  onCancel,
  isConfirmation = false
}) => {
  const [password, setPassword] = useState('');
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!password) {
      setError('Password is required');
      return;
    }

    setIsAuthenticating(true);
    setError(null);
    
    try {
      const result = await onAuthenticate(password);
      
      if (result) {
        setSuccess(true);
        // Success is handled by the parent component
      } else {
        setError('Authentication failed. Please check your password and try again.');
      }
    } catch (err: any) {
      setError(err.message || 'Authentication failed. Please try again.');
    } finally {
      setIsAuthenticating(false);
    }
  };

  return (
    <div className="p-6 rounded-lg shadow-lg max-w-md mx-auto" style={{ backgroundColor: 'var(--color-surface-light)', borderColor: 'var(--color-border)' }}>
      <div className="flex items-center mb-4">
        <LockClosedIcon className="h-6 w-6 mr-2" style={{ color: 'var(--color-primary)' }} />
        <h2 className="text-xl font-medium" style={{ color: 'var(--color-text)' }}>
          {isConfirmation ? 'Confirm SSH Password' : 'SSH Authentication'}
        </h2>
      </div>
      
      <p className="mb-4" style={{ color: 'var(--color-text-secondary)' }}>
        {isConfirmation 
          ? `Please confirm your SSH password for ${sshConfigName} to proceed with the installation.`
          : `Enter your SSH password for ${sshConfigName} to connect and browse directories.`}
      </p>
      
      {error && (
        <div className="mb-4 p-3 rounded-md flex items-start" style={{ backgroundColor: 'rgba(var(--color-error-rgb), 0.1)', borderColor: 'var(--color-error)' }}>
          <ExclamationCircleIcon className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" style={{ color: 'var(--color-error)' }} />
          <span style={{ color: 'var(--color-error)' }}>{error}</span>
        </div>
      )}
      
      {success && (
        <div className="mb-4 p-3 rounded-md flex items-start" style={{ backgroundColor: 'rgba(var(--color-success-rgb), 0.1)', borderColor: 'var(--color-success)' }}>
          <CheckCircleIcon className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" style={{ color: 'var(--color-success)' }} />
          <span style={{ color: 'var(--color-success)' }}>Authentication successful!</span>
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="ssh-password" className="block mb-2 text-sm font-medium" style={{ color: 'var(--color-text)' }}>
            Password
          </label>
          <input
            type="password"
            id="ssh-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full p-2 rounded-md border"
            style={{ 
              backgroundColor: 'var(--color-surface-dark)', 
              borderColor: error ? 'var(--color-error)' : 'var(--color-border)',
              color: 'var(--color-text)'
            }}
            placeholder="Enter SSH password"
            autoComplete="off"
            disabled={isAuthenticating || success}
          />
        </div>
        
        <div className="flex justify-between">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 rounded"
            style={{
              backgroundColor: 'var(--color-surface-dark)',
              color: 'var(--color-text)',
              cursor: isAuthenticating ? 'not-allowed' : 'pointer'
            }}
            disabled={isAuthenticating}
          >
            Cancel
          </button>
          
          <button
            type="submit"
            className="px-4 py-2 rounded"
            style={{
              backgroundColor: isAuthenticating ? 'var(--color-primary-muted)' : 'var(--color-primary)',
              color: 'white',
              cursor: isAuthenticating ? 'not-allowed' : 'pointer'
            }}
            disabled={isAuthenticating || success}
          >
            {isAuthenticating ? 'Authenticating...' : isConfirmation ? 'Confirm & Install' : 'Connect'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default SSHPasswordPrompt;

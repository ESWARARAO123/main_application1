import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeftIcon,
  InformationCircleIcon,
  DocumentDuplicateIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';

const MCPManualInstall: React.FC = () => {
  const navigate = useNavigate();
  const [copied, setCopied] = useState<boolean>(false);
  const [installationCommand, setInstallationCommand] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch the installation command from the backend config API
    const fetchInstallationCommand = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/mcp/config');
        
        if (!response.ok) {
          throw new Error('Failed to fetch MCP configuration');
        }
        
        const data = await response.json();
        if (data && data.defaultTool && data.defaultTool.installCommand) {
          setInstallationCommand(data.defaultTool.installCommand);
        } else {
          setError('Installation command not found in configuration');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
        console.error('Error fetching installation command:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchInstallationCommand();
  }, []);

  const handleCopyCommand = () => {
    navigator.clipboard.writeText(installationCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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
          Manual MCP Installation
        </h1>
      </div>

      <div className="space-y-6">
        <div className="p-4 rounded-lg" style={{ backgroundColor: 'var(--color-surface-dark)', borderColor: 'var(--color-border)' }}>
          <div className="flex items-start">
            <InformationCircleIcon className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" style={{ color: 'var(--color-primary)' }} />
            <p style={{ color: 'var(--color-text-secondary)' }}>
              Follow these instructions to manually install the MCP server on your machine.
              The Model Context Protocol (MCP) server allows AI to interact with your system in a controlled manner.
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <h2 className="text-xl font-medium" style={{ color: 'var(--color-text)' }}>Installation Steps</h2>
          
          <div className="p-4 rounded-lg border" style={{ backgroundColor: 'var(--color-surface-light)', borderColor: 'var(--color-border)' }}>
            <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--color-text)' }}>
              Step 1: Install Prerequisites
            </h3>
            <p className="mb-2" style={{ color: 'var(--color-text-secondary)' }}>
              Ensure your system has the following prerequisites:
            </p>
            <ul className="list-disc pl-5 mb-3" style={{ color: 'var(--color-text-secondary)' }}>
              <li>Linux-based operating system (Ubuntu/Debian recommended)</li>
              <li>bash shell</li>
              <li>curl or wget</li>
              <li>Administrative (sudo) access</li>
            </ul>
          </div>
          
          <div className="p-4 rounded-lg border" style={{ backgroundColor: 'var(--color-surface-light)', borderColor: 'var(--color-border)' }}>
            <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--color-text)' }}>
              Step 2: Run Installation Command
            </h3>
            <p className="mb-3" style={{ color: 'var(--color-text-secondary)' }}>
              Open a terminal on your target machine and run the following command:
            </p>
            <div className="relative">
              {loading ? (
                <div className="p-3 rounded-md font-mono text-sm mb-3 overflow-x-auto"
                     style={{ backgroundColor: 'var(--color-bg-dark)' }}>
                  <div style={{ color: 'var(--color-text-muted)' }}>Loading installation command...</div>
                </div>
              ) : error ? (
                <div className="p-3 rounded-md font-mono text-sm mb-3 overflow-x-auto"
                     style={{ backgroundColor: 'var(--color-bg-dark)', color: 'var(--color-error)' }}>
                  Error: {error}
                </div>
              ) : (
                <div className="p-3 rounded-md font-mono text-sm mb-3 overflow-x-auto"
                     style={{ backgroundColor: 'var(--color-bg-dark)' }}>
                  <code style={{ color: 'var(--color-text)' }}>
                    {installationCommand}
                  </code>
                </div>
              )}
              {!loading && !error && (
                <button
                  onClick={handleCopyCommand}
                  className="absolute top-2 right-2 p-1.5 rounded hover:bg-opacity-80 flex items-center"
                  style={{ backgroundColor: 'var(--color-surface)' }}
                  title="Copy to clipboard"
                >
                  {copied ? (
                    <CheckCircleIcon className="h-4 w-4" style={{ color: 'var(--color-success)' }} />
                  ) : (
                    <DocumentDuplicateIcon className="h-4 w-4" style={{ color: 'var(--color-text-muted)' }} />
                  )}
                </button>
              )}
            </div>
            <p className="text-xs italic" style={{ color: 'var(--color-text-muted)' }}>
              This command will download and run the installation script with root privileges.
            </p>
          </div>
          
          <div className="p-4 rounded-lg border" style={{ backgroundColor: 'var(--color-surface-light)', borderColor: 'var(--color-border)' }}>
            <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--color-text)' }}>
              Step 3: Configure and Connect
            </h3>
            <p className="mb-2" style={{ color: 'var(--color-text-secondary)' }}>
              After installation is complete:
            </p>
            <ol className="list-decimal pl-5 space-y-2" style={{ color: 'var(--color-text-secondary)' }}>
              <li>Note the URL and port where the MCP server is running (default: <code>http://localhost:8080</code>)</li>
              <li>Return to the Connect tab in the MCP settings</li>
              <li>Add a new connection using the hostname/IP and port from the installation</li>
            </ol>
          </div>
          
          <div className="p-4 rounded-lg border" style={{ backgroundColor: 'var(--color-surface-light)', borderColor: 'var(--color-border)' }}>
            <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--color-text)' }}>
              Troubleshooting
            </h3>
            <p className="mb-2" style={{ color: 'var(--color-text-secondary)' }}>
              If you encounter issues during installation:
            </p>
            <ul className="list-disc pl-5" style={{ color: 'var(--color-text-secondary)' }}>
              <li>Check the installation logs in <code>/var/log/mcp-install.log</code></li>
              <li>Ensure your system has internet access</li>
              <li>Verify that you have sufficient permissions</li>
              <li>Check that ports 8080 is not already in use by another application</li>
            </ul>
          </div>
        </div>

        <div className="mt-6">
          <button
            onClick={() => navigate('/settings')}
            className="px-4 py-2 rounded"
            style={{
              backgroundColor: 'var(--color-primary)',
              color: 'white'
            }}
          >
            Return to Settings
          </button>
        </div>
      </div>
    </div>
  );
};

export default MCPManualInstall; 
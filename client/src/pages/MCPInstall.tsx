import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeftIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  XCircleIcon,
  ServerIcon,
  FolderIcon,
  ArrowRightIcon
} from '@heroicons/react/24/outline';
import { api } from '../services/api';
import SSHPasswordPrompt from '../components/ssh/SSHPasswordPrompt';
import RemoteFilesystemExplorer from '../components/ssh/RemoteFilesystemExplorer';

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

interface MCPTool {
  id: string;
  name: string;
  installCommand: string;
  description?: string;
  isDefault?: boolean;
}

// Define installation steps
enum InstallStep {
  SELECT_CONFIG = 0,
  AUTHENTICATE = 1,
  SELECT_DIRECTORY = 2,
  CONFIRM_INSTALL = 3,
  INSTALLING = 4,
  COMPLETE = 5
}

const MCPInstall: React.FC = () => {
  const navigate = useNavigate();
  const [sshConfigurations, setSSHConfigurations] = useState<SSHConfiguration[]>([]);
  const [mcpTools, setMCPTools] = useState<MCPTool[]>([]);
  const [selectedSSH, setSelectedSSH] = useState<string>('');
  const [selectedMCP, setSelectedMCP] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [installing, setInstalling] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [installationOutput, setInstallationOutput] = useState<string>('');

  // New state for multi-step installation
  const [currentStep, setCurrentStep] = useState<InstallStep>(InstallStep.SELECT_CONFIG);
  const [sshPassword, setSSHPassword] = useState<string>('');
  const [selectedDirectory, setSelectedDirectory] = useState<string>('');
  const [selectedSSHConfig, setSelectedSSHConfig] = useState<SSHConfiguration | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      // Fetch SSH configurations
      const sshResponse = await api.get('/mcp/ssh/config');
      const configurations = sshResponse.data.configurations || [];
      setSSHConfigurations(configurations);

      // Fetch MCP tools from config
      const mcpResponse = await api.get('/mcp/config');

      // Create a default MCP tool from config
      const defaultTool = mcpResponse.data.defaultTool;
      if (defaultTool) {
        setMCPTools([{
          id: 'default',
          name: defaultTool.name || 'Default MCP Tool',
          installCommand: defaultTool.installCommand || '',
          description: 'Default MCP installation tool from configuration',
          isDefault: true
        }]);

        // Set as selected by default
        setSelectedMCP('default');
      }

      setError(null);
    } catch (err: any) {
      setError('Failed to load configurations');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Update selected SSH config when selectedSSH changes
  useEffect(() => {
    if (selectedSSH && sshConfigurations.length > 0) {
      const config = sshConfigurations.find(config => config.id === selectedSSH);
      setSelectedSSHConfig(config || null);
    } else {
      setSelectedSSHConfig(null);
    }
  }, [selectedSSH, sshConfigurations]);

  // Handle proceeding to next step
  const handleNextStep = () => {
    setError(null);

    // Validate current step
    if (currentStep === InstallStep.SELECT_CONFIG) {
      if (!selectedSSH || !selectedMCP) {
        setError('Please select both an SSH configuration and an MCP tool');
        return;
      }
      setCurrentStep(InstallStep.AUTHENTICATE);
    } else if (currentStep === InstallStep.SELECT_DIRECTORY) {
      if (!selectedDirectory) {
        setError('Please select an installation directory');
        return;
      }
      setCurrentStep(InstallStep.CONFIRM_INSTALL);
    }
  };

  // Handle going back to previous step
  const handlePreviousStep = () => {
    setError(null);

    if (currentStep === InstallStep.AUTHENTICATE) {
      setCurrentStep(InstallStep.SELECT_CONFIG);
    } else if (currentStep === InstallStep.SELECT_DIRECTORY) {
      setCurrentStep(InstallStep.AUTHENTICATE);
    } else if (currentStep === InstallStep.CONFIRM_INSTALL) {
      setCurrentStep(InstallStep.SELECT_DIRECTORY);
    }
  };

  // Handle SSH authentication
  const handleAuthenticate = async (password: string): Promise<boolean> => {
    if (!selectedSSH) return false;

    try {
      const response = await api.post('/mcp/ssh/authenticate', {
        sshConfigId: selectedSSH,
        password
      });

      if (response.data.success) {
        setSSHPassword(password);
        setCurrentStep(InstallStep.SELECT_DIRECTORY);
        return true;
      } else {
        setError(`Authentication failed: ${response.data.error}`);
        return false;
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.error || 'Authentication failed';
      setError(errorMessage);
      return false;
    }
  };

  // Handle directory selection
  const handleDirectorySelect = (path: string) => {
    setSelectedDirectory(path);
    setCurrentStep(InstallStep.CONFIRM_INSTALL);
  };

  // Handle final installation
  const handleConfirmInstall = async (password: string): Promise<boolean> => {
    if (!selectedSSH || !selectedMCP || !selectedDirectory) return false;

    setCurrentStep(InstallStep.INSTALLING);
    setInstalling(true);
    setError(null);
    setSuccess(null);
    setInstallationOutput('Starting MCP installation...\n');

    try {
      const response = await api.post('/mcp/ssh/install-in-directory', {
        sshConfigId: selectedSSH,
        mcpToolId: selectedMCP,
        installDir: selectedDirectory,
        password
      });

      if (response.data.success) {
        // Check if there's a warning message
        if (response.data.warning) {
          setSuccess('MCP server installed with warnings');
          setInstallationOutput(prev => prev +
            '\nInstallation completed in ' + selectedDirectory +
            ', but with warnings:\n' + response.data.warning + '\n');
        } else {
          setSuccess('MCP server installed successfully');
          setInstallationOutput(prev => prev +
            '\nInstallation completed successfully in ' + selectedDirectory + '!\n' +
            'MCP server is running on port ' + response.data.port + '\n');
        }
        setCurrentStep(InstallStep.COMPLETE);
        return true;
      } else {
        setError(`Installation failed: ${response.data.error}`);
        setInstallationOutput(prev => prev + `\nInstallation failed: ${response.data.error}\n`);
        return false;
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.error || 'Failed to install MCP server';
      setError(errorMessage);
      setInstallationOutput(prev => prev + `\nError: ${errorMessage}\n`);
      return false;
    } finally {
      setInstalling(false);
    }
  };

  // Legacy installation handler (kept for reference)
  const handleInstallMCP = async () => {
    if (!selectedSSH || !selectedMCP) {
      setError('Please select both an SSH configuration and an MCP tool');
      return;
    }

    // Start the multi-step installation process
    setCurrentStep(InstallStep.AUTHENTICATE);
  };

  // Render step progress indicator
  const renderStepIndicator = () => {
    const steps = [
      { name: 'Select Configuration', step: InstallStep.SELECT_CONFIG },
      { name: 'Authenticate', step: InstallStep.AUTHENTICATE },
      { name: 'Select Directory', step: InstallStep.SELECT_DIRECTORY },
      { name: 'Confirm & Install', step: InstallStep.CONFIRM_INSTALL },
      { name: 'Complete', step: InstallStep.COMPLETE }
    ];

    return (
      <div className="flex items-center justify-between mb-6 px-2">
        {steps.map((step, index) => (
          <React.Fragment key={step.step}>
            {/* Step circle */}
            <div className="flex flex-col items-center">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  currentStep >= step.step ? 'bg-blue-500' : 'bg-gray-700'
                }`}
              >
                {currentStep > step.step ? (
                  <CheckCircleIcon className="h-5 w-5 text-white" />
                ) : (
                  <span className="text-white text-sm">{index + 1}</span>
                )}
              </div>
              <span
                className="text-xs mt-1 text-center"
                style={{
                  color: currentStep >= step.step ? 'var(--color-text)' : 'var(--color-text-muted)',
                  maxWidth: '80px'
                }}
              >
                {step.name}
              </span>
            </div>

            {/* Connector line (except after last step) */}
            {index < steps.length - 1 && (
              <div
                className="flex-1 h-0.5 mx-2"
                style={{
                  backgroundColor: currentStep > step.step ? 'var(--color-primary)' : 'var(--color-border)'
                }}
              />
            )}
          </React.Fragment>
        ))}
      </div>
    );
  };

  // Render content based on current step
  const renderStepContent = () => {
    switch (currentStep) {
      case InstallStep.SELECT_CONFIG:
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* SSH Configuration Selection */}
            <div className="p-4 rounded-lg border" style={{ backgroundColor: 'var(--color-surface-light)', borderColor: 'var(--color-border)' }}>
              <h2 className="text-lg font-medium mb-4" style={{ color: 'var(--color-text)' }}>
                1. Select SSH Configuration
              </h2>

              {sshConfigurations.length === 0 ? (
                <div className="p-3 rounded-md" style={{ backgroundColor: 'var(--color-surface-dark)', color: 'var(--color-text-muted)' }}>
                  No SSH configurations found. Please add one first.
                </div>
              ) : (
                <div className="space-y-3">
                  {sshConfigurations.map(config => (
                    <div
                      key={config.id}
                      className={`p-3 rounded-md cursor-pointer border transition-colors ${
                        selectedSSH === config.id ? 'border-blue-500' : 'border-transparent'
                      }`}
                      style={{
                        backgroundColor: selectedSSH === config.id ? 'var(--color-primary-light)' : 'var(--color-surface-dark)',
                        color: 'var(--color-text)'
                      }}
                      onClick={() => setSelectedSSH(config.id)}
                    >
                      <div className="flex items-center">
                        <ServerIcon className="h-5 w-5 mr-2" style={{ color: 'var(--color-primary)' }} />
                        <div>
                          <div className="font-medium">{config.machine_nickname}</div>
                          <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                            {config.ssh_user}@{config.ssh_host}:{config.ssh_port}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* MCP Tool Selection */}
            <div className="p-4 rounded-lg border" style={{ backgroundColor: 'var(--color-surface-light)', borderColor: 'var(--color-border)' }}>
              <h2 className="text-lg font-medium mb-4" style={{ color: 'var(--color-text)' }}>
                2. Select MCP Installation Tool
              </h2>

              {mcpTools.length === 0 ? (
                <div className="p-3 rounded-md" style={{ backgroundColor: 'var(--color-surface-dark)', color: 'var(--color-text-muted)' }}>
                  No MCP tools found. Please check your configuration.
                </div>
              ) : (
                <div className="space-y-3">
                  {mcpTools.map(tool => (
                    <div
                      key={tool.id}
                      className={`p-3 rounded-md cursor-pointer border transition-colors ${
                        selectedMCP === tool.id ? 'border-blue-500' : 'border-transparent'
                      }`}
                      style={{
                        backgroundColor: selectedMCP === tool.id ? 'var(--color-primary-light)' : 'var(--color-surface-dark)',
                        color: 'var(--color-text)'
                      }}
                      onClick={() => setSelectedMCP(tool.id)}
                    >
                      <div className="font-medium">{tool.name}</div>
                      <div className="text-sm mt-1" style={{ color: 'var(--color-text-secondary)' }}>
                        {tool.description}
                      </div>
                      <div className="text-xs mt-2 font-mono p-2 rounded" style={{ backgroundColor: 'var(--color-bg-dark)' }}>
                        {tool.installCommand}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        );

      case InstallStep.AUTHENTICATE:
        return (
          <div className="flex justify-center">
            {selectedSSHConfig && (
              <SSHPasswordPrompt
                sshConfigId={selectedSSHConfig.id}
                sshConfigName={selectedSSHConfig.machine_nickname}
                onAuthenticate={handleAuthenticate}
                onCancel={handlePreviousStep}
              />
            )}
          </div>
        );

      case InstallStep.SELECT_DIRECTORY:
        return (
          <div>
            {selectedSSHConfig && sshPassword && (
              <RemoteFilesystemExplorer
                sshConfigId={selectedSSHConfig.id}
                sshPassword={sshPassword}
                initialPath="/home"
                onPathSelect={handleDirectorySelect}
                onCancel={handlePreviousStep}
              />
            )}
          </div>
        );

      case InstallStep.CONFIRM_INSTALL:
        return (
          <div className="p-6 rounded-lg shadow-lg max-w-md mx-auto" style={{ backgroundColor: 'var(--color-surface-light)', borderColor: 'var(--color-border)' }}>
            <div className="flex items-center mb-4">
              <FolderIcon className="h-6 w-6 mr-2" style={{ color: 'var(--color-primary)' }} />
              <h2 className="text-xl font-medium" style={{ color: 'var(--color-text)' }}>
                Confirm Installation
              </h2>
            </div>

            <div className="mb-4">
              <h3 className="font-medium mb-1" style={{ color: 'var(--color-text)' }}>Installation Details</h3>
              <div className="p-3 rounded-md" style={{ backgroundColor: 'var(--color-surface-dark)' }}>
                <div className="mb-2">
                  <span className="text-sm font-medium" style={{ color: 'var(--color-text-muted)' }}>SSH Configuration:</span>
                  <div style={{ color: 'var(--color-text)' }}>{selectedSSHConfig?.machine_nickname}</div>
                </div>
                <div className="mb-2">
                  <span className="text-sm font-medium" style={{ color: 'var(--color-text-muted)' }}>Installation Directory:</span>
                  <div className="font-mono text-sm" style={{ color: 'var(--color-text)' }}>{selectedDirectory}</div>
                </div>
                <div>
                  <span className="text-sm font-medium" style={{ color: 'var(--color-text-muted)' }}>Installation Command:</span>
                  <div className="font-mono text-sm" style={{ color: 'var(--color-text)' }}>
                    {mcpTools.find(tool => tool.id === selectedMCP)?.installCommand}
                  </div>
                </div>
              </div>
            </div>

            <p className="mb-4" style={{ color: 'var(--color-text-secondary)' }}>
              Please confirm your SSH password to proceed with the installation in the selected directory.
            </p>

            <SSHPasswordPrompt
              sshConfigId={selectedSSH}
              sshConfigName={selectedSSHConfig?.machine_nickname || ''}
              onAuthenticate={handleConfirmInstall}
              onCancel={handlePreviousStep}
              isConfirmation={true}
            />
          </div>
        );

      case InstallStep.INSTALLING:
      case InstallStep.COMPLETE:
        return (
          <div className="p-4 rounded-lg border" style={{ backgroundColor: 'var(--color-surface-light)', borderColor: 'var(--color-border)' }}>
            <h2 className="text-lg font-medium mb-2" style={{ color: 'var(--color-text)' }}>
              Installation {currentStep === InstallStep.COMPLETE ? 'Complete' : 'Progress'}
            </h2>

            <div className="mb-4">
              <div className="p-3 rounded-md" style={{ backgroundColor: 'var(--color-surface-dark)' }}>
                <div className="mb-2">
                  <span className="text-sm font-medium" style={{ color: 'var(--color-text-muted)' }}>SSH Configuration:</span>
                  <div style={{ color: 'var(--color-text)' }}>{selectedSSHConfig?.machine_nickname}</div>
                </div>
                <div className="mb-2">
                  <span className="text-sm font-medium" style={{ color: 'var(--color-text-muted)' }}>Installation Directory:</span>
                  <div className="font-mono text-sm" style={{ color: 'var(--color-text)' }}>{selectedDirectory}</div>
                </div>
              </div>
            </div>

            <pre
              className="p-3 rounded-md font-mono text-sm overflow-auto max-h-60"
              style={{ backgroundColor: 'var(--color-bg-dark)', color: 'var(--color-text)' }}
            >
              {installationOutput}
            </pre>

            {currentStep === InstallStep.COMPLETE && (
              <div className="mt-4 flex justify-end">
                <button
                  onClick={() => navigate('/settings/mcp/ssh-setup')}
                  className="px-4 py-2 rounded"
                  style={{
                    backgroundColor: 'var(--color-primary)',
                    color: 'white'
                  }}
                >
                  Done
                </button>
              </div>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  // Render action buttons based on current step
  const renderActionButtons = () => {
    if (currentStep === InstallStep.SELECT_CONFIG) {
      return (
        <div className="flex justify-between">
          <button
            onClick={() => navigate('/settings/mcp/ssh-setup')}
            className="px-4 py-2 rounded"
            style={{
              backgroundColor: 'var(--color-surface-dark)',
              color: 'var(--color-text)'
            }}
          >
            Cancel
          </button>

          <button
            onClick={handleNextStep}
            disabled={!selectedSSH || !selectedMCP}
            className="px-4 py-2 rounded flex items-center"
            style={{
              backgroundColor: !selectedSSH || !selectedMCP ? 'var(--color-primary-muted)' : 'var(--color-primary)',
              color: 'white',
              cursor: !selectedSSH || !selectedMCP ? 'not-allowed' : 'pointer'
            }}
          >
            Next
            <ArrowRightIcon className="h-4 w-4 ml-1" />
          </button>
        </div>
      );
    }

    // No buttons for other steps as they have their own buttons
    return null;
  };

  return (
    <div className="py-6 px-4 md:px-8 max-w-[1400px] mx-auto" style={{ backgroundColor: 'var(--color-bg)' }}>
      <div className="flex items-center mb-6">
        <button
          onClick={() => navigate('/settings/mcp/ssh-setup')}
          className="mr-4 flex items-center text-sm"
          style={{ color: 'var(--color-primary)' }}
        >
          <ArrowLeftIcon className="h-4 w-4 mr-1" />
          Back to SSH Setup
        </button>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--color-text)' }}>
          MCP Installation
        </h1>
      </div>

      {/* Step Indicator */}
      {renderStepIndicator()}

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
        {/* Info box (only show on first step) */}
        {currentStep === InstallStep.SELECT_CONFIG && (
          <div className="p-4 rounded-lg" style={{ backgroundColor: 'var(--color-surface-dark)', borderColor: 'var(--color-border)' }}>
            <div className="flex items-start">
              <InformationCircleIcon className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" style={{ color: 'var(--color-primary)' }} />
              <p style={{ color: 'var(--color-text-secondary)' }}>
                Install MCP on your target machines using SSH. Select an SSH configuration and an MCP installation tool to proceed.
              </p>
            </div>
          </div>
        )}

        {/* Loading indicator */}
        {loading ? (
          <div className="p-4 text-center" style={{ color: 'var(--color-text-muted)' }}>
            Loading configurations...
          </div>
        ) : (
          /* Step content */
          renderStepContent()
        )}

        {/* Action buttons */}
        {renderActionButtons()}
      </div>
    </div>
  );
};

export default MCPInstall;

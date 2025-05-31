import React, { createContext, useContext, useState, useEffect, ReactNode, useRef, useCallback } from 'react';
import { useMCP } from './MCPContext';
import { useWebSocket } from './WebSocketContext';
import axios from 'axios';

// Define the structure for a command that needs approval
interface PendingCommand {
  id: string;
  command: string;
  toolName: string;
  parameters: any;
  description: string;
  timestamp: number;
}

// Define the structure for a command result
interface CommandResult {
  id: string;
  commandId: string;
  success: boolean;
  result: any;
  error?: string;
  timestamp: number;
  needsAnalysis?: boolean; // Flag to indicate if this result needs analysis
  isAnalyzed?: boolean;    // Flag to indicate if analysis is complete
}

// Define the context type
interface MCPAgentContextType {
  isAgentEnabled: boolean;
  toggleAgent: () => void;
  pendingCommands: PendingCommand[];
  commandResults: CommandResult[];
  approveCommand: (commandId: string) => Promise<void>;
  rejectCommand: (commandId: string) => void;
  modifyCommand: (commandId: string, newParameters: any) => void;
  clearCommandResult: (resultId: string) => void;
  processUserRequest: (request: string, chatCallback?: (response: string) => void) => Promise<void>;
  isProcessing: boolean;
  isConnected: boolean;
  sseStatus: string;
  reconnectToServer: () => void;
  addCommandResult: (result: CommandResult) => void;
  isAnalyzing: boolean;
  isStopped: boolean;
  toggleStopResume: () => void;
  addToPendingAnalysis: (result: CommandResult) => void;
  hasClientIdIssue: boolean;
  attemptClientIdRecovery: () => Promise<boolean>;
  notificationResults: CommandResult[];
  clearNotifications: () => void;
}

// Create the context
const MCPAgentContext = createContext<MCPAgentContextType | undefined>(undefined);

// Hook for using the context
export const useMCPAgent = () => {
  const context = useContext(MCPAgentContext);
  if (context === undefined) {
    throw new Error('useMCPAgent must be used within a MCPAgentProvider');
  }
  return context;
};

interface MCPAgentProviderProps {
  children: ReactNode;
}

export const MCPAgentProvider: React.FC<MCPAgentProviderProps> = ({ children }) => {
  // Get MCP context
  const {
    isConnected,
    defaultServer,
    availableTools,
    executeCommand,
    mcpConnection,
    reconnect,
    getClientId,
    registerToolResultHandler
  } = useMCP();

  // Get WebSocket context
  const { isFullyReady, waitForReady } = useWebSocket();

  // State for agent
  const [isAgentEnabled, setIsAgentEnabled] = useState<boolean>(() => {
    // Try to get the agent enabled state from localStorage
    const savedState = localStorage.getItem('mcp_agent_enabled');
    return savedState ? JSON.parse(savedState) : false;
  });

  const [pendingCommands, setPendingCommands] = useState<PendingCommand[]>([]);
  const [commandResults, setCommandResults] = useState<CommandResult[]>([]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState<boolean>(false);

  // New state for analysis loop
  const [pendingAnalysis, setPendingAnalysis] = useState<CommandResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [isStopped, setIsStopped] = useState<boolean>(false);

  // Add state for tracking client ID issues
  const [hasClientIdIssue, setHasClientIdIssue] = useState<boolean>(false);
  const recoveryAttemptsRef = useRef<number>(0);
  const MAX_RECOVERY_ATTEMPTS = 3;

  const lastProcessedClientIdRef = useRef<string | null>(null); // Ref to track processed clientId for welcome message

  // Use a ref to store processed client IDs across renders
  const processedClientIdsRef = useRef<Set<string>>(new Set());

  // Add these at the top of the component with other state variables
  const [welcomeInProgress, setWelcomeInProgress] = useState(false);
  const clientIdRef = useRef<string | null>(null);

  // Add a new state for notifications that won't be shown in the main chat
  const [notificationResults, setNotificationResults] = useState<CommandResult[]>([]);

  // Replace the old addCommandResult with a new version that adds to notifications
  const addCommandResult = useCallback((result: CommandResult) => {
    // Add to notifications instead of command results
    setNotificationResults(prev => [...prev, result]);
  }, []);

  // Add a new function to clear notifications
  const clearNotifications = useCallback(() => {
    setNotificationResults([]);
  }, []);

  // Stable addToPendingAnalysis using useCallback (moved up)
  const addToPendingAnalysis = useCallback((result: CommandResult) => {
    setPendingAnalysis(prev => [...prev, result]);
  }, []);

  // Get the active AI model from localStorage
  useEffect(() => {
    const storedModel = localStorage.getItem('selectedModelId');
    if (storedModel) {
      setActiveModel(storedModel);
    }
  }, []);

  // Save agent enabled state to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('mcp_agent_enabled', JSON.stringify(isAgentEnabled));
  }, [isAgentEnabled]);

  // Session persistence - save state to sessionStorage
  useEffect(() => {
    if (isAgentEnabled && commandResults.length > 0) {
      sessionStorage.setItem('mcp_command_results', JSON.stringify(commandResults));
    }
  }, [isAgentEnabled, commandResults]);

  useEffect(() => {
    if (isAgentEnabled && pendingCommands.length > 0) {
      sessionStorage.setItem('mcp_pending_commands', JSON.stringify(pendingCommands));
    }
  }, [isAgentEnabled, pendingCommands]);

  // Restore state from sessionStorage
  useEffect(() => {
    if (isAgentEnabled && isConnected) {
      const savedResults = sessionStorage.getItem('mcp_command_results');
      const savedCommands = sessionStorage.getItem('mcp_pending_commands');

      if (savedResults && commandResults.length === 0) {
        try {
          setCommandResults(JSON.parse(savedResults));
        } catch (e) {
          console.error('Error restoring command results:', e);
        }
      }

      if (savedCommands && pendingCommands.length === 0) {
        try {
          setPendingCommands(JSON.parse(savedCommands));
        } catch (e) {
          console.error('Error restoring pending commands:', e);
        }
      }
    }
  }, [isAgentEnabled, isConnected]);

  // Register the tool result handler with MCPContext
  useEffect(() => {
    const handleToolResult = (data: any) => {
      console.log('Tool result received in MCPAgentContext:', data);
      const commandId = data.commandId || data.metadata?.commandId || 'unknown';
      const commandResult: CommandResult = {
        id: `result-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
        commandId,
        success: !data.error,
        result: data.result || data,
        error: data.error,
        timestamp: Date.now(),
        needsAnalysis: true,
        isAnalyzed: false
      };
      addCommandResult(commandResult);
      if (!data.error) {
        addToPendingAnalysis(commandResult);
      }
    };
    registerToolResultHandler(handleToolResult);
    return () => {
      registerToolResultHandler(() => {});
    };
  }, [registerToolResultHandler, addCommandResult, addToPendingAnalysis]);

  // Analysis worker effect
  useEffect(() => {
    if (
      isAgentEnabled &&
      !isProcessing &&
      !isAnalyzing &&
      pendingAnalysis.length > 0 &&
      !isStopped &&
      isConnected
    ) {
      const analyzeNextResult = async () => {
        setIsAnalyzing(true);

        try {
          const resultToAnalyze = pendingAnalysis[0];

          // Find the original command
          const originalCommand = pendingCommands.find(
            cmd => cmd.id === resultToAnalyze.commandId
          );

          // If we couldn't find the original command, we'll still analyze
          // but with limited context
          const commandContext = originalCommand ? {
            command: originalCommand.command,
            toolName: originalCommand.toolName,
            parameters: originalCommand.parameters
          } : {
            command: "Unknown command",
            toolName: "unknown",
            parameters: {}
          };

          console.log(`Analyzing result for command: ${commandContext.toolName}`);

          // Call analysis endpoint
          const response = await axios.post('/api/ai/analyze-result', {
            result: resultToAnalyze.result,
            originalRequest: commandContext.command,
            toolName: commandContext.toolName,
            parameters: commandContext.parameters,
            modelId: activeModel
          });

          if (response.data) {
            // Add analysis as a command result
            const analysisResult: CommandResult = {
              id: `analysis-${Date.now()}`,
              commandId: 'analysis',
              success: true,
              result: {
                type: 'text',
                text: response.data.analysis
              },
              timestamp: Date.now()
            };

            addCommandResult(analysisResult);

            // Mark the original result as analyzed
            setCommandResults(prev =>
              prev.map(res =>
                res.id === resultToAnalyze.id
                  ? {...res, isAnalyzed: true}
                  : res
              )
            );

            // Add follow-up commands if any
            if (
              response.data.followUpCommands &&
              Array.isArray(response.data.followUpCommands) &&
              response.data.followUpCommands.length > 0
            ) {
              const newCommands = response.data.followUpCommands.map((cmd: any) => ({
                id: `cmd-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
                command: cmd.command,
                toolName: cmd.toolName,
                parameters: cmd.parameters,
                description: cmd.description,
                timestamp: Date.now()
              }));

              setPendingCommands(prev => [...prev, ...newCommands]);
            } else if (response.data.isComplete) {
              // Task is complete
              const completionMessage: CommandResult = {
                id: `completion-${Date.now()}`,
                commandId: 'completion',
                success: true,
                result: {
                  type: 'text',
                  text: 'Task completed. No further actions needed.'
                },
                timestamp: Date.now()
              };

              addCommandResult(completionMessage);
            }
          }
        } catch (error) {
          console.error('Error analyzing result:', error);

          // Add error message
          const errorResult: CommandResult = {
            id: `analysis-error-${Date.now()}`,
            commandId: 'error',
            success: false,
            result: null,
            error: `Error analyzing result: ${error.message}`,
            timestamp: Date.now()
          };

          addCommandResult(errorResult);
        } finally {
          // Remove the analyzed result from the queue
          setPendingAnalysis(prev => prev.slice(1));
          setIsAnalyzing(false);
        }
      };

      analyzeNextResult();
    }
  }, [
    isAgentEnabled,
    isProcessing,
    isAnalyzing,
    pendingAnalysis,
    isStopped,
    pendingCommands,
    activeModel,
    isConnected,
    addCommandResult,
    addToPendingAnalysis
  ]);

  // Track if we're currently processing a welcome message
  const isProcessingWelcomeRef = useRef(false);

  // Improved welcome message handling with better deduplication
  useEffect(() => {
    // If agent is not enabled or we're in the process of connecting, do nothing
    if (!isAgentEnabled || isConnecting) {
      return;
    }

    // Add debouncing for welcome messages to prevent duplicates
    const welcomeMessageDebounceTime = 300; // Reduced to 300ms for faster responsiveness
    let welcomeMessageTimer: number | null = null;
    let connectingMessageTimer: number | null = null;

    // Function to clear any existing timers
    const clearTimers = () => {
      if (welcomeMessageTimer) {
        clearTimeout(welcomeMessageTimer);
        welcomeMessageTimer = null;
      }

      if (connectingMessageTimer) {
        clearTimeout(connectingMessageTimer);
        connectingMessageTimer = null;
      }
    };

    // Clear timers on initial run
    clearTimers();

    const initializeAgentLogic = () => {
      // If agent enabled, connected, and we have a server with client ID
      if (
        isAgentEnabled &&
        mcpConnection.status === 'connected' &&
        mcpConnection.clientId &&
        availableTools.length > 0
      ) {
        // Check if we've already processed this client ID
        const clientId = mcpConnection.clientId;
        const alreadyProcessed = processedClientIdsRef.current.has(clientId);

        // Check if we already have a welcome message for this client ID
        const alreadyHasWelcome = commandResults.some(
          r => r.commandId === 'welcome' &&
               (r.result?.text?.includes(`ClientId: ${clientId}`) ||
                r.result?.text?.includes(clientId.substring(0, 8)))
        );

        if (!alreadyProcessed && !isProcessingWelcomeRef.current && !alreadyHasWelcome) {
          console.log(`[MCP-AGENT] New connection detected with clientId: ${clientId}. Scheduling welcome message.`);

          // Mark as processing to prevent race conditions
          isProcessingWelcomeRef.current = true;

          // Mark this client ID as processed immediately
          processedClientIdsRef.current.add(clientId);
          lastProcessedClientIdRef.current = clientId;

          // Set a short timeout just to let other operations complete
          welcomeMessageTimer = window.setTimeout(() => {
            console.log(`[MCP-AGENT] Displaying welcome message for clientId: ${clientId}`);

            // Determine server details to show in welcome message - handle case of missing defaultServer
            const serverName = defaultServer ? 
              (defaultServer.mcp_nickname || defaultServer.mcp_host) : 
              'Unknown';
            const serverAddress = defaultServer ? 
              `${defaultServer.mcp_host}:${defaultServer.mcp_port}` : 
              'Unknown address';

            const welcomeResult: CommandResult = {
              id: `result-welcome-${Date.now()}`,
              commandId: 'welcome', // Consistent commandId for welcome messages
              success: true,
              result: {
                type: 'text',
                text: `MCP Agent initialized. Connected to ${serverName} at ${serverAddress}. ${availableTools.length} tools available. ClientId: ${clientId}`
              },
              timestamp: Date.now()
            };

            addCommandResult(welcomeResult);
            welcomeMessageTimer = null;
            isProcessingWelcomeRef.current = false;
          }, welcomeMessageDebounceTime);
        } else {
          console.log(`[MCP-AGENT] Client ID ${clientId} already processed or welcome message in progress, skipping.`);
        }
      } else if (isAgentEnabled && mcpConnection.status === 'connecting' && !mcpConnection.clientId) {
        // We're connecting but missing clientId - show connecting message if it doesn't exist
        const connectingMessageExists = commandResults.some(r => r.commandId === 'agent_connecting');

        if (!connectingMessageExists && mcpConnection.error === null) {
          connectingMessageTimer = window.setTimeout(() => {
            const connectingMsg: CommandResult = {
              id: `result-agent-connecting-${Date.now()}`,
              commandId: 'agent_connecting',
              success: true,
              result: {
                type: 'text',
                text: `Establishing connection to MCP server ${defaultServer?.mcp_nickname || 'Unknown'}. Please wait...`
              },
              timestamp: Date.now()
            };

            addCommandResult(connectingMsg);
            connectingMessageTimer = null;
          }, 300); // Short delay to avoid racing with other messages
        }
      }
    };

    initializeAgentLogic();

    // Clean up timers on unmount
    return () => {
      clearTimers();
    };
  }, [
    isAgentEnabled,
    isConnecting,
    mcpConnection.status,
    mcpConnection.clientId,
    mcpConnection.error,
    defaultServer,
    availableTools.length,
    addCommandResult,
    commandResults
  ]);

  // Monitor SSE connection status changes for errors (from MCPAgentContext perspective)
  useEffect(() => {
    if (isAgentEnabled && mcpConnection.status === 'error' && mcpConnection.error) {
      // Add an error message
      const errorResult: CommandResult = {
        id: `result-sse-error-${Date.now()}`,
        commandId: 'error',
        success: false,
        result: null,
        error: `MCP connection error: ${mcpConnection.error}. Please try reconnecting.`,
        timestamp: Date.now()
      };

      addCommandResult(errorResult);
    }
  }, [isAgentEnabled, mcpConnection.status, mcpConnection.error, addCommandResult]);

  // Verify client ID availability
  useEffect(() => {
    const checkClientId = () => {
      const clientId = getClientId();
      const newHasIssue = mcpConnection.status === 'connected' && !clientId;

      if (newHasIssue !== hasClientIdIssue) {
        setHasClientIdIssue(newHasIssue);

        // If issue resolved, reset recovery attempts
        if (!newHasIssue && recoveryAttemptsRef.current > 0) {
          console.log('Client ID issue resolved, resetting recovery attempts');
          recoveryAttemptsRef.current = 0;
        }
      }
    };

    // Check initially
    checkClientId();

    // Set up an interval to check periodically
    const interval = setInterval(checkClientId, 5000);

    return () => {
      clearInterval(interval);
    };
  }, [mcpConnection, getClientId, hasClientIdIssue]);

  // Enhanced function to attempt client ID recovery with better error handling
  const attemptClientIdRecovery = async (): Promise<boolean> => {
    if (!defaultServer) {
      console.error('No default server selected');
      return false;
    }

    if (recoveryAttemptsRef.current >= MAX_RECOVERY_ATTEMPTS) {
      console.error(`Max recovery attempts (${MAX_RECOVERY_ATTEMPTS}) reached`);

      // Add a recovery failure message
      const recoveryFailedMsg: CommandResult = {
        id: `result-recovery-failed-${Date.now()}`,
        commandId: 'error',
        success: false,
        result: null,
        error: `Client ID recovery failed after ${MAX_RECOVERY_ATTEMPTS} attempts. Please try refreshing the page.`,
        timestamp: Date.now()
      };

      addCommandResult(recoveryFailedMsg);
      return false;
    }

    recoveryAttemptsRef.current++;
    console.log(`Attempting client ID recovery (attempt ${recoveryAttemptsRef.current}/${MAX_RECOVERY_ATTEMPTS})`);

    // Add a recovery attempt message
    const recoveryAttemptMsg: CommandResult = {
      id: `result-recovery-attempt-${Date.now()}`,
      commandId: 'system',
      success: true,
      result: {
        type: 'text',
        text: `Attempting to recover client ID (attempt ${recoveryAttemptsRef.current}/${MAX_RECOVERY_ATTEMPTS})...`
      },
      timestamp: Date.now()
    };

    addCommandResult(recoveryAttemptMsg);

    try {
      // First try direct retrieval - this is more reliable than reconnection
      console.log('Attempting direct client ID retrieval from API endpoint...');
      
      // Try the direct API endpoint
      try {
        const response = await axios.get('/api/mcp/get-client-id', {
          params: {
            host: defaultServer.mcp_host,
            port: defaultServer.mcp_port
          },
          timeout: 8000
        });

        if (response.data && response.data.clientId) {
          const directClientId = response.data.clientId;
          console.log(`Recovered client ID via direct API: ${directClientId}`);

          // Store this client ID in local storage to be picked up by MCPContext
          const connectionInfo = {
            connectionId: `recovery-${Date.now()}`,
            clientId: directClientId,
            timestamp: Date.now(),
            server: `${defaultServer.mcp_host}:${defaultServer.mcp_port}`
          };
          
          try {
            localStorage.setItem('mcp_connection_info', JSON.stringify(connectionInfo));
            console.log('Successfully saved connection info to localStorage');
          } catch (storageError) {
            console.error('Error saving connection info to localStorage:', storageError);
          }

          // Force a reconnect to use the new client ID
          reconnect();

          // Wait a bit and check if we now have a client ID
          await new Promise(resolve => setTimeout(resolve, 2000));

          const finalClientId = getClientId();
          if (finalClientId) {
            // Create a concise connection message for user
            const successMessage = `MCP Client ID: ${finalClientId} connected. You can now chat with AI as usual.`;

            // Log the message
            console.log(`MCP connected at ${defaultServer.mcp_host}:${defaultServer.mcp_port}. Client ID: ${finalClientId}`);

            // Add the connection success notification
            const commandResult: CommandResult = {
              id: `result-connect-${Date.now()}`,
              commandId: 'system',
              success: true,
              result: {
                type: 'text',
                text: successMessage
              },
              timestamp: Date.now()
            };

            // Replace any existing notifications
            setNotificationResults([commandResult]);

            // Store the timestamp of when we showed the success message
            localStorage.setItem('mcp_connection_success_time', Date.now().toString());

            // Don't add any more notifications after this
            return true;
          }
        }
      } catch (directError) {
        console.error('Error using direct client ID retrieval:', directError);
      }
      
      // If direct retrieval failed, try a direct fetch to the SSE endpoint
      try {
        console.log(`Attempting direct fetch to SSE endpoint: http://${defaultServer.mcp_host}:${defaultServer.mcp_port}/sse`);
        
        // Use fetch API with AbortController for timeout control
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000);
        
        const directResponse = await fetch(`http://${defaultServer.mcp_host}:${defaultServer.mcp_port}/sse`, {
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        const text = await directResponse.text();
        console.log('Direct SSE response:', text);
        
        // Try to extract clientId from the SSE response
        const clientIdMatch = text.match(/clientId["\s:]*["']?([^"',}\s]+)/);
        if (clientIdMatch && clientIdMatch[1]) {
          const extractedClientId = clientIdMatch[1];
          console.log(`Extracted client ID from direct SSE response: ${extractedClientId}`);
          
          // Store this client ID
          const connectionInfo = {
            connectionId: `recovery-sse-${Date.now()}`,
            clientId: extractedClientId,
            timestamp: Date.now(),
            server: `${defaultServer.mcp_host}:${defaultServer.mcp_port}`
          };
          
          localStorage.setItem('mcp_connection_info', JSON.stringify(connectionInfo));
          
          // Force a reconnect to use the new client ID
          reconnect();
          
          // Wait a bit and check if we now have a client ID
          await new Promise(resolve => setTimeout(resolve, 2000));
          
          const sseClientId = getClientId();
          if (sseClientId) {
            // Create a concise connection message for user
            const successMessage = `MCP Client ID: ${sseClientId} connected. You can now chat with AI as usual.`;

            // Log the message
            console.log(`MCP connected at ${defaultServer.mcp_host}:${defaultServer.mcp_port}. Client ID: ${sseClientId}`);

            // Add the connection success notification
            const commandResult: CommandResult = {
              id: `result-connect-${Date.now()}`,
              commandId: 'system',
              success: true,
              result: {
                type: 'text',
                text: successMessage
              },
              timestamp: Date.now()
            };

            // Replace any existing notifications
            setNotificationResults([commandResult]);

            // Store the timestamp of when we showed the success message
            localStorage.setItem('mcp_connection_success_time', Date.now().toString());

            // Don't add any more notifications after this
            return true;
          }
        }
      } catch (sseError) {
        console.error('Error accessing SSE endpoint directly:', sseError);
      }

      // If all direct methods failed, try reconnection
      console.log('Direct client ID retrieval methods failed, trying reconnection');

      reconnect();

      // Wait a bit and check if that fixed it
      await new Promise(resolve => setTimeout(resolve, 2000));

      // If we have a client ID now, we're good
      const clientId = getClientId();
      if (clientId) {
        console.log('Client ID recovery successful via reconnection');

        // Create a concise connection message for user
        const successMessage = `MCP Client ID: ${clientId} connected. You can now chat with AI as usual.`;

        // Log the message
        console.log(`MCP connected at ${defaultServer.mcp_host}:${defaultServer.mcp_port}. Client ID: ${clientId}`);

        // Add the connection success notification
        const commandResult: CommandResult = {
          id: `result-connect-${Date.now()}`,
          commandId: 'system',
          success: true,
          result: {
            type: 'text',
            text: successMessage
          },
          timestamp: Date.now()
        };

        // Replace any existing notifications
        setNotificationResults([commandResult]);

        // Store the timestamp of when we showed the success message
        localStorage.setItem('mcp_connection_success_time', Date.now().toString());

        // Don't add any more notifications after this
        return true;
      }

      console.log('Client ID recovery failed');

      // Add a failure message
      const recoveryFailedMsg: CommandResult = {
        id: `result-recovery-failed-${Date.now()}`,
        commandId: 'error',
        success: false,
        result: null,
        error: `Client ID recovery attempt ${recoveryAttemptsRef.current} failed. ${recoveryAttemptsRef.current < MAX_RECOVERY_ATTEMPTS ? 'Will try again.' : 'Please refresh the page.'}`,
        timestamp: Date.now()
      };

      addCommandResult(recoveryFailedMsg);

      return false;
    } catch (error) {
      console.error('Error during client ID recovery:', error);

      // Add an error message
      const recoveryErrorMsg: CommandResult = {
        id: `result-recovery-error-${Date.now()}`,
        commandId: 'error',
        success: false,
        result: null,
        error: `Error during client ID recovery: ${error.message || 'Unknown error'}`,
        timestamp: Date.now()
      };

      addCommandResult(recoveryErrorMsg);

      return false;
    }
  };

  // Enhanced function to reconnect to the server with better feedback
  const reconnectToServer = () => {
    if (isAgentEnabled) {
      // Clear existing results but preserve important system messages
      const systemMessages = commandResults.filter(r =>
        r.commandId === 'system' ||
        r.commandId === 'error' ||
        r.commandId === 'welcome'
      );

      // Add a reconnecting message
      const reconnectingResult: CommandResult = {
        id: `result-reconnecting-${Date.now()}`,
        commandId: 'reconnecting',
        success: true,
        result: {
          type: 'text',
          text: 'Reconnecting to MCP server...'
        },
        timestamp: Date.now()
      };

      // Keep system messages and add the reconnecting message
      setCommandResults([...systemMessages, reconnectingResult]);

      // Reset the processed client IDs to allow a new welcome message
      processedClientIdsRef.current.clear();

      // Reset recovery attempts counter
      recoveryAttemptsRef.current = 0;

      // Attempt to reconnect
      reconnect();

      // Schedule a check to verify if reconnection was successful
      setTimeout(() => {
        if (!isConnected || !getClientId()) {
          // If still not connected after a delay, show a message suggesting page refresh
          const reconnectFailedResult: CommandResult = {
            id: `result-reconnect-failed-${Date.now()}`,
            commandId: 'error',
            success: false,
            result: null,
            error: 'Reconnection attempt did not establish a connection. You may need to refresh the page.',
            timestamp: Date.now()
          };

          addCommandResult(reconnectFailedResult);
        }
      }, 5000); // Check after 5 seconds
    }
  };

  // Further enhanced toggle agent function with improved connection handling and WebSocket readiness check
  const toggleAgent = async () => {
    const newState = !isAgentEnabled;
    setIsAgentEnabled(newState);

    // If enabling the agent, clear any previous state and reset
    if (newState) {
      // Don't try to connect if we already have a clientId or are in the process of connecting
      if (mcpConnection.clientId) {
        console.log('[MCP-AGENT] Already connected with client ID, skipping connection process');
        checkAndEstablishMCPConnection();
        return;
      }
      
      if (isConnecting) {
        console.log('[MCP-AGENT] Connection already in progress, skipping duplicate attempt');
        return;
      }
      
      setIsConnecting(true);
      
      // Clear existing results
      setCommandResults([]);

      // Reset the processed client IDs to allow a new welcome message
      processedClientIdsRef.current.clear();

      // Reset recovery attempts counter
      recoveryAttemptsRef.current = 0;

      // Add an initial status message
      const initializingMsg: CommandResult = {
        id: `result-initializing-${Date.now()}`,
        commandId: 'system',
        success: true,
        result: {
          type: 'text',
          text: 'Initializing MCP Agent...'
        },
        timestamp: Date.now()
      };

      addCommandResult(initializingMsg);

      try {
        // Check WebSocket readiness first with a more robust approach
        if (!isFullyReady) {
          console.log('[MCP-AGENT] WebSocket not fully ready, using waitForReady function...');

          // Add a waiting message
          const waitingMsg: CommandResult = {
            id: `result-waiting-websocket-${Date.now()}`,
            commandId: 'system',
            success: true,
            result: {
              type: 'text',
              text: 'Waiting for WebSocket connection to be fully established...'
            },
            timestamp: Date.now()
          };

          addCommandResult(waitingMsg);

          try {
            // Use the Promise-based waitForReady function with a longer timeout
            await waitForReady(15000); // 15 second timeout for better reliability
            console.log('[MCP-AGENT] WebSocket is now ready, proceeding with MCP connection');

            // Add a success message
            const readyMsg: CommandResult = {
              id: `result-websocket-ready-${Date.now()}`,
              commandId: 'system',
              success: true,
              result: {
                type: 'text',
                text: 'WebSocket connection established successfully.'
              },
              timestamp: Date.now()
            };

            addCommandResult(readyMsg);
          } catch (error) {
            console.error('[MCP-AGENT] Error waiting for WebSocket readiness:', error);

            // Add an error message
            const errorMsg: CommandResult = {
              id: `result-websocket-error-${Date.now()}`,
              commandId: 'error',
              success: false,
              result: null,
              error: `WebSocket connection failed to establish: ${error.message}. Please refresh the page.`,
              timestamp: Date.now()
            };

            addCommandResult(errorMsg);

            // Try one more time with a different approach - sometimes the WebSocket is actually ready
            // but the readiness detection failed
            setTimeout(() => {
              console.log('[MCP-AGENT] Attempting MCP connection despite WebSocket readiness failure');
              checkAndEstablishMCPConnection();
            }, 2000);

            return;
          }
        } else {
          console.log('[MCP-AGENT] WebSocket is already ready, proceeding with MCP connection');
        }

        // Now that WebSocket check is complete, proceed with MCP connection
        await checkAndEstablishMCPConnection();

      } catch (error) {
        console.error('[MCP-AGENT] Error during agent initialization:', error);

        // Add a general error message
        const generalErrorMsg: CommandResult = {
          id: `result-init-error-${Date.now()}`,
          commandId: 'error',
          success: false,
          result: null,
          error: `Error initializing MCP Agent: ${error.message}. Please try again or refresh the page.`,
          timestamp: Date.now()
        };

        addCommandResult(generalErrorMsg);
      } finally {
        setIsConnecting(false);
      }
    }
  };

  // Enhanced helper function to check and establish MCP connection with async support
  const checkAndEstablishMCPConnection = async (): Promise<boolean> => {
    // If we're already connected with a client ID, just show a welcome message
    if (mcpConnection.clientId) {
      console.log(`[MCP-AGENT] Agent enabled with existing connection. Displaying welcome message.`);

      // Skip if we're already showing a welcome message for this clientId
      const clientId = mcpConnection.clientId;
      const alreadyHasWelcome = commandResults.some(
        r => r.commandId === 'welcome' &&
            (r.result?.text?.includes(`ClientId: ${clientId}`) ||
              r.result?.text?.includes(clientId.substring(0, 8)))
      );

      if (!alreadyHasWelcome) {
        // Determine server details to show in welcome message - handle case of missing defaultServer
        const serverName = defaultServer ? 
          (defaultServer.mcp_nickname || defaultServer.mcp_host) : 
          'Unknown';
        const serverAddress = defaultServer ? 
          `${defaultServer.mcp_host}:${defaultServer.mcp_port}` : 
          'Unknown address';

        // Add a welcome message
        const welcomeResult: CommandResult = {
          id: `result-welcome-${Date.now()}`,
          commandId: 'welcome',
          success: true,
          result: {
            type: 'text',
            text: `MCP Agent initialized. Connected to ${serverName} at ${serverAddress}. ${availableTools.length} tools available. ClientId: ${mcpConnection.clientId}`
          },
          timestamp: Date.now()
        };

        addCommandResult(welcomeResult);
      } else {
        console.log(`[MCP-AGENT] Welcome message already exists for clientId: ${mcpConnection.clientId}, skipping.`);
      }
      
      setHasClientIdIssue(false);
      return true;
    }
    
    // If we're not connected but connecting, don't start another connection
    if (mcpConnection.status === 'connecting') {
      console.log('[MCP-AGENT] Connection already in progress, waiting...');
      
      const waitingMsg: CommandResult = {
        id: `result-waiting-${Date.now()}`,
        commandId: 'system',
        success: true,
        result: {
          type: 'text',
          text: 'Connection attempt already in progress, waiting...'
        },
        timestamp: Date.now()
      };
      
      addCommandResult(waitingMsg);
      return false;
    }
    
    // If we're not connected, try to reconnect
    if (defaultServer) {
      console.log('[MCP-AGENT] Not connected to MCP server, attempting to connect...');

      // Add a connecting message
      const connectingMsg: CommandResult = {
        id: `result-connecting-${Date.now()}`,
        commandId: 'system',
        success: true,
        result: {
          type: 'text',
          text: `Connecting to MCP server ${defaultServer.mcp_nickname || defaultServer.mcp_host}...`
        },
        timestamp: Date.now()
      };

      addCommandResult(connectingMsg);

      // Attempt to reconnect
      reconnect();

      // Wait a bit to see if connection is established
      await new Promise(resolve => setTimeout(resolve, 5000));

      // Check if connection was successful
      if (mcpConnection.clientId) {
        console.log('[MCP-AGENT] Successfully connected to MCP server');

        // Create a concise connection message for user
        const successMessage = `MCP Client ID: ${mcpConnection.clientId} connected. You can now chat with AI as usual.`;

        // Log the message
        console.log(`MCP connected at ${defaultServer.mcp_host}:${defaultServer.mcp_port}. Client ID: ${mcpConnection.clientId}`);

        // Add the connection success notification
        const commandResult: CommandResult = {
          id: `result-connect-${Date.now()}`,
          commandId: 'system',
          success: true,
          result: {
            type: 'text',
            text: successMessage
          },
          timestamp: Date.now()
        };

        // Replace any existing notifications
        setNotificationResults([commandResult]);

        // Store the timestamp of when we showed the success message
        localStorage.setItem('mcp_connection_success_time', Date.now().toString());

        // Don't add any more notifications after this
        return true;
      } else {
        console.log('[MCP-AGENT] Failed to connect to MCP server');

        // Add a failure message
        const failureMsg: CommandResult = {
          id: `result-connection-failure-${Date.now()}`,
          commandId: 'error',
          success: false,
          result: null,
          error: 'Failed to connect to MCP server. Please try again or check your server settings.',
          timestamp: Date.now()
        };

        addCommandResult(failureMsg);

        // Try to recover client ID if we have a connection issue
        if (hasClientIdIssue) {
          console.log('[MCP-AGENT] Detected client ID issue, attempting recovery...');

          const recoveryMsg: CommandResult = {
            id: `result-recovery-attempt-${Date.now()}`,
            commandId: 'system',
            success: true,
            result: {
              type: 'text',
              text: 'Attempting to recover client ID...'
            },
            timestamp: Date.now()
          };

          addCommandResult(recoveryMsg);

          // Attempt client ID recovery
          const recovered = await attemptClientIdRecovery();
          return recovered;
        }

        return false;
      }
    } else if (!defaultServer) {
      // No default server selected
      const noServerMsg: CommandResult = {
        id: `result-no-server-${Date.now()}`,
        commandId: 'error',
        success: false,
        result: null,
        error: 'No MCP server selected. Please select a server from the dropdown.',
        timestamp: Date.now()
      };

      addCommandResult(noServerMsg);
      return false;
    }

    return false;
  };

  // Toggle stop/resume processing
  const toggleStopResume = () => {
    setIsStopped(!isStopped);

    if (isStopped) {
      // We're resuming, add a message
      const resumeMessage: CommandResult = {
        id: `result-resume-${Date.now()}`,
        commandId: 'system',
        success: true,
        result: {
          type: 'text',
          text: 'Processing resumed.'
        },
        timestamp: Date.now()
      };

      addCommandResult(resumeMessage);
    } else {
      // We're stopping, add a message
      const stopMessage: CommandResult = {
        id: `result-stop-${Date.now()}`,
        commandId: 'system',
        success: true,
        result: {
          type: 'text',
          text: 'Processing paused. Click "Resume" to continue.'
        },
        timestamp: Date.now()
      };

      addCommandResult(stopMessage);
    }
  };

  // Simplified processUserRequest function that doesn't actually process user requests
  // This avoids interference with the normal chat flow
  const processUserRequest = async (request: string, chatCallback?: (response: string) => void) => {
    console.log('MCP Agent processUserRequest called, but not processing request: ', request.substring(0, 30) + '...');
    
    // Do nothing - the chat should be handled by the normal chat flow
    if (chatCallback) {
      chatCallback("This message is being handled by the regular chat system now.");
    }
    
    return;
  };

  // Approve and execute a command
  const approveCommand = async (commandId: string) => {
    const command = pendingCommands.find(cmd => cmd.id === commandId);
    if (!command) {
      console.error(`Command with id ${commandId} not found`);
      return;
    }

    // Mark as processing
    setIsProcessing(true);

    try {
      let result: any;
      let hasError = false;

      try {
        // Execute command
        result = await executeCommand(command.toolName, command.parameters);
      } catch (error) {
        // Check if this is a client ID issue
        const errorMessage = error.message || '';
        if (
          errorMessage.includes('client') &&
          errorMessage.includes('id') &&
          mcpConnection.status === 'connected'
        ) {
          console.log('Detected client ID issue during command execution, attempting recovery');

          // Try to recover client ID
          const recovered = await attemptClientIdRecovery();

          if (recovered) {
            // Retry the command with the recovered client ID
            console.log('Retrying command with recovered client ID');
            try {
              result = await executeCommand(command.toolName, command.parameters);
              hasError = false;
            } catch (retryError) {
              console.error('Command retry failed:', retryError);
              result = {
                error: `Failed to execute command after client ID recovery: ${retryError.message}`
              };
              hasError = true;
            }
          } else {
            result = {
              error: 'Failed to recover client ID. Please try manually reconnecting.'
            };
            hasError = true;
          }
        } else {
          // This is not a client ID issue or we couldn't recover
          console.error('Command execution error:', error);
          result = { error: error.message || 'Unknown error executing command' };
          hasError = true;
        }
      }

      // Create result entry
      const commandResult: CommandResult = {
        id: `result-${Date.now()}`,
        commandId: command.id,
        success: !hasError,
        result: result,
        timestamp: Date.now(),
        needsAnalysis: !hasError,
        isAnalyzed: false
      };

      // Add result and queue for analysis if successful
      addCommandResult(commandResult);
      if (!hasError) {
        addToPendingAnalysis(commandResult);
      }

      // Remove from pending commands
      setPendingCommands(prev => prev.filter(cmd => cmd.id !== commandId));
    } catch (error) {
      console.error('Error approving command:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  // Reject a command
  const rejectCommand = (commandId: string) => {
    // Simply remove the command from pending commands
    setPendingCommands(prev => prev.filter(cmd => cmd.id !== commandId));
  };

  // Modify a command's parameters
  const modifyCommand = (commandId: string, newParameters: any) => {
    // Update the command's parameters
    setPendingCommands(prev =>
      prev.map(cmd =>
        cmd.id === commandId ? { ...cmd, parameters: newParameters } : cmd
      )
    );
  };

  // Clear a command result
  const clearCommandResult = (resultId: string) => {
    // Remove the result from command results
    setCommandResults(prev => prev.filter(res => res.id !== resultId));
  };

  // Update clientId processing with better deduplication
  useEffect(() => {
    if (!mcpConnection.clientId || !isConnected) return;

    // Store the current client ID
    const currentClientId = mcpConnection.clientId;
    
    // Only process client ID if it's new or we haven't processed it yet
    if (clientIdRef.current !== currentClientId && !processedClientIdsRef.current.has(currentClientId)) {
      console.log(`Processing new client ID: ${currentClientId}`);
      
      // Update clientIdRef
      clientIdRef.current = currentClientId;
      
      // Add to processed list to prevent duplicate messages
      processedClientIdsRef.current.add(currentClientId);
      
      // Create welcome message
      setWelcomeInProgress(true);
      
      const welcomeMsg: CommandResult = {
        id: `result-connected-${Date.now()}`,
        commandId: 'system',
        success: true,
        result: {
          type: 'text',
          text: `Connected to MCP Server. You can now chat with AI as usual.`
        },
        timestamp: Date.now()
      };
      
      setNotificationResults(prev => [...prev, welcomeMsg]);
      setWelcomeInProgress(false);
    } else if (clientIdRef.current === currentClientId) {
      console.log(`[MCP-AGENT] Client ID ${currentClientId} already processed, skipping.`);
    } else if (welcomeInProgress) {
      console.log(`[MCP-AGENT] Welcome message in progress for ${currentClientId}, skipping.`);
    }
  }, [mcpConnection.clientId, isConnected]);

  return (
    <MCPAgentContext.Provider
      value={{
        isAgentEnabled,
        toggleAgent,
        pendingCommands,
        commandResults,
        approveCommand,
        rejectCommand,
        modifyCommand,
        clearCommandResult,
        processUserRequest,
        isProcessing,
        isConnected,
        sseStatus: mcpConnection.status,
        reconnectToServer,
        addCommandResult,
        isAnalyzing,
        isStopped,
        toggleStopResume,
        addToPendingAnalysis,
        hasClientIdIssue,
        attemptClientIdRecovery,
        notificationResults,
        clearNotifications
      }}
    >
      {children}
    </MCPAgentContext.Provider>
  );
};

export default MCPAgentContext;
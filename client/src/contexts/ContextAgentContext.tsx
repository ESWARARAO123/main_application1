import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { checkContextAgentStatus, getUserContext, ContextAgentStatus, UserContext } from '../services/contextAgentService';
import { useMCP } from './MCPContext';

interface ContextAgentContextType {
  isContextAgentAvailable: boolean;
  hasUserRules: boolean;
  isContextAgentEnabled: boolean;
  toggleContextAgent: () => void;
  userContext: UserContext | null;
  isLoadingContext: boolean;
  refreshContext: () => Promise<void>;
}

const ContextAgentContext = createContext<ContextAgentContextType | undefined>(undefined);

export const useContextAgent = (): ContextAgentContextType => {
  const context = useContext(ContextAgentContext);
  if (!context) {
    throw new Error('useContextAgent must be used within a ContextAgentProvider');
  }
  return context;
};

interface ContextAgentProviderProps {
  children: ReactNode;
}

export const ContextAgentProvider: React.FC<ContextAgentProviderProps> = ({ children }) => {
  const { isMCPEnabled } = useMCP();
  const [status, setStatus] = useState<ContextAgentStatus>({
    available: false,
    has_rules: false
  });
  const [isContextAgentEnabled, setIsContextAgentEnabled] = useState<boolean>(false);
  const [userContext, setUserContext] = useState<UserContext | null>(null);
  const [isLoadingContext, setIsLoadingContext] = useState<boolean>(false);

  // Check context agent status when MCP is enabled
  useEffect(() => {
    if (isMCPEnabled) {
      checkStatus();
    } else {
      setIsContextAgentEnabled(false);
    }
  }, [isMCPEnabled]);

  // Load user context when context agent is enabled
  useEffect(() => {
    if (isContextAgentEnabled && status.available) {
      loadUserContext();
    }
  }, [isContextAgentEnabled, status.available]);

  const checkStatus = async () => {
    try {
      const agentStatus = await checkContextAgentStatus();
      setStatus(agentStatus);
      
      // If context agent was previously enabled and is still available, keep it enabled
      if (isContextAgentEnabled && !agentStatus.available) {
        setIsContextAgentEnabled(false);
      }
    } catch (error) {
      console.error('Error checking context agent status:', error);
      setStatus({
        available: false,
        has_rules: false
      });
      setIsContextAgentEnabled(false);
    }
  };

  const loadUserContext = async () => {
    try {
      setIsLoadingContext(true);
      const context = await getUserContext();
      setUserContext(context);
    } catch (error) {
      console.error('Error loading user context:', error);
      setUserContext(null);
    } finally {
      setIsLoadingContext(false);
    }
  };

  const refreshContext = async () => {
    if (isContextAgentEnabled) {
      await loadUserContext();
    }
  };

  const toggleContextAgent = () => {
    // Only allow toggling if MCP is enabled and context agent is available
    if (isMCPEnabled && status.available) {
      setIsContextAgentEnabled(prev => !prev);
    }
  };

  const value = {
    isContextAgentAvailable: status.available,
    hasUserRules: status.has_rules,
    isContextAgentEnabled,
    toggleContextAgent,
    userContext,
    isLoadingContext,
    refreshContext
  };

  return (
    <ContextAgentContext.Provider value={value}>
      {children}
    </ContextAgentContext.Provider>
  );
};

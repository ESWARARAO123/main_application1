import { useState } from 'react';
import { ExtendedChatMessage, ChatSession } from '../types';
import { chatbotService } from '../services/chatbotService';
import { ragChatService } from '../services/ragChatService';

/**
 * Hook for managing chat sessions
 */
export const useChatSessions = () => {
  // Session state
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sessionTitle, setSessionTitle] = useState('');
  const [editingTitle, setEditingTitle] = useState(false);
  
  // Message state
  const [messages, setMessages] = useState<ExtendedChatMessage[]>([]);

  // Message loading state
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [messageOffset, setMessageOffset] = useState(0);
  const [hasMoreMessages, setHasMoreMessages] = useState(false);
  const [totalMessages, setTotalMessages] = useState(0);

  // UI state
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({
    'Today': true,
    'Yesterday': true,
    'Previous 7 Days': true,
    'Previous 30 Days': false,
    'Older': false
  });
  const [showSidebar, setShowSidebar] = useState(() => {
    const savedPreference = localStorage.getItem('chatSidebarExpanded');
    return savedPreference !== null ? savedPreference === 'true' : true;
  });

  // Fetch all chat sessions
  const fetchSessions = async () => {
    try {
      setLoadingSessions(true);
      const fetchedSessions = await chatbotService.getSessions();
      setSessions(fetchedSessions);

      if (fetchedSessions.length > 0 && !activeSessionId) {
        setActiveSessionId(fetchedSessions[0].id);
        setSessionTitle(fetchedSessions[0].title);
      }
    } catch (error) {
      console.error('Error fetching chat sessions:', error);
    } finally {
      setLoadingSessions(false);
    }
  };

  // Fetch messages for a specific session
  const fetchSessionMessages = async (sessionId: string, append = false) => {
    try {
      setLoadingMessages(true);
      const offset = append ? messageOffset : 0;
      const response = await chatbotService.getSession(sessionId, 12, offset);

      const { messages: fetchedMessages, total } = response;
      setTotalMessages(total);
      setHasMoreMessages(offset + fetchedMessages.length < total);

      // Convert standard ChatMessage to ExtendedChatMessage
      const extendedMessages = fetchedMessages.map(msg => ({
        ...msg,
        role: msg.role as 'user' | 'assistant' | 'system'
      }));

      if (append) {
        setMessages(prev => [...extendedMessages, ...prev]);
        setMessageOffset(prev => prev + extendedMessages.length);
      } else {
        setMessages(extendedMessages);
        setMessageOffset(extendedMessages.length);
      }

      setSessionTitle(response.session.title);
    } catch (error) {
      console.error('Error fetching session messages:', error);
    } finally {
      setLoadingMessages(false);
    }
  };

  // Load more messages for pagination
  const loadMoreMessages = async () => {
    if (!activeSessionId || !hasMoreMessages || loadingMessages) return;
    await fetchSessionMessages(activeSessionId, true);
  };

  // Create a new chat session
  const createNewSession = async () => {
    try {
      const newSession = await chatbotService.createSession('New Chat');
      setSessions(prev => [newSession, ...prev]);
      setActiveSessionId(newSession.id);
      setSessionTitle(newSession.title);
      setMessages([]);
      setMessageOffset(0);
      setHasMoreMessages(false);
      setTotalMessages(0);
    } catch (error) {
      console.error('Error creating new session:', error);
    }
  };

  // Delete a chat session
  const deleteSession = async (sessionId: string, event?: React.MouseEvent) => {
    if (event) {
      event.stopPropagation();
    }

    if (!confirm('Are you sure you want to delete this chat?')) return;

    try {
      // Delete the chat session from the database
      await chatbotService.deleteSession(sessionId);

      // Also clear any RAG data associated with this session
      try {
        await ragChatService.clearRagData(sessionId);
        console.log('RAG data cleared for session:', sessionId);
      } catch (ragError) {
        console.error('Error clearing RAG data:', ragError);
        // Continue with session deletion even if RAG data clearing fails
      }

      // Update the UI
      setSessions(prev => prev.filter(s => s.id !== sessionId));

      if (activeSessionId === sessionId) {
        const remainingSessions = sessions.filter(s => s.id !== sessionId);
        if (remainingSessions.length > 0) {
          setActiveSessionId(remainingSessions[0].id);
          setSessionTitle(remainingSessions[0].title);
        } else {
          setActiveSessionId(null);
          setSessionTitle('');
        }
      }
    } catch (error) {
      console.error('Error deleting session:', error);
    }
  };

  // Update session title
  const updateSessionTitle = async () => {
    if (!activeSessionId || !sessionTitle.trim()) return;

    try {
      await chatbotService.updateSession(activeSessionId, { title: sessionTitle });
      setSessions(prev => prev.map(s =>
        s.id === activeSessionId ? { ...s, title: sessionTitle } : s
      ));
      setEditingTitle(false);
    } catch (error) {
      console.error('Error updating session title:', error);
    }
  };

  // Toggle sidebar visibility
  const toggleSidebar = () => {
    setShowSidebar(prev => {
      const newValue = !prev;
      localStorage.setItem('chatSidebarExpanded', String(newValue));
      return newValue;
    });
  };

  // Toggle session group expansion
  const toggleGroup = (groupLabel: string) => {
    setExpandedGroups(prev => ({
      ...prev,
      [groupLabel]: !prev[groupLabel]
    }));
  };

  // Clear current chat
  const resetChat = () => {
    if (confirm('Are you sure you want to clear the current chat?')) {
      setMessages([]);
    }
  };

  return {
    // State
    sessions,
    activeSessionId,
    sessionTitle,
    editingTitle,
    messages,
    loadingMessages,
    loadingSessions,
    hasMoreMessages,
    expandedGroups,
    showSidebar,
    
    // Setters
    setActiveSessionId,
    setSessionTitle,
    setEditingTitle,
    setMessages,
    
    // Actions
    fetchSessions,
    fetchSessionMessages,
    loadMoreMessages,
    createNewSession,
    deleteSession,
    updateSessionTitle,
    toggleSidebar,
    toggleGroup,
    resetChat
  };
}; 
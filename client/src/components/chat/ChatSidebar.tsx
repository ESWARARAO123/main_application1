import React, { useEffect, useState } from 'react';
import { ChatSession } from '../../types';
import {
  PlusIcon,
  TrashIcon,
  ChevronDownIcon,
  CalendarDaysIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  PencilIcon
} from '@heroicons/react/24/outline';
import { 
  Box, 
  Flex, 
  Text, 
  Badge, 
  Tooltip, 
  Spinner, 
  Input, 
  InputGroup, 
  InputLeftElement, 
  InputRightElement, 
  IconButton
} from '@chakra-ui/react';
import { animations } from './chatStyles';

interface TimeGroup {
  label: string;
  sessions: ChatSession[];
}

interface ChatSidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  expandedGroups: Record<string, boolean>;
  loadingSessions: boolean;
  isCollapsed?: boolean;
  onCreateSession: (title?: string) => void;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string, event?: React.MouseEvent) => void;
  onEditSession: (sessionId: string, newTitle: string) => void;
  onToggleGroup: (groupLabel: string) => void;
  onToggleCollapse?: () => void;
}

const ChatSidebar: React.FC<ChatSidebarProps> = ({
  sessions,
  activeSessionId,
  expandedGroups,
  loadingSessions,
  isCollapsed = false,
  onCreateSession,
  onSelectSession,
  onDeleteSession,
  onEditSession,
  onToggleGroup,
  onToggleCollapse
}) => {
  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  

  
  // Edit session state
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  
  // Filter sessions based on search query
  const filteredSessions = sessions.filter(session =>
    session.title.toLowerCase().includes(searchQuery.toLowerCase())
  );



  // Handle starting edit mode for a session
  const handleStartEdit = (sessionId: string, currentTitle: string, event: React.MouseEvent) => {
    event.stopPropagation();
    setEditingSessionId(sessionId);
    setEditingTitle(currentTitle);
  };

  // Handle saving edited title
  const handleSaveEdit = (sessionId: string) => {
    if (editingTitle.trim()) {
      onEditSession(sessionId, editingTitle.trim());
    }
    setEditingSessionId(null);
    setEditingTitle('');
  };

  // Handle canceling edit
  const handleCancelEdit = () => {
    setEditingSessionId(null);
    setEditingTitle('');
  };
  // Group sessions by time
  const groupSessionsByTime = (sessions: ChatSession[]): TimeGroup[] => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    const lastWeekStart = new Date(today);
    lastWeekStart.setDate(lastWeekStart.getDate() - 7);

    const lastMonthStart = new Date(today);
    lastMonthStart.setDate(lastMonthStart.getDate() - 30);

    const groups: TimeGroup[] = [
      { label: 'Today', sessions: [] },
      { label: 'Yesterday', sessions: [] },
      { label: 'Previous 7 Days', sessions: [] },
      { label: 'Previous 30 Days', sessions: [] },
      { label: 'Older', sessions: [] }
    ];

    sessions.forEach(session => {
      const lastMessageDate = new Date(session.last_message_timestamp);

      if (lastMessageDate >= today) {
        groups[0].sessions.push(session);
      } else if (lastMessageDate >= yesterday) {
        groups[1].sessions.push(session);
      } else if (lastMessageDate >= lastWeekStart) {
        groups[2].sessions.push(session);
      } else if (lastMessageDate >= lastMonthStart) {
        groups[3].sessions.push(session);
      } else {
        groups[4].sessions.push(session);
      }
    });

    // Remove empty groups
    return groups.filter(group => group.sessions.length > 0);
  };

  const groupedSessions = groupSessionsByTime(filteredSessions);

  // Add animation styles
  useEffect(() => {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
      ${animations.fadeIn}
      ${animations.slideIn}
      .session-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      }
    `;
    document.head.appendChild(styleElement);

    return () => {
      document.head.removeChild(styleElement);
    };
  }, []);

  // Render just a floating arrow button when sidebar is collapsed
  if (isCollapsed) {
    return (
      <Box
        position="absolute"
        left="16px"
        top="16px"
        zIndex={10}
      >
        <Tooltip label="Open chat sidebar" placement="right" hasArrow>
          <Box
            as="button"
            onClick={onToggleCollapse}
            p={2}
            borderRadius="md"
            bg="var(--color-surface)"
            color="var(--color-primary)"
            boxShadow="0 2px 6px rgba(0,0,0,0.15)"
            _hover={{
              bg: "var(--color-surface-dark)",
              boxShadow: "0 3px 8px rgba(0,0,0,0.2)"
            }}
            sx={{
              animation: "fadeIn 0.3s ease-in-out",
              transition: "all 0.2s ease"
            }}
          >
            <ChevronRightIcon className="w-5 h-5" />
          </Box>
        </Tooltip>
      </Box>
    );
  }

  // Render the expanded sidebar
  return (
    <Box h="100%" w="100%" overflow="hidden" bg="var(--color-surface-light)" position="relative">
      <Box position="absolute" top={2} right={2} zIndex={2}>
        <Tooltip label="Collapse sidebar" placement="left" hasArrow>
          <Box
            as="button"
            onClick={onToggleCollapse}
            p={1}
            borderRadius="md"
            bg="var(--color-surface)"
            color="var(--color-text-muted)"
            _hover={{ bg: "var(--color-surface-dark)", color: "var(--color-text)" }}
            sx={{ transition: "all 0.2s ease" }}
          >
            <ChevronLeftIcon className="w-4 h-4" />
          </Box>
        </Tooltip>
      </Box>

      <Box p={4}>
        {/* Header with title */}
        <Box mb={4} textAlign="center">
          <Text 
            fontSize="lg" 
            fontWeight="bold" 
            color="var(--color-primary)"
            letterSpacing="wide"
          >
            PinnacleAI
          </Text>
        </Box>

        {/* New Chat Button */}
        <Tooltip label="Start a new conversation" placement="right" hasArrow>
          <button
            onClick={() => onCreateSession()}
            className="w-full py-2 px-3 rounded-full flex items-center justify-center transition-all duration-200 hover:scale-105 mb-4"
            style={{
              background: 'linear-gradient(to right, var(--color-primary), var(--color-secondary))',
              color: 'white',
              border: 'none',
              boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
              fontWeight: 500
            }}
          >
            <PlusIcon className="w-5 h-5 mr-2" />
            New Chat
          </button>
        </Tooltip>

        {/* Search Bar */}
        <Box mb={4}>
          <InputGroup size="sm">
            <InputLeftElement pointerEvents="none">
              <MagnifyingGlassIcon 
                className="w-4 h-4" 
                style={{ color: 'var(--color-text-muted)' }} 
              />
            </InputLeftElement>
            <Input
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              bg="var(--color-surface-dark)"
              border="1px solid var(--color-border)"
              borderRadius="md"
              color="var(--color-text)"
              fontSize="sm"
              _placeholder={{ color: 'var(--color-text-muted)' }}
              _focus={{
                borderColor: 'var(--color-primary)',
                boxShadow: '0 0 0 1px var(--color-primary-translucent)'
              }}
            />
            {searchQuery && (
              <InputRightElement>
                <IconButton
                  aria-label="Clear search"
                  icon={<XMarkIcon className="w-3 h-3" />}
                  size="xs"
                  variant="ghost"
                  color="var(--color-text-muted)"
                  onClick={() => setSearchQuery('')}
                  _hover={{ color: 'var(--color-text)' }}
                />
              </InputRightElement>
            )}
          </InputGroup>
        </Box>

        {/* Search Results Info */}
        {searchQuery && (
          <Box pb={2}>
            <Text fontSize="xs" color="var(--color-text-muted)" textAlign="center">
              {filteredSessions.length} result{filteredSessions.length !== 1 ? 's' : ''} for "{searchQuery}"
            </Text>
          </Box>
        )}
      </Box>

      {loadingSessions ? (
        <Flex p={4} justify="center" align="center" color="var(--color-text-muted)">
          <Spinner size="sm" mr={2} color="var(--color-primary)" />
          <Text fontSize="sm">Loading sessions...</Text>
        </Flex>
      ) : sessions.length === 0 ? (
        <Box p={4} textAlign="center" color="var(--color-text-muted)">
          <Text fontSize="sm" fontWeight="medium">No chat history yet</Text>
          <Text fontSize="xs" mt={1}>Start a new chat to begin!</Text>
        </Box>
      ) : filteredSessions.length === 0 ? (
        <Box p={4} textAlign="center" color="var(--color-text-muted)">
          <MagnifyingGlassIcon className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <Text fontSize="sm" fontWeight="medium">No conversations found</Text>
          <Text fontSize="xs" mt={1}>Try a different search term</Text>
        </Box>
      ) : (
        <Box overflowY="auto" h="calc(100% - 200px)" px={2}>
          {groupedSessions.map((group, groupIndex) => (
            <Box key={group.label} mb={2} sx={{ animation: `fadeIn 0.3s ease-in-out ${groupIndex * 0.1}s`, animationFillMode: "both" }}>
              {/* Group header */}
              <Flex
                onClick={() => onToggleGroup(group.label)}
                px={3} py={2} align="center" justify="space-between" cursor="pointer"
                bg={expandedGroups[group.label] ? 'var(--color-surface)' : 'var(--color-surface-dark)'}
                borderTop="1px solid var(--color-border-subtle)"
                borderBottom={expandedGroups[group.label] ? '1px solid var(--color-border-subtle)' : 'none'}
                _hover={{ bg: 'var(--color-surface-dark)', transition: 'all 0.2s ease' }}
                transition="all 0.2s ease"
              >
                <Flex align="center">
                  <CalendarDaysIcon className="w-4 h-4 mr-2" style={{ color: 'var(--color-primary)' }} />
                  <Text fontSize="sm" fontWeight="medium" color="var(--color-text-secondary)">
                    {group.label}
                  </Text>
                  <Badge
                    ml={2}
                    fontSize="0.7em"
                    colorScheme="blue"
                    variant="subtle"
                    borderRadius="full"
                  >
                    {group.sessions.length}
                  </Badge>
                </Flex>
                <Box
                  transition="transform 0.2s ease"
                  transform={expandedGroups[group.label] ? 'rotate(0deg)' : 'rotate(-90deg)'}
                >
                  <ChevronDownIcon className="w-4 h-4" style={{ color: 'var(--color-text-muted)' }} />
                </Box>
              </Flex>

              {/* Group sessions */}
              {expandedGroups[group.label] && group.sessions.map((session, sessionIndex) => (
                <Tooltip
                  key={session.id}
                  label={session.title}
                  placement="right"
                  hasArrow
                  bg="var(--color-surface-dark)"
                  color="var(--color-text)"
                >
                  <Box
                    className="session-card group"
                    onClick={() => editingSessionId !== session.id && onSelectSession(session.id)}
                    px={3} py={2} mx={1} mb={1}
                    borderRadius="md"
                    bg={activeSessionId === session.id ? 'var(--color-primary-translucent)' : 'var(--color-surface-light)'}
                    borderLeft={activeSessionId === session.id ? '3px solid var(--color-primary)' : '3px solid transparent'}
                    cursor={editingSessionId === session.id ? 'default' : 'pointer'}
                    transition="all 0.2s ease"
                    sx={{
                      animation: `slideIn 0.2s ease-out ${sessionIndex * 0.05}s`,
                      animationFillMode: "both"
                    }}
                    _hover={{ bg: editingSessionId === session.id ? undefined : 'var(--color-surface-dark)' }}
                  >
                    <Flex justify="space-between" align="center">
                      <Box flex="1" minW={0}>
                        {editingSessionId === session.id ? (
                          <Input
                            value={editingTitle}
                            onChange={(e) => setEditingTitle(e.target.value)}
                            onKeyPress={(e) => {
                              if (e.key === 'Enter') {
                                handleSaveEdit(session.id);
                              } else if (e.key === 'Escape') {
                                handleCancelEdit();
                              }
                            }}
                            onBlur={() => handleSaveEdit(session.id)}
                            size="sm"
                            fontSize="sm"
                            bg="var(--color-surface-dark)"
                            border="1px solid var(--color-primary)"
                            color="var(--color-text)"
                            autoFocus
                          />
                        ) : (
                          <>
                            <Text
                              fontSize="sm"
                              fontWeight="medium"
                              color="var(--color-text)"
                              isTruncated
                            >
                              {session.title}
                            </Text>
                            <Flex align="center" mt={1}>
                              <Text
                                fontSize="xs"
                                color="var(--color-text-muted)"
                                mr={2}
                              >
                                {new Date(session.last_message_timestamp).toLocaleString(undefined, {
                                  month: 'short',
                                  day: 'numeric',
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </Text>
                            </Flex>
                          </>
                        )}
                      </Box>
                      
                      {editingSessionId !== session.id && (
                        <Flex className="opacity-0 group-hover:opacity-100 transition-all">
                          <Tooltip label="Edit session" placement="top" hasArrow>
                            <IconButton
                              aria-label="Edit session"
                              icon={<PencilIcon className="w-3 h-3" />}
                              size="xs"
                              variant="ghost"
                              color="var(--color-text-muted)"
                              onClick={(e) => handleStartEdit(session.id, session.title, e)}
                              _hover={{ color: 'var(--color-text)', bg: 'var(--color-surface)' }}
                              mr={1}
                            />
                          </Tooltip>
                          <Tooltip label="Delete session" placement="top" hasArrow>
                            <IconButton
                              aria-label="Delete session"
                              icon={<TrashIcon className="w-3 h-3" />}
                              size="xs"
                              variant="ghost"
                              color="var(--color-text-muted)"
                              onClick={(e) => {
                                e.stopPropagation();
                                onDeleteSession(session.id, e);
                              }}
                              _hover={{ color: 'var(--color-error)', bg: 'var(--color-surface)' }}
                            />
                          </Tooltip>
                        </Flex>
                      )}
                    </Flex>
                  </Box>
                </Tooltip>
              ))}
            </Box>
          ))}
        </Box>
      )}

      {/* Bottom Branding */}
      <Box 
        position="absolute" 
        bottom={0} 
        left={0} 
        right={0} 
        p={4} 
        borderTop="1px solid var(--color-border-subtle)"
        bg="var(--color-surface-light)"
      >
        <Flex align="center" justify="center" direction="column">
          <Text 
            fontSize="xs" 
            color="var(--color-text-muted)" 
            textAlign="center"
            mb={1}
          >
            Powered by
          </Text>
          <Text 
            fontSize="sm" 
            fontWeight="bold" 
            color="var(--color-primary)"
            textAlign="center"
            letterSpacing="wide"
          >
            @SierraEdge AI
          </Text>
        </Flex>
      </Box>


    </Box>
  );
};

export default ChatSidebar;
import React, { useRef, useEffect } from 'react';
import { ChatMessage as ChatMessageType, ExtendedChatMessage } from '../../types';
import ChatMessage from './ChatMessage';
import { ChatBubbleLeftRightIcon, ArrowDownIcon } from '@heroicons/react/24/outline';
import { messageListStyles, messageBubbleStyles } from './chatStyles';
import { useMCPAgent } from '../../contexts/MCPAgentContext';

// Import RagSource type
import { RagSource } from '../../services/ragChatService';

// Allow for extended message types that include system messages
interface MessageListProps {
  messages: ExtendedChatMessage[];
  isLoading: boolean;
  hasMoreMessages: boolean;
  loadMoreMessages: () => void;
  loadingMessages: boolean;
  isEmpty?: boolean;
  conversationId?: string; // Current conversation ID
}

const MessageList: React.FC<MessageListProps> = ({
  messages,
  isLoading,
  hasMoreMessages,
  loadMoreMessages,
  loadingMessages,
  isEmpty = false,
  conversationId
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const { pendingCommands, commandResults } = useMCPAgent();

  // Auto-scroll to bottom of messages when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, pendingCommands, commandResults]);

  // Setup infinite scrolling for messages
  useEffect(() => {
    const messagesContainer = messagesContainerRef.current;
    if (!messagesContainer) return;

    const handleScroll = () => {
      if (messagesContainer.scrollTop === 0 && hasMoreMessages && !loadingMessages) {
        loadMoreMessages();
      }
    };

    messagesContainer.addEventListener('scroll', handleScroll);
    return () => messagesContainer.removeEventListener('scroll', handleScroll);
  }, [messagesContainerRef, hasMoreMessages, loadingMessages, loadMoreMessages]);

  // Group messages by role
  const groupedMessages = React.useMemo(() => {
    const groups: { role: string; messages: ExtendedChatMessage[] }[] = [];

    messages.forEach(message => {
      const lastGroup = groups[groups.length - 1];
      if (lastGroup && lastGroup.role === message.role) {
        lastGroup.messages.push(message);
      } else {
        groups.push({ role: message.role, messages: [message] });
      }
    });

    return groups;
  }, [messages]);

  if (isEmpty) {
    return (
      <div style={messageListStyles.emptyState} className="chat-empty-state">
        <div style={messageListStyles.emptyIcon}>
          <ChatBubbleLeftRightIcon className="w-8 h-8" style={{ color: 'var(--color-primary)' }} />
        </div>
        <h3 className="text-xl font-bold mb-2 text-center" style={{ color: 'var(--color-text)' }}>Chat Assistant</h3>
        <p className="mb-8 text-center max-w-md" style={{ color: 'var(--color-text-muted)' }}>
          I'm here to help with your tasks. You can ask me questions, request assistance, or get information about the platform.
        </p>
      </div>
    );
  }

  return (
    <div
      ref={messagesContainerRef}
      style={{
        ...messageListStyles.container,
        scrollbarWidth: 'thin',
        scrollbarColor: 'var(--color-primary) var(--color-surface-dark)',
      }}
      className="scrollbar-thin scrollbar-thumb-primary scrollbar-track-surface-dark message-list-container"
    >
      {/* Load more messages indicator */}
      {hasMoreMessages && (
        <div style={{ display: 'flex', justifyContent: 'center', margin: '1rem 0' }}>
          <button
            onClick={loadMoreMessages}
            style={{
              ...messageListStyles.loadMoreButton,
              opacity: loadingMessages ? 0.7 : 1,
              cursor: loadingMessages ? 'not-allowed' : 'pointer',
            }}
            disabled={loadingMessages}
          >
            {loadingMessages ? (
              <div className="animate-spin mr-2 h-4 w-4 border-t-2 border-b-2 rounded-full" style={{ borderColor: 'var(--color-primary)' }}></div>
            ) : (
              <ArrowDownIcon className="w-4 h-4 mr-2" />
            )}
            {loadingMessages ? 'Loading...' : 'Load more messages'}
          </button>
        </div>
      )}

      <div style={{ width: '100%', maxWidth: '1200px', margin: '0 auto', padding: '1rem 1rem 0' }}>
        {/* Grouped messages */}
        {groupedMessages.map((group, groupIndex) => (
          <div
            key={groupIndex}
            style={{
              marginBottom: '1.5rem',
              animation: 'fadeIn 0.3s ease-in-out',
              animationFillMode: 'both',
              animationDelay: `${groupIndex * 0.05}s`
            }}
          >
            {group.messages.map((message, msgIdx) => (
              <div
                key={message.id || msgIdx}
                style={{
                  animation: 'slideIn 0.2s ease-out',
                  animationFillMode: 'both',
                  animationDelay: `${msgIdx * 0.05}s`
                }}
              >
                <ChatMessage
                  message={message}
                  isAI={group.role === 'assistant'}
                  conversationId={conversationId || message.conversationId}
                />
              </div>
            ))}
          </div>
        ))}

        {/* Loading indicator - only show if there's no streaming message already */}
        {isLoading && !messages.some(msg => msg.isStreaming) && (
          <div style={{ display: 'flex', marginBottom: '1rem', paddingBottom: '1rem' }}>
            <div style={messageBubbleStyles.ai.avatar}>
              AI
            </div>
            <div style={messageListStyles.typingIndicator}>
              <div style={{ ...messageListStyles.typingDot, animationDelay: '0ms' }}></div>
              <div style={{ ...messageListStyles.typingDot, animationDelay: '150ms' }}></div>
              <div style={{ ...messageListStyles.typingDot, animationDelay: '300ms' }}></div>
            </div>
          </div>
        )}

        {/* Extra space at the bottom to ensure messages aren't hidden behind input */}
        <div style={{ height: '150px' }}></div>
        <div ref={messagesEndRef} />
      </div>

      {/* Animation styles are defined in chatStyles.ts */}
    </div>
  );
};

export default MessageList;
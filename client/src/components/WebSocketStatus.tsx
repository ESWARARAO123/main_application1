import React, { useEffect, useState } from 'react';
import { useWebSocket } from '../contexts/WebSocketContext';
import { Badge, Box, Tooltip, Icon, Circle } from '@chakra-ui/react';
import { CheckCircleIcon, WarningIcon } from '@chakra-ui/icons';

/**
 * WebSocketStatus Component
 *
 * Displays the current WebSocket connection status in a subtle way
 */
const WebSocketStatus: React.FC = () => {
  const { connected, lastMessage } = useWebSocket();
  const [showLastMessage, setShowLastMessage] = useState<boolean>(false);
  const [connectionStatus, setConnectionStatus] = useState<string>(connected ? 'Connected' : 'Disconnected');

  // Update connection status when connected state changes
  useEffect(() => {
    setConnectionStatus(connected ? 'Connected' : 'Disconnected');

    // Log connection status changes
    console.log(`WebSocketStatus: Connection status changed to ${connected ? 'Connected' : 'Disconnected'}`);
  }, [connected]);

  // Reset the last message display after 5 seconds
  useEffect(() => {
    if (lastMessage) {
      setShowLastMessage(true);
      const timer = setTimeout(() => {
        setShowLastMessage(false);
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [lastMessage]);

  return (
    <Box display="flex" alignItems="center" gap={2}>
      <Tooltip
        label={`Real-time updates: ${connected ? 'Active' : 'Inactive'}`}
        placement="bottom"
      >
        <Box>
          {connected ? (
            <Circle size="10px" bg="green.400" display="inline-block" />
          ) : (
            <Circle size="10px" bg="red.400" display="inline-block" />
          )}
        </Box>
      </Tooltip>

      {showLastMessage && lastMessage && (
        <Tooltip
          label={`New update received: ${lastMessage.type}`}
          placement="bottom"
        >
          <Box animation="pulse 1s infinite">
            <Circle size="10px" bg="blue.400" display="inline-block" />
          </Box>
        </Tooltip>
      )}
    </Box>
  );
};

export default WebSocketStatus;

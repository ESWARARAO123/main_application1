// Singleton pattern for WebSocket connections using global variables
import React, { createContext, useContext, useEffect, useState, useCallback, ReactNode, useRef } from 'react';
import { useAuth } from './AuthContext';

/**
 * WebSocket message type
 */
interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

/**
 * WebSocket connection status enum
 * Used for more detailed connection state tracking
 */
enum ConnectionStatus {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  ERROR = 'error'
}

/**
 * WebSocket context type
 */
interface WebSocketContextType {
  connected: boolean;
  isFullyReady: boolean;
  connectionStatus: ConnectionStatus;
  reconnectAttempts: number;
  send: (data: any) => void;
  lastMessage: WebSocketMessage | null;
  // Add a listener for specific message types
  addMessageListener: (type: string, callback: (message: WebSocketMessage) => void) => () => void;
  // Force reconnection of the WebSocket
  reconnect: () => void;
  // Wait for WebSocket to be fully ready
  waitForReady: (timeoutMs?: number) => Promise<void>;
}

// Create the context with default values
const WebSocketContext = createContext<WebSocketContextType | null>(null);

// Global WebSocket instance to ensure only one connection is created
// This prevents multiple connections when components mount/unmount
let globalWebSocket: WebSocket | null = null;
let globalConnected = false;
let globalMessageListeners = new Map<string, Set<(message: WebSocketMessage) => void>>();
let globalLastMessage: WebSocketMessage | null = null;

// Message deduplication tracking
interface SentMessage {
  data: string; // Stringified message
  timestamp: number;
}
let recentlySentMessages: SentMessage[] = [];
const MESSAGE_DEDUPLICATION_WINDOW = 1000; // 1 second window for deduplication
const MESSAGE_DEDUPLICATION_LIMIT = 50; // Maximum number of messages to track

// Debounce timers for different message types
const debounceTimers = new Map<string, number>();
const DEFAULT_DEBOUNCE_DELAY = 300; // 300ms default debounce delay

// Expose the global WebSocket instance to the window object for debugging
(window as any).globalWebSocket = globalWebSocket;

/**
 * WebSocket Provider Props
 */
interface WebSocketProviderProps {
  children: ReactNode;
}

/**
 * WebSocket Provider Component
 *
 * Manages WebSocket connection and provides context for components to interact with WebSockets
 */
export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  // Use state to track the global WebSocket instance
  const [socket, setSocket] = useState<WebSocket | null>(globalWebSocket);
  const [connected, setConnected] = useState<boolean>(globalConnected);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(globalLastMessage);
  const [messageListeners, setMessageListeners] = useState<Map<string, Set<(message: WebSocketMessage) => void>>>(
    globalMessageListeners
  );
  const { user } = useAuth();

  // Reconnection settings
  const [reconnectAttempts, setReconnectAttempts] = useState<number>(0);
  const maxReconnectAttempts = 15; // Increased to 15 for more persistent reconnection
  const reconnectDelay = 1000; // Start with 1 second
  const maxReconnectDelay = 30000; // Maximum delay of 30 seconds
  const reconnectBackoffFactor = 1.5; // More gradual backoff factor (was 1.3)
  const reconnectResetDelay = 30000; // Wait 30 seconds before resetting reconnect attempts

  // Reconnection state tracking
  const reconnectTimeoutRef = useRef<number | null>(null);
  const lastReconnectTimeRef = useRef<number>(0);

  // Heartbeat settings
  const heartbeatInterval = 15000; // Reduced to 15 seconds for more responsive connection monitoring

  // Connection throttling to prevent excessive connections
  const connectionThrottleRef = useRef<number>(0);
  const connectionThrottleDelay = 5000; // 5 seconds between connection attempts

  // Debug counter to track component mounts
  const [mountCount] = useState(() => {
    console.log('WebSocketProvider mounted');
    return Math.random();
  });

  // Track if this is the primary WebSocketProvider instance
  // We'll use this to prevent multiple instances from creating connections
  const isPrimaryProvider = useRef<boolean>(false);

  // Check if we're the primary provider on mount
  useEffect(() => {
    if (!(window as any).__websocketProviderMounted) {
      console.log('This is the primary WebSocketProvider instance');
      (window as any).__websocketProviderMounted = true;
      isPrimaryProvider.current = true;
    } else {
      console.log('Secondary WebSocketProvider instance detected, will not create new connections');
    }

    return () => {
      // Only clear the flag if we're the primary provider
      if (isPrimaryProvider.current) {
        console.log('Primary WebSocketProvider unmounting, clearing global flag');
        (window as any).__websocketProviderMounted = false;
      }
    };
  }, []);

  // Track WebSocket readiness state with more granular control
  const [isFullyReady, setIsFullyReady] = useState(false);
  const readyTimeout = useRef<number | null>(null);
  const readyStateRef = useRef<boolean>(false); // Use a ref to track ready state across renders
  const connectionEstablishedRef = useRef<boolean>(false); // Track if onopen has fired
  const messageReceivedRef = useRef<boolean>(false); // Track if we've received any message

  // Promise resolver for waitForReady
  const readyPromiseResolverRef = useRef<{
    resolve: () => void;
    reject: (error: Error) => void;
    timeoutId: number | null;
  } | null>(null);

  /**
   * Connect to WebSocket server
   */
  const connect = useCallback(() => {
    if (!user) {
      console.log('Not connecting WebSocket: User not authenticated');
      setConnectionStatus(ConnectionStatus.DISCONNECTED);
      return; // Don't connect if not authenticated
    }

    // Update connection status to connecting
    setConnectionStatus(reconnectAttempts > 0 ? ConnectionStatus.RECONNECTING : ConnectionStatus.CONNECTING);

    // Implement connection throttling to prevent excessive connections
    const now = Date.now();
    if (now - connectionThrottleRef.current < connectionThrottleDelay) {
      console.log(`Connection throttled: Last attempt was ${(now - connectionThrottleRef.current) / 1000}s ago, waiting...`);
      return;
    }

    // Update the throttle timestamp
    connectionThrottleRef.current = now;

    // Only allow the primary provider to create connections
    if (!isPrimaryProvider.current) {
      console.log('Not connecting WebSocket: Not the primary provider');

      // Still update local state to reflect global state
      if (globalWebSocket && globalWebSocket.readyState === WebSocket.OPEN) {
        setSocket(globalWebSocket);
        setConnected(globalConnected);
        setLastMessage(globalLastMessage);
        setMessageListeners(globalMessageListeners);
      }

      return;
    }

    // If we already have a global connection, check if it's actually working
    if (globalWebSocket) {
      if (globalWebSocket.readyState === WebSocket.OPEN) {
        console.log('Using existing global WebSocket connection');
        setSocket(globalWebSocket);
        setConnected(true);
        globalConnected = true;
        setLastMessage(globalLastMessage);
        setMessageListeners(globalMessageListeners);

        // Send a ping to verify the connection is truly working
        try {
          globalWebSocket.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
          console.log('Sent verification ping to existing WebSocket connection');
        } catch (error) {
          console.error('Error sending ping to existing connection, will reconnect:', error);
          globalWebSocket.close();
          globalWebSocket = null;
          // Continue to create a new connection below
        }

        return;
      } else if (globalWebSocket.readyState === WebSocket.CONNECTING) {
        console.log('WebSocket connection is already in progress');
        setSocket(globalWebSocket);
        return;
      } else {
        // Connection is closing or closed, clean it up
        console.log('Cleaning up non-open WebSocket connection');
        globalWebSocket.close();
        globalWebSocket = null;
      }
    }

    // Determine WebSocket protocol based on current page protocol
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log(`Creating new global WebSocket connection to ${wsUrl}`);

    try {
      // Set a flag in localStorage to indicate we're trying to connect
      localStorage.setItem('websocketConnecting', 'true');

      console.log(`Attempting to connect to WebSocket at ${wsUrl}`);

      // Add more detailed debugging
      console.log('Current user:', user);
      console.log('Cookie available:', document.cookie ? 'Yes' : 'No');

      const newSocket = new WebSocket(wsUrl);
      console.log('WebSocket object created, readyState:', newSocket.readyState);

      // Store the socket in the global variable
      globalWebSocket = newSocket;

      // Update the window reference for debugging
      (window as any).globalWebSocket = globalWebSocket;

      // Set a connection timeout
      const connectionTimeout = setTimeout(() => {
        if (newSocket.readyState !== WebSocket.OPEN) {
          console.log('WebSocket connection timeout, closing and retrying');
          console.log('Final readyState before timeout:', newSocket.readyState);
          newSocket.close();
          // This will trigger the onclose handler which will attempt to reconnect
        }
      }, 10000); // 10 second timeout

      newSocket.onopen = () => {
        console.log('WebSocket connection established');
        clearTimeout(connectionTimeout);
        localStorage.removeItem('websocketConnecting');
        localStorage.setItem('websocketConnected', 'true');
        globalConnected = true;
        setConnected(true);
        setConnectionStatus(ConnectionStatus.CONNECTED);

        // Don't reset reconnect attempts immediately to prevent rapid reconnect cycles
        // Instead, reset after a delay if the connection remains stable
        if (reconnectAttempts > 0) {
          console.log(`Connection successful, will reset reconnect attempts after ${reconnectResetDelay/1000}s if stable`);
          setTimeout(() => {
            // Only reset if we're still connected
            if (globalConnected) {
              console.log('Connection remained stable, resetting reconnect attempts');
              setReconnectAttempts(0);
            }
          }, reconnectResetDelay);
        }

        // Mark connection as established
        connectionEstablishedRef.current = true;

        // Send an initial ping to verify the connection
        try {
          newSocket.send(JSON.stringify({
            type: 'ping',
            timestamp: Date.now(),
            message: 'Connection verification ping'
          }));
        } catch (error) {
          console.error('Error sending initial ping:', error);
        }

        // Clear any existing timeout
        if (readyTimeout.current) {
          clearTimeout(readyTimeout.current);
        }

        // Mark as fully ready after a short delay even if no pong is received
        // This ensures we don't get stuck in "not ready" state
        readyTimeout.current = window.setTimeout(() => {
          console.log('WebSocket connection established - marking as ready after timeout');

          // Only set ready if we haven't already done so via message receipt
          if (!readyStateRef.current) {
            console.log('Setting ready state via timeout (no messages received yet)');
            setIsFullyReady(true);
            readyStateRef.current = true;

            // Broadcast a connection ready event to help synchronize other components
            window.dispatchEvent(new CustomEvent('websocket-ready', {
              detail: { timestamp: Date.now() }
            }));

            // Resolve any pending waitForReady promises
            if (readyPromiseResolverRef.current) {
              console.log('Resolving waitForReady promise due to timeout');
              readyPromiseResolverRef.current.resolve();

              // Clear any timeout
              if (readyPromiseResolverRef.current.timeoutId) {
                clearTimeout(readyPromiseResolverRef.current.timeoutId);
              }

              readyPromiseResolverRef.current = null;
            }
          }

          readyTimeout.current = null;
        }, 300); // Use an even shorter timeout to avoid waiting too long
      };

      newSocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WebSocketMessage;

          // Only log non-ping messages to reduce console noise
          if (data.type !== 'ping' && data.type !== 'pong') {
            console.log('WebSocket message received:', data);
          }

          // Mark that we've received a message
          messageReceivedRef.current = true;

          // Mark as fully ready if we receive any message from the server
          // This ensures we don't get stuck waiting for specific message types
          if (!readyStateRef.current) {
            console.log('WebSocket connection confirmed with server response');
            setIsFullyReady(true);
            readyStateRef.current = true;

            // Broadcast a connection ready event to help synchronize other components
            window.dispatchEvent(new CustomEvent('websocket-ready', {
              detail: { timestamp: Date.now() }
            }));

            // Resolve any pending waitForReady promises
            if (readyPromiseResolverRef.current) {
              console.log('Resolving waitForReady promise due to message receipt');
              readyPromiseResolverRef.current.resolve();

              // Clear any timeout
              if (readyPromiseResolverRef.current.timeoutId) {
                clearTimeout(readyPromiseResolverRef.current.timeoutId);
              }

              readyPromiseResolverRef.current = null;
            }

            if (readyTimeout.current) {
              clearTimeout(readyTimeout.current);
              readyTimeout.current = null;
            }
          }

          // Update last message
          globalLastMessage = data;
          setLastMessage(data);

          // Notify listeners for this message type
          if (data.type && globalMessageListeners.has(data.type)) {
            const listeners = globalMessageListeners.get(data.type);
            if (listeners) {
              listeners.forEach(callback => callback(data));
            }
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      newSocket.onclose = (event) => {
        console.log(`WebSocket connection closed: ${event.code} ${event.reason}`);
        localStorage.removeItem('websocketConnected');
        localStorage.removeItem('websocketConnecting');
        globalConnected = false;
        setConnected(false);
        setIsFullyReady(false); // Reset fully ready state
        readyStateRef.current = false; // Also reset the ref
        connectionEstablishedRef.current = false; // Reset connection established state
        messageReceivedRef.current = false; // Reset message received state

        // Update connection status
        setConnectionStatus(ConnectionStatus.DISCONNECTED);

        // Broadcast a connection closed event
        window.dispatchEvent(new CustomEvent('websocket-closed', {
          detail: { timestamp: Date.now(), code: event.code, reason: event.reason }
        }));

        // Only attempt to reconnect if:
        // 1. It wasn't a clean close (or was closed due to error)
        // 2. We haven't exceeded max attempts
        // 3. The user is still authenticated
        if ((event.code !== 1000 || !event.wasClean) && reconnectAttempts < maxReconnectAttempts && user) {
          // Update connection status to reconnecting
          setConnectionStatus(ConnectionStatus.RECONNECTING);

          const nextAttempt = reconnectAttempts + 1;

          // Calculate delay with improved exponential backoff
          let delay: number;
          if (nextAttempt <= 3) {
            // Linear backoff for first 3 attempts (1s, 2s, 3s)
            delay = reconnectDelay * nextAttempt;
          } else {
            // Exponential backoff after that, but capped at maxReconnectDelay
            const backoffDelay = reconnectDelay * Math.pow(reconnectBackoffFactor, nextAttempt - 3);
            delay = Math.min(backoffDelay, maxReconnectDelay);
          }

          console.log(`Attempting to reconnect (${nextAttempt}/${maxReconnectAttempts}) in ${delay}ms...`);
          localStorage.setItem('websocketReconnecting', 'true');
          setReconnectAttempts(nextAttempt);

          // Store the last reconnect time
          lastReconnectTimeRef.current = Date.now();

          // Clear any existing reconnect timeout
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
          }

          // Store the timeout ID so we can clear it if needed
          reconnectTimeoutRef.current = window.setTimeout(() => {
            reconnectTimeoutRef.current = null;

            // Double-check that user is still authenticated before reconnecting
            if (user) {
              // Reset the global WebSocket to ensure a fresh connection
              if (globalWebSocket) {
                globalWebSocket.close();
                globalWebSocket = null;
              }
              localStorage.removeItem('websocketReconnecting');
              connect();
            } else {
              console.log('User no longer authenticated, canceling reconnection attempt');
              localStorage.removeItem('websocketReconnecting');
              setConnectionStatus(ConnectionStatus.ERROR);
            }
          }, delay);

          // Return a cleanup function that will be called if the component unmounts
          // before the timeout completes
          return () => {
            console.log('Cleaning up reconnection timeout');
            if (reconnectTimeoutRef.current) {
              clearTimeout(reconnectTimeoutRef.current);
              reconnectTimeoutRef.current = null;
            }
            localStorage.removeItem('websocketReconnecting');
          };
        } else if (reconnectAttempts >= maxReconnectAttempts) {
          console.log('Maximum reconnection attempts reached. Will try again when user navigates or refreshes.');
          localStorage.removeItem('websocketReconnecting');

          // Update connection status to error
          setConnectionStatus(ConnectionStatus.ERROR);

          // Reset reconnect attempts after a delay to allow future reconnection attempts
          setTimeout(() => {
            console.log('Resetting reconnect attempts after maximum attempts reached');
            setReconnectAttempts(0);

            // Try one more time after the reset
            if (user) {
              console.log('Attempting one final reconnection after reset');
              setConnectionStatus(ConnectionStatus.CONNECTING);
              connect();
            }
          }, reconnectResetDelay); // Use the configurable reset delay
        }
      };

      newSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Don't close the connection here, let the onclose handler deal with reconnection
      };

      setSocket(newSocket);
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
    }
  }, [user, reconnectAttempts]);

  /**
   * Connect/disconnect based on authentication state
   */
  useEffect(() => {
    if (user) {
      connect();
    } else if (socket) {
      // Close connection if user logs out
      console.log('Closing WebSocket connection due to user logout');
      socket.close();
      setSocket(null);
      setConnected(false);
    }

    // Cleanup on unmount
    return () => {
      // Clear all debounce timers
      debounceTimers.forEach((timerId) => {
        clearTimeout(timerId);
      });
      debounceTimers.clear();

      // Clear the recent messages array
      recentlySentMessages = [];

      if (socket) {
        console.log('Closing WebSocket connection on component unmount');
        socket.close();
        setSocket(null);
      }
    };
  }, [user, connect, socket]);

  /**
   * Handle page navigation and browser close events
   */
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (socket) {
        console.log('Closing WebSocket connection before page unload');
        // Use synchronous close for beforeunload event
        socket.close();
      }
    };

    // Add event listeners for page unload
    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [socket]);

  /**
   * Implement heartbeat mechanism to keep connection alive
   * This uses the global WebSocket instance to ensure only one heartbeat is active
   */
  useEffect(() => {
    // Only set up heartbeat if we have a global connection and we're the primary provider
    if (!isPrimaryProvider.current) return;

    // Check if heartbeat is already active
    if ((window as any).__websocketHeartbeatActive) {
      console.log('Heartbeat already active, not setting up another one');
      return;
    }

    console.log('Setting up WebSocket heartbeat');

    // Track last successful pong response time
    let lastPongTime = Date.now();

    // Track if we're waiting for a pong response
    let waitingForPong = false;

    // Set up a handler for pong responses
    const handlePongResponse = (message: WebSocketMessage) => {
      if (message.type === 'pong') {
        waitingForPong = false;
        lastPongTime = Date.now();
        // console.log('Received pong response, connection is alive');
      }
    };

    // Add the pong message listener
    const removeListener = addMessageListener('pong', handlePongResponse);

    // More frequent heartbeat during file uploads or document processing
    const getHeartbeatInterval = () => {
      const uploadInProgress = localStorage.getItem('uploadInProgress') === 'true';
      const documentProcessing = localStorage.getItem('documentProcessing') === 'true';

      if (uploadInProgress || documentProcessing) {
        console.log(`${uploadInProgress ? 'Upload' : 'Document processing'} in progress, using more frequent heartbeat interval`);
        return 5000; // Use a fixed 5-second interval during active operations for more reliable connection
      }

      return heartbeatInterval;
    };

    // Send ping every heartbeatInterval
    const pingTimer = setInterval(() => {
      // Check if we're waiting too long for a pong response (connection might be dead)
      const currentTime = Date.now();
      const timeSinceLastPong = currentTime - lastPongTime;

      // If we've been waiting for a pong for too long (3x the heartbeat interval), force reconnect
      if (waitingForPong && timeSinceLastPong > heartbeatInterval * 3) {
        console.log(`No pong response received for ${timeSinceLastPong}ms, forcing reconnection`);
        setConnectionStatus(ConnectionStatus.RECONNECTING);

        if (globalWebSocket) {
          globalWebSocket.close();
          globalWebSocket = null;
        }
        globalConnected = false;
        setConnected(false);
        waitingForPong = false;
        connect();
        return;
      }

      // Normal heartbeat logic
      if (globalWebSocket) {
        if (globalWebSocket.readyState === WebSocket.OPEN) {
          try {
            // Mark that we're waiting for a pong response
            waitingForPong = true;
            globalWebSocket.send(JSON.stringify({
              type: 'ping',
              timestamp: currentTime
            }));
          } catch (error) {
            console.error('Error sending heartbeat ping:', error);
            // If we can't send a ping, the connection might be dead
            // Force a reconnection
            if (globalWebSocket) {
              console.log('Error sending heartbeat ping, forcing reconnection');
              setConnectionStatus(ConnectionStatus.RECONNECTING);

              globalWebSocket.close();
              globalWebSocket = null;
              globalConnected = false;
              setConnected(false);
              // Trigger reconnect
              connect();
            }
          }
        } else if (globalWebSocket.readyState === WebSocket.CLOSED || globalWebSocket.readyState === WebSocket.CLOSING) {
          // If the connection is closed or closing, try to reconnect
          console.log('Heartbeat detected closed connection, attempting to reconnect');
          setConnectionStatus(ConnectionStatus.RECONNECTING);

          globalWebSocket = null;
          globalConnected = false;
          setConnected(false);
          connect();
        }
      } else if (user) {
        // If we have a user but no WebSocket, try to connect
        console.log('Heartbeat detected missing WebSocket connection, attempting to connect');
        setConnectionStatus(ConnectionStatus.CONNECTING);
        connect();
      }
    }, getHeartbeatInterval());

    // Set up a global variable to track if heartbeat is already set up
    // This prevents multiple heartbeats if multiple WebSocketProvider components are mounted
    (window as any).__websocketHeartbeatActive = true;

    return () => {
      // Only clean up if this is the last WebSocketProvider being unmounted
      console.log('Cleaning up WebSocket heartbeat');
      clearInterval(pingTimer);
      removeListener(); // Remove the pong message listener
      (window as any).__websocketHeartbeatActive = false;
    };
  }, [connect, heartbeatInterval, user]);

  /**
   * Helper function to check if a message is a duplicate
   */
  const isDuplicateMessage = (data: any): boolean => {
    const stringifiedData = JSON.stringify(data);
    const now = Date.now();

    // Clean up old messages first
    recentlySentMessages = recentlySentMessages.filter(
      msg => now - msg.timestamp < MESSAGE_DEDUPLICATION_WINDOW
    );

    // Check if this message is a duplicate
    const isDuplicate = recentlySentMessages.some(msg => msg.data === stringifiedData);

    // If not a duplicate, add it to the recent messages
    if (!isDuplicate) {
      recentlySentMessages.push({
        data: stringifiedData,
        timestamp: now
      });

      // Limit the size of the recent messages array
      if (recentlySentMessages.length > MESSAGE_DEDUPLICATION_LIMIT) {
        recentlySentMessages.shift(); // Remove the oldest message
      }
    } else {
      console.log('Duplicate message detected, not sending:', data);
    }

    return isDuplicate;
  };

  /**
   * Helper function to actually send the message
   */
  const sendMessage = (data: any) => {
    // Don't send duplicate messages
    if (isDuplicateMessage(data)) {
      return;
    }

    const stringifiedData = JSON.stringify(data);

    if (globalWebSocket && globalWebSocket.readyState === WebSocket.OPEN) {
      globalWebSocket.send(stringifiedData);
    } else if (socket && connected) {
      // Fallback to local socket if global is not available
      socket.send(stringifiedData);
    } else {
      console.warn('Cannot send message: WebSocket is not connected');
    }
  };

  /**
   * Send data through the WebSocket connection
   * Uses the global WebSocket instance to ensure messages are sent through the single connection
   * Implements debouncing for certain message types to prevent rapid duplicate messages
   */
  const send = useCallback((data: any) => {
    // Special handling for different message types
    if (data && typeof data === 'object' && data.type) {
      // For connection-related messages, don't debounce
      if (data.type === 'mcp_connect' || data.type === 'mcp_disconnect') {
        console.log(`Sending ${data.type} message immediately (no debounce)`);
        sendMessage(data);
        return;
      }

      // For initialization messages, use debouncing
      if (data.type === 'mcp_init' || data.type === 'mcp_agent_init') {
        // Clear any existing timer for this message type
        if (debounceTimers.has(data.type)) {
          clearTimeout(debounceTimers.get(data.type));
        }

        // Set a new timer
        const timerId = window.setTimeout(() => {
          console.log(`Sending debounced ${data.type} message`);
          sendMessage(data);
          debounceTimers.delete(data.type);
        }, DEFAULT_DEBOUNCE_DELAY);

        debounceTimers.set(data.type, timerId);
        return;
      }
    }

    // For all other messages, send immediately
    sendMessage(data);
  }, [socket, connected]);

  /**
   * Add a listener for a specific message type
   */
  const addMessageListener = useCallback((type: string, callback: (message: WebSocketMessage) => void) => {
    // Update the global message listeners
    if (!globalMessageListeners.has(type)) {
      globalMessageListeners.set(type, new Set());
    }

    globalMessageListeners.get(type)?.add(callback);

    // Update the local state to trigger re-renders
    setMessageListeners(new Map(globalMessageListeners));

    // Return a function to remove this listener
    return () => {
      if (globalMessageListeners.has(type)) {
        const listeners = globalMessageListeners.get(type);
        if (listeners) {
          listeners.delete(callback);

          if (listeners.size === 0) {
            globalMessageListeners.delete(type);
          }

          // Update the local state to trigger re-renders
          setMessageListeners(new Map(globalMessageListeners));
        }
      }
    };
  }, []);

  /**
   * Wait for WebSocket to be fully ready
   * Returns a promise that resolves when the WebSocket is ready
   */
  const waitForReady = useCallback((timeoutMs: number = 10000): Promise<void> => {
    return new Promise((resolve, reject) => {
      // If already ready, resolve immediately
      if (isFullyReady) {
        console.log('WebSocket already fully ready, resolving immediately');
        return resolve();
      }

      console.log(`Setting up waitForReady promise with ${timeoutMs}ms timeout`);

      // Set up timeout to reject the promise if it takes too long
      const timeoutId = window.setTimeout(() => {
        console.error(`WebSocket readiness timeout after ${timeoutMs}ms`);

        // Clean up the resolver ref
        readyPromiseResolverRef.current = null;

        // Reject with a timeout error
        reject(new Error(`Timed out waiting for WebSocket to be ready after ${timeoutMs}ms`));
      }, timeoutMs);

      // Store the resolver functions
      readyPromiseResolverRef.current = {
        resolve: () => {
          console.log('WebSocket ready promise resolved');
          resolve();
        },
        reject,
        timeoutId
      };
    });
  }, [isFullyReady]);

  /**
   * Force a reconnection of the WebSocket
   */
  const reconnect = useCallback(() => {
    console.log('Manually reconnecting WebSocket...');

    // Update connection status to reconnecting
    setConnectionStatus(ConnectionStatus.RECONNECTING);

    // Only allow the primary provider to reconnect
    if (!isPrimaryProvider.current) {
      console.log('Not reconnecting WebSocket: Not the primary provider');
      return;
    }

    // Implement connection throttling to prevent excessive reconnections
    const now = Date.now();
    if (now - connectionThrottleRef.current < connectionThrottleDelay) {
      console.log(`Reconnection throttled: Last attempt was ${(now - connectionThrottleRef.current) / 1000}s ago, waiting...`);

      // Schedule a reconnection after the throttle delay
      setTimeout(() => {
        console.log('Executing delayed reconnection after throttle period');
        reconnect();
      }, connectionThrottleDelay);

      return;
    }

    // Update the throttle timestamp
    connectionThrottleRef.current = now;

    // Store the last reconnect time
    lastReconnectTimeRef.current = Date.now();

    // Reset all connection state refs
    setIsFullyReady(false);
    readyStateRef.current = false;
    connectionEstablishedRef.current = false;
    messageReceivedRef.current = false;

    // Reject any pending waitForReady promises
    if (readyPromiseResolverRef.current) {
      console.log('Rejecting pending waitForReady promise due to reconnection');
      readyPromiseResolverRef.current.reject(new Error('WebSocket reconnection initiated'));

      // Clear any timeout
      if (readyPromiseResolverRef.current.timeoutId) {
        clearTimeout(readyPromiseResolverRef.current.timeoutId);
      }

      readyPromiseResolverRef.current = null;
    }

    // Log detailed reconnection attempt
    console.log('Resetting all WebSocket connection state for reconnection');

    // Clear any existing reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close existing connection if any
    if (globalWebSocket) {
      console.log('Closing existing WebSocket connection for manual reconnect');
      globalWebSocket.close();
      globalWebSocket = null;
    }

    // Reset connection state
    globalConnected = false;
    setConnected(false);

    // Reset reconnect attempts for manual reconnect
    // This allows a fresh start for the reconnection process
    setReconnectAttempts(0);

    // Trigger reconnect
    connect();
  }, [connect, connectionThrottleDelay]);

  return (
    <WebSocketContext.Provider
      value={{
        connected,
        isFullyReady,
        connectionStatus,
        reconnectAttempts,
        send,
        lastMessage,
        addMessageListener,
        reconnect,
        waitForReady
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
};

/**
 * Hook to use the WebSocket context
 */
export const useWebSocket = () => {
  const context = useContext(WebSocketContext);

  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }

  return context;
};

export default WebSocketContext;

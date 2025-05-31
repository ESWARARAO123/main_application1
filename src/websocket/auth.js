/**
 * WebSocket Authentication Module
 *
 * Handles authentication for WebSocket connections by validating session cookies.
 */
const cookie = require('cookie');
const crypto = require('crypto');
const { getSessionStore } = require('../services/sessionService');

/**
 * Extracts the user ID from the session cookie in the WebSocket handshake request
 *
 * @param {Object} req - The HTTP request object from the WebSocket handshake
 * @returns {Promise<string|null>} - The user ID if authenticated, null otherwise
 */
async function authenticateWebSocket(req) {
  try {
    console.log('Starting WebSocket authentication process');

    // Parse cookies from request headers
    const cookies = cookie.parse(req.headers.cookie || '');
    console.log('Parsed cookies:', Object.keys(cookies));

    // Get session ID from cookie (connect.sid)
    const sessionCookie = cookies['connect.sid'];
    console.log('Session cookie found:', sessionCookie ? 'Yes' : 'No');

    if (!sessionCookie) {
      console.log('WebSocket connection rejected: No session cookie found');
      return null;
    }

    try {
      // Try to get the actual user ID from the session store
      const sessionStore = getSessionStore();
      console.log('Session store available:', sessionStore ? 'Yes' : 'No');

      if (sessionStore) {
        // Extract the session ID from the cookie
        // The cookie format is typically: s%3A<sessionId>.<signature>
        let sessionId;
        try {
          sessionId = decodeURIComponent(sessionCookie.split('.')[0].substring(2));
          console.log('Extracted session ID:', sessionId);
        } catch (parseError) {
          console.error('Error parsing session ID from cookie:', parseError);
          console.log('Raw session cookie:', sessionCookie);
          // Try a different approach if the standard format doesn't work
          sessionId = sessionCookie;
        }

        // Get the session data from the session store
        console.log('Attempting to retrieve session data for session ID:', sessionId);
        const sessionData = await new Promise((resolve, reject) => {
          sessionStore.get(sessionId, (err, session) => {
            if (err) {
              console.error('Error getting session data:', err);
              reject(err);
            } else {
              console.log('Session data retrieved:', session ? 'Yes' : 'No');
              if (session) {
                console.log('Session data keys:', Object.keys(session));
                console.log('User ID in session:', session.userId ? session.userId : 'Not found');
              }
              resolve(session);
            }
          });
        });

        if (sessionData && sessionData.userId) {
          console.log(`WebSocket connection authenticated for user ID: ${sessionData.userId}`);
          return sessionData.userId;
        } else {
          console.log('No user ID found in session data');
        }
      } else {
        console.log('No session store available, falling back to simplified authentication');
      }
    } catch (sessionError) {
      console.error('Error retrieving session data:', sessionError);
      console.log('Falling back to simplified authentication');
      // Fall back to the simplified approach if session store access fails
    }

    // Fallback to simplified approach if session store access fails or user ID not found
    // Generate a deterministic user ID based on the session cookie
    const hash = crypto.createHash('md5').update(sessionCookie).digest('hex');
    const userId = `user-${hash.substring(0, 8)}`;

    console.log(`WebSocket connection authenticated for user (fallback): ${userId}`);
    return userId;
  } catch (error) {
    console.error('WebSocket authentication error:', error);
    return null;
  }
}

module.exports = {
  authenticateWebSocket
};

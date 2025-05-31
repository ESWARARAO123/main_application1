/**
 * Session Service
 * 
 * Provides access to the session store for authentication and user management.
 */

// Global reference to the session store
let sessionStore = null;

/**
 * Set the session store reference
 * This should be called during application initialization
 * 
 * @param {Object} store - Express session store instance
 */
function setSessionStore(store) {
  sessionStore = store;
  console.log('Session store reference set in sessionService');
}

/**
 * Get the session store reference
 * 
 * @returns {Object|null} - The session store or null if not set
 */
function getSessionStore() {
  return sessionStore;
}

/**
 * Get session data by session ID
 * 
 * @param {string} sessionId - The session ID
 * @returns {Promise<Object|null>} - The session data or null if not found
 */
async function getSessionData(sessionId) {
  if (!sessionStore) {
    console.warn('Session store not available');
    return null;
  }

  return new Promise((resolve, reject) => {
    sessionStore.get(sessionId, (err, session) => {
      if (err) {
        reject(err);
      } else {
        resolve(session);
      }
    });
  });
}

/**
 * Get user ID from session
 * 
 * @param {string} sessionId - The session ID
 * @returns {Promise<string|null>} - The user ID or null if not found
 */
async function getUserIdFromSession(sessionId) {
  try {
    const session = await getSessionData(sessionId);
    return session && session.userId ? session.userId : null;
  } catch (error) {
    console.error('Error getting user ID from session:', error);
    return null;
  }
}

module.exports = {
  setSessionStore,
  getSessionStore,
  getSessionData,
  getUserIdFromSession
};

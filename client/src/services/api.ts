import axios from 'axios';

// Create axios instance with defaults
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Important for session cookies
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle session timeout or unauthorized access
    if (error.response && error.response.status === 401) {
      // Check if the error was from the /auth/me endpoint
      const isAuthMeRequest = error.config.url === '/auth/me';
      
      // Only redirect if it's NOT the initial /auth/me check failing
      if (!isAuthMeRequest) {
        console.log('Redirecting to login due to 401 on non-/auth/me endpoint');
        window.location.href = '/login';
      } else {
        // For /auth/me 401, just let the AuthContext handle setting user to null
        console.log('/auth/me returned 401, user is likely not logged in.');
      }
    }
    return Promise.reject(error);
  }
);

export { api }; 
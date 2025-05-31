// Read the backend URL from environment or use a default
const backendUrl = process.env.REACT_APP_API_URL || '/api';

export const config = {
  apiBaseUrl: backendUrl
}; 
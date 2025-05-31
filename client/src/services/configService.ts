import { api } from './api';

export interface AppConfig {
  title: string;
  appName: string;
  apiUrl: string;
  defaultTheme: 'light' | 'dark';
}

// Default configuration in case API call fails
const defaultConfig: AppConfig = {
  title: 'Product Demo',
  appName: 'Product Demo',
  apiUrl: '/api',
  defaultTheme: 'light'
};

// Store the loaded configuration to avoid multiple API calls
let loadedConfig: AppConfig | null = null;

/**
 * Loads the application configuration from the backend
 * Caches the result to avoid multiple API calls
 */
export async function loadConfig(): Promise<AppConfig> {
  // Return cached config if available
  if (loadedConfig) {
    return loadedConfig;
  }

  try {
    // Fetch configuration from backend
    const response = await api.get('/frontend-config');
    loadedConfig = { ...defaultConfig, ...response.data };
    return loadedConfig;
  } catch (error) {
    console.error('Error loading configuration from backend:', error);
    return defaultConfig;
  }
}

/**
 * Get current configuration synchronously
 * Returns the cached config or default if not loaded yet
 */
export function getConfig(): AppConfig {
  return loadedConfig || defaultConfig;
} 
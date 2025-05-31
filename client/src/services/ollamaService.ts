import { api } from './api';

export interface OllamaSettings {
  host: string;
  port: number;
  default_model: string;
}

export interface OllamaModel {
  id: string;
  ollama_model_id: string;
  name: string;
  description: string;
  parameters: any;
  is_active: boolean;
}

export interface ConnectionTestResult {
  success: boolean;
  message: string;
  responseTime?: number;
  version?: string;
  error?: string;
}

export interface SyncResult {
  added: number;
  updated: number;
  unchanged: number;
  total: number;
  inactivated?: number;
}

// Get Ollama settings
export const getOllamaSettings = async (): Promise<OllamaSettings> => {
  const response = await api.get('/ollama/settings');
  return response.data;
};

// Save Ollama settings
export const saveOllamaSettings = async (settings: OllamaSettings): Promise<OllamaSettings> => {
  const response = await api.put('/ollama/settings', settings);
  return response.data;
};

// Test connection to Ollama server
export const testOllamaConnection = async (host: string, port: number): Promise<ConnectionTestResult> => {
  const response = await api.post('/ollama/test-connection', { host, port });
  return response.data;
};

// Get models available on Ollama server
export const getAvailableOllamaModels = async (): Promise<OllamaModel[]> => {
  const response = await api.get('/ollama/models/available');
  return response.data.models;
};

// Sync models from Ollama server to database
export const syncOllamaModels = async (selectedModelIds?: string[]): Promise<SyncResult> => {
  const response = await api.post('/ollama/models/sync', { selectedModelIds });
  return response.data;
};

// Get all models from database
export const getOllamaModelsFromDB = async (): Promise<OllamaModel[]> => {
  const response = await api.get('/ollama/models/db');
  return response.data.models;
};

// Get active models from database
export const getActiveOllamaModels = async (): Promise<OllamaModel[]> => {
  const response = await api.get('/ollama/models/active');
  return response.data.models;
};

// Toggle model active status
export const toggleOllamaModelStatus = async (id: string, isActive: boolean): Promise<OllamaModel> => {
  const response = await api.put(`/ollama/models/${id}/status`, { is_active: isActive });
  return response.data;
};

// Check Ollama API status
export const checkOllamaStatus = async (): Promise<ConnectionTestResult> => {
  const response = await api.get('/ollama/status');
  return response.data;
}; 
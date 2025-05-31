# Ollama Direct Connection Guide

## Overview

This guide explains the new direct connection feature which allows you to connect to Ollama servers directly from your browser, bypassing the need for a working backend server.

## Key Features

1. **Direct Browser-to-Ollama Connection**: Connect straight to Ollama without needing the backend API
2. **Local Storage Persistence**: Save settings and models locally in your browser
3. **Offline Mode**: Work with cached models when backend or Ollama servers are unavailable
4. **CORS-Aware**: Handles cross-origin resource sharing issues common in browser-to-server connections

## Getting Started

### Step 1: Set Your Ollama Host & Port

1. Navigate to Settings > AI Settings
2. Enter your Ollama server host (e.g., `localhost`, `172.16.16.26`, etc.)
3. Enter your Ollama server port (default: `11434`)
4. Enter your default model (e.g., `llama3`)

### Step 2: Test Direct Connection

1. Click the "Direct Test" button in the debugging tools section
2. This will attempt to connect directly to Ollama from your browser
3. If successful, you'll see models listed and they'll be saved to your browser's local storage

### Step 3: Save Settings

1. After a successful direct connection, click "Save Settings"
2. This will save the settings to both local storage and attempt to save to the backend

### Step 4: Sync Models

1. After connecting, click "Sync Models" to save the fetched models for later use
2. This will attempt to sync with the database, but will fall back to local storage if the backend is unavailable

## Troubleshooting Common Issues

### CORS Errors

If you see network errors in the console with messages about CORS (Cross-Origin Resource Sharing), you have options:

1. **Enable CORS on your Ollama server** - Running Ollama with the appropriate CORS headers
2. **Access your Ollama server through a CORS proxy** - Either a local proxy or a service

### Connection Refused

If the connection is refused:

1. Verify Ollama is running on the specified host and port
2. Check for firewalls or network restrictions
3. Try using IP address `127.0.0.1` instead of `localhost` if on the same machine
4. If using Docker, ensure proper network configuration

### Unknown Host

If the host cannot be resolved:

1. Check the spelling of the hostname
2. Try using an IP address instead
3. Verify DNS resolution is working

## Working Offline

Once you've successfully connected and synced models:

1. Models are saved to your browser's local storage
2. Settings are saved to local storage
3. You can view and work with these models even when the backend or Ollama servers are offline
4. The app will automatically use locally saved data when servers are unreachable

## For Developers

The direct connection feature is implemented using:

1. Browser's native `fetch` API for direct requests
2. `localStorage` API for persistence
3. Error handling specific to browser networking constraints
4. Connection test that reports detailed error information

This implementation is located in `client/src/components/settings/DirectOllamaConnection.tsx` and can be extended as needed.

## Security Considerations

Be aware that:

1. Data saved to local storage persists between sessions
2. Connections made directly from the browser may expose your Ollama server to the internet if not properly secured
3. Consider using this feature only on trusted networks or with appropriate access controls 
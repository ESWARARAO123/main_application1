# Ollama Integration Troubleshooting Guide

This guide helps diagnose and fix common issues with the Ollama AI integration in the Platform Dashboard.

## Common Issues

1. **Connection Failures**
   - "Connection test failed" in the admin UI
   - Unable to save Ollama server settings
   - Cannot sync models from Ollama to the database

2. **Model Synchronization Issues**
   - Models appear on the Ollama server but not in the Platform Dashboard
   - "Failed to sync models" error message
   - Empty model list in the AI Model Management section

3. **API Communication Issues**
   - Frontend to backend API errors
   - Backend to Ollama server connection problems
   - CORS or network configuration issues

## Diagnostic Steps

### Step 1: Verify Ollama Server is Running

1. Open a terminal and run:
   ```
   curl http://localhost:11434/api/tags
   ```

2. You should receive a JSON response with available models
   ```json
   {"models":[{"name":"llama3:latest","model":"llama3:latest","modified_at":"2025-04-16T15:56:21.6797515+05:30","size":4661224676,"digest":"365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad...","details":{"parent_model":"","format":"gguf","family":"llama","families":["llama"],"parameter_size":"8B","quantization_level":"Q5_K_M"}}]}
   ```

3. If you get an error or no response, your Ollama server is not running or accessible

### Step 2: Run the Diagnostic Script

1. Run the diagnostic script from the project root:
   ```
   node scripts/test-ollama-connection.js
   ```

2. For a different host/port:
   ```
   node scripts/test-ollama-connection.js 172.16.16.26 11434
   ```

3. Review the output and follow the recommendations

### Step 3: Fix Database Settings

If the diagnostic tool shows that your database settings are incorrect, you can fix them using the provided SQL script:

1. Run the fix script:
   ```
   psql -U postgres -d copilot -f src/scripts/sql/fix_ollama_settings.sql
   ```

2. Alternatively, use the admin UI to update the settings:
   - Navigate to Settings > Ollama AI Integration
   - Update the host to "localhost" (or your correct host)
   - Update the port to "11434" (or your correct port)
   - Update the default model to match an available model on your server

### Step 4: Verify Server Configuration

1. Check the `conf/config.ini` file to ensure the database connection is correct:
   ```ini
   [database]
   database-type = postgres
   database-host = 172.16.16.26
   database-port = 5432
   database-user = postgres
   database-password = root
   database-name = copilot
   ```

2. Make sure your Vite proxy configuration in `client/vite.config.ts` includes:
   ```js
   '/api/ollama': {
     target: `${config.server.protocol}://${config.server.domain}:${config.server.port}`,
     changeOrigin: true,
     secure: config.security.strict_transport_security === 'true'
   }
   ```

### Step 5: Restart the Application

1. Stop both the frontend and backend servers
2. Start the backend:
   ```
   npm run server
   ```
3. Start the frontend in a new terminal:
   ```
   npm run client
   ```

## Advanced Troubleshooting

### Network Connectivity Issues

If you're using different machines for the Platform Dashboard and Ollama server:

1. Ensure the Ollama machine allows incoming connections on port 11434
2. Check for firewalls or network restrictions
3. Try using an IP address instead of 'localhost' if they're on different machines
4. Verify the Ollama server is binding to all interfaces, not just localhost

### CORS Issues

If you see CORS errors in the browser console:

1. Ensure the Ollama server allows cross-origin requests
2. Check that your proxy settings in `vite.config.ts` are correct
3. Test with browser CORS restrictions disabled temporarily (for testing only)

### Database Connectivity

If the diagnostic script shows database connection issues:

1. Verify PostgreSQL is running
2. Check the connection settings in `conf/config.ini`
3. Ensure the database user has proper permissions
4. Test direct database connection with:
   ```
   psql -U postgres -h 172.16.16.26 -d copilot
   ```

## If All Else Fails

1. Reset the Ollama settings to defaults:
   ```sql
   -- Run in psql
   DELETE FROM ollama_settings;
   INSERT INTO ollama_settings (host, port, default_model) 
   VALUES ('localhost', 11434, 'llama3');
   ```

2. Clear the browser cache and cookies for the application

3. Check server logs for more detailed error messages:
   ```
   npm run server -- --verbose
   ```

4. Contact the development team with your diagnostic script output and detailed error messages 
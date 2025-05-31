# Migration Plan: Unified Backend-Served Frontend (Grafana-Style)

## Objective
Transform the application to serve the frontend directly from the backend, remove Vite, and centralize configuration using a `.ini` file, enabling single-port deployment similar to Grafana.

## Rationale
- Simplifies deployment and configuration.
- Reduces resource usage (one process, one port).
- Centralizes all settings for easier management.
- Aligns with proven patterns used by platforms like Grafana.

## Migration Steps

### Phase 1: Client Build Process Migration ✅ (Completed 2023-06-10)

1. ✅ Configure the client build to use React Scripts instead of Vite
   - ✅ Create a proper entry point (index.js) for React Scripts
   - ✅ Set up the public directory with necessary files (index.html, manifest.json)
   - ✅ Update environment variable usage to work with React Scripts
   - ✅ Fix TypeScript configuration to handle the build process
   - ✅ Successfully build the client with React Scripts

### Phase 2: Server Configuration ✅ (Completed 2023-06-10)

1. ✅ Configure the server to serve static files
   - ✅ Update the server to serve the client/build directory for static assets
   - ✅ Implement proper handling for SPA routing (serve index.html for client routes)
   - ✅ Ensure proper MIME type handling for all asset types

2. ✅ Update API endpoints
   - ✅ Prefix all API endpoints with `/api` to distinguish from static assets
   - ✅ Update client API service to use the new prefixed endpoints
   - ✅ Create frontend configuration API endpoint
   - ✅ Create frontend configuration service

### Phase 3: Configuration Centralization (In Progress - Started 2023-06-10)

1. Finalize centralized configuration
   - Update config.ini file with all needed settings
   - Ensure all services use the centralized configuration
   - Validate that configuration flows correctly from backend to frontend

## Detailed Implementation Steps

### 1. Remove Vite and Related Dev Dependencies

**Files to modify:**
- `client/package.json`: Remove Vite dependencies and related scripts
- `client/vite.config.js` and `client/vite.config.ts`: Delete these files
- Root `package.json`: Update scripts to remove Vite-related commands

**Example changes:**
```diff
// client/package.json
{
  "dependencies": {
-   "vite": "^4.5.0",
-   "@vitejs/plugin-react": "^4.0.0",
    "react": "^18.2.0",
    // other dependencies remain
  },
  "scripts": {
-   "dev": "vite",
-   "build": "vite build",
+   "build": "react-scripts build",
    // other scripts updated
  }
}
```

### 2. Build Frontend as Static Files

**Files to modify:**
- `client/package.json`: Update build script
- Root `package.json`: Add script to build frontend during deployment

**Implementation steps:**
1. Install react-scripts if not already present
2. Configure build output to `client/dist/` or `client/build/`
3. Create a build script in the root package.json

**Example:**
```json
// Root package.json - new script
"scripts": {
  "build:client": "cd client && npm run build",
  "build:all": "npm run build:client && npm run build:server"
}
```

### 3. Update Backend to Serve Static Files

**Files to modify:**
- `src/server.js`: Add middleware to serve static files

**Implementation steps:**
1. Add Express static middleware
2. Add client-side routing fallback
3. Remove CORS configuration for same-origin requests

**Example code:**
```javascript
// src/server.js - Add this code
const path = require('path');

// Serve static files from the React app build directory
const staticPath = config.server.static_root_path || path.join(__dirname, '../client/dist');
app.use(express.static(staticPath));

// For any request that doesn't match an API route or static file, 
// send the React app's index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(staticPath, 'index.html'));
});
```

### 4. Centralize Configuration in `conf/config.ini`

**Files to modify:**
- `conf/config.ini`: Add new sections for unified configuration
- `src/server.js`: Update to read all settings from config.ini
- `client/.env`: Remove in favor of injecting values from backend

**New configuration sections:**
```ini
[paths]
data = ./data
logs = ./logs
static_files = ./client/dist

[server]
protocol = http
domain = localhost
port = 5634
static_root_path = ./client/dist

[frontend]
title = Platform Dashboard
api_url = /api
default_theme = dark
```

### 5. Update Backend to Read from `.ini` File

**Files to modify:**
- `src/server.js`: Enhance config reading
- Create a new utility file `src/services/config.js`

**Example implementation:**
```javascript
// src/services/config.js
const fs = require('fs');
const ini = require('ini');
const path = require('path');

function loadConfig(configPath) {
  try {
    const configContent = fs.readFileSync(configPath, 'utf-8');
    return ini.parse(configContent);
  } catch (error) {
    console.error(`Error loading config from ${configPath}:`, error);
    process.exit(1);
  }
}

module.exports = { loadConfig };
```

```javascript
// src/server.js - Update config initialization
const { loadConfig } = require('./services/config');
const configPath = process.argv.find(arg => arg.startsWith('--config=')).split('=')[1] || './conf/config.ini';
const config = loadConfig(configPath);
```

### 6. Inject Configuration to Frontend

**Files to create:**
- `src/routes/config.js`: New API endpoint to provide frontend config
- `client/src/services/config.ts`: New service to fetch config from backend

**Example implementation:**
```javascript
// src/routes/config.js
const express = require('express');
const router = express.Router();

module.exports = function(config) {
  router.get('/frontend-config', (req, res) => {
    // Send only necessary frontend configuration
    res.json({
      title: config.frontend.title,
      apiUrl: config.frontend.api_url,
      defaultTheme: config.frontend.default_theme
    });
  });
  
  return router;
};
```

```typescript
// client/src/services/config.ts
export async function loadConfig() {
  const response = await fetch('/api/frontend-config');
  if (!response.ok) {
    throw new Error('Failed to load application configuration');
  }
  return response.json();
}
```

### 7. Update API Client in Frontend

**Files to modify:**
- `client/src/services/api.ts`: Remove hardcoded baseURL

**Example implementation:**
```typescript
// client/src/services/api.ts
import axios from 'axios';

// Create API client with relative URLs instead of hardcoded baseURL
const api = axios.create({
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json'
  }
});

export default api;
```

### 8. Update Deployment/Start Scripts

**Files to modify:**
- Root `package.json`: Update scripts for the new deployment model

**Example implementation:**
```json
{
  "scripts": {
    "start": "node src/server.js --config=./conf/config.ini",
    "build": "cd client && npm run build",
    "deploy": "npm run build && npm run start",
    "dev": "nodemon src/server.js --config=./conf/config.ini",
    "dev:client": "cd client && npm run build -- --watch"
  }
}
```

### 9. Clean Up Obsolete Code and Documentation

**Files to modify:**
- `README.md`: Update with new deployment and configuration information
- All Vite-specific files and references

**Areas to update:**
- Running instructions
- Deployment instructions
- Configuration documentation
- Development workflow

## Expected Outcomes
- The entire app is served from a single backend process and port.
- All configuration is managed via `conf/config.ini`.
- The frontend is built once and served as static files by the backend.
- No Vite or separate frontend dev server remains. 

## Testing
After implementation, test the following:
1. Static assets are properly served from backend
2. SPA routing works correctly
3. API requests work without CORS issues
4. Configuration changes in config.ini properly affect both frontend and backend 
# MCP Integration Implementation Progress Tracker

## Overview
This document tracks the implementation of Model Context Protocol (MCP) integration into our application. MCP will allow the AI to interact with the user's environment through a secure, controlled interface. This integration needs to be completed in phases with careful attention to security and user experience.

## Implementation Phases

### Phase 1: Database Schema Creation
- [ ] **Create database migration scripts:**
  - [ ] Create `user_ssh_configurations` table migration in `src/migrations/{timestamp}_create_user_ssh_configurations.js`
  - [ ] Create `user_mcp_server_configurations` table migration in `src/migrations/{timestamp}_create_user_mcp_server_configurations.js`
  - [ ] Add appropriate indexes and constraints
  - [ ] Update `DatabaseStructure.md` with new table definitions

#### Database Schema Details
```sql
-- Enable the uuid-ossp extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- user_ssh_configurations table
CREATE TABLE IF NOT EXISTS user_ssh_configurations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  machine_nickname VARCHAR(255) NOT NULL,
  ssh_host VARCHAR(255) NOT NULL,
  ssh_port INTEGER NOT NULL DEFAULT 22,
  ssh_user VARCHAR(255) NOT NULL,
  ssh_auth_method VARCHAR(50) NOT NULL,
  ssh_password_encrypted TEXT,
  ssh_key_path TEXT,
  last_ssh_connection_status VARCHAR(50) DEFAULT 'unknown',
  last_ssh_error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, machine_nickname)
);

CREATE INDEX idx_user_ssh_configurations_user_id ON user_ssh_configurations(user_id);

-- user_mcp_server_configurations table
CREATE TABLE IF NOT EXISTS user_mcp_server_configurations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  ssh_configuration_id UUID REFERENCES user_ssh_configurations(id) ON DELETE SET NULL,
  mcp_nickname VARCHAR(255) NOT NULL,
  mcp_host VARCHAR(255) NOT NULL,
  mcp_port INTEGER NOT NULL,
  mcp_connection_status VARCHAR(50) DEFAULT 'unknown',
  mcp_last_error_message TEXT,
  mcp_discovered_tools_schema JSONB,
  mcp_server_version VARCHAR(100),
  is_default BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, mcp_nickname)
);

CREATE INDEX idx_user_mcp_server_configurations_user_id ON user_mcp_server_configurations(user_id);
CREATE INDEX idx_user_mcp_server_configurations_ssh_config_id ON user_mcp_server_configurations(ssh_configuration_id);
```

### Phase 2: Backend Services Implementation
- [x] **Create core MCP service:**
  - [x] Create `src/services/mcpService.js` for MCP connection management with the following functions:
    - `connectToMCP` - Connect to an MCP server
    - `disconnectFromMCP` - Disconnect from an MCP server
    - `getMCPStatus` - Get MCP connection status
    - `discoverTools` - Discover available MCP tools
    - `executeTool` - Execute an MCP tool
    - `getActiveConnections` - Get active MCP connections
    - `testMCPConnection` - Test connection to an MCP server
  - [x] Add EventSource for Server-Sent Events (SSE) connection
  - [x] Implement timeout and reconnection logic
  - [x] Add proper error handling and logging

- [x] **Create SSH management service:**
  - [x] Create `src/services/sshService.js` with the following functions:
    - `testSSHConnection` - Test SSH connection
    - `executeSSHCommand` - Execute command over SSH
    - `installMCPViaSSH` - Install MCP via SSH
    - `encryptPassword` - Secure password encryption
    - `decryptPassword` - Secure password decryption
  - [x] Add secure password storage with encryption
  - [x] Implement proper connection validation
  - [x] Add proper error handling and logging

- [x] **Create database access layer:**
  - [x] Create `src/services/mcpDBService.js` with the following functions:
    - SSH Configuration CRUD operations
    - MCP Server Configuration CRUD operations
    - Status update functions
  - [x] Add transaction support for critical operations
  - [x] Implement proper data validation
  - [x] Add proper error handling and logging

- [x] **Create MCP API routes:**
  - [x] Create `src/routes/mcp.js` to handle API requests for:
    - SSH Configuration management
    - MCP Server configuration management
    - MCP connection and status
    - Tool discovery and execution
    - Test MCP connection
    - [x] Fix API endpoint parameters mismatch between frontend and backend for test-connection
    - [x] Update MCP connection testing to use /info endpoint instead of /health for more reliable testing
  - [x] Add authentication/authorization middleware
  - [x] Implement proper request validation
  - [x] Add proper error handling and logging

- [x] **Update server.js:**
  - [x] Add MCP API routes to express app
  - [x] Configure MCP API endpoints under '/api/mcp'
  - [x] Add MCP pages routes under '/settings/mcp'

### Phase 3: Frontend Components Creation
- [x] **Create MCP settings component:**
  - [x] Create `client/src/components/settings/MCPSettings.tsx` - Main MCP settings component
  - [x] Implement connection testing UI
  - [x] Add server management interface
  - [x] Create tool discovery and display

- [x] **Create SSH configuration components:**
  - [x] Create `client/src/components/settings/SSHConfigForm.tsx` - SSH configuration form
  - [x] Create `client/src/components/settings/SSHConfigList.tsx` - List of SSH configurations
  - [x] Implement secure credential handling on frontend

- [ ] **Create MCP installation components:**
  - [ ] Create `client/src/components/settings/MCPInstaller.tsx` with:
    - Installation step progress indicator
    - SSH connection selector
    - Installation command customization
    - Installation progress display
  - [ ] Create `client/src/components/settings/MCPInstallStatus.tsx` with:
    - Real-time installation status updates
    - Error display and troubleshooting guidance
    - Success confirmation and next steps

- [x] **Create MCP chat integration components:**
  - [x] Create `client/src/components/chat/MCPCommandApproval.tsx` - Command approval UI
  - [x] Create `client/src/components/chat/MCPToolResult.tsx` - Tool result display

### Phase 4: Frontend Services and Context
- [x] **Create MCP context:**
  - [x] Create `client/src/contexts/MCPContext.tsx` for app-wide MCP state management
  - [x] Implement connection status tracking
  - [x] Add tool availability state
  - [x] Create context provider in app hierarchy

- [ ] **Create frontend services:**
  - [ ] Create `client/src/services/mcpService.ts` with the following methods:
    ```typescript
    // Main methods to implement
    async function getMCPStatus(): Promise<MCPStatusResponse>
    async function getMCPTools(): Promise<MCPToolsResponse>
    async function testMCPConnection(config: MCPConnectionConfig): Promise<MCPTestResponse>
    async function saveMCPConnection(config: MCPConnectionConfig): Promise<MCPSaveResponse>
    async function deleteMCPServer(serverId: string): Promise<MCPDeleteResponse>
    async function executeMCPCommand(command: string, serverId: string): Promise<MCPExecuteResponse>
    ```
  - [ ] Create `client/src/services/sshService.ts` with the following methods:
    ```typescript
    // Main methods to implement
    async function getSSHConfigurations(): Promise<SSHConfigResponse>
    async function saveSSHConfiguration(config: SSHConfig): Promise<SSHSaveResponse>
    async function testSSHConnection(config: SSHTestConfig): Promise<SSHTestResponse>
    async function deleteSSHConfiguration(configId: string): Promise<SSHDeleteResponse>
    async function installMCPViaSSH(sshId: string, options?: InstallOptions): Promise<InstallResponse>
    ```
  - [ ] Implement proper TypeScript interfaces for all API responses
  - [ ] Set up Axios interceptors for error handling
  - [ ] Add proper error types and handling

### Phase 5: Settings Page Integration
- [x] **Add MCP tab to settings page:**
  - [x] Modify `client/src/pages/Settings.tsx` to include MCP tab
  - [x] Import the newly created MCP settings components
  - [x] Add icon and label for MCP settings

```tsx
// Example addition to Settings.tsx tabs array
const tabs = [
  { id: 'profile', name: 'Profile', icon: UserIcon },
  { id: 'appearance', name: 'Appearance', icon: PaintBrushIcon },
  { id: 'ollama', name: 'Ollama AI', icon: ServerIcon },
  { id: 'mcp', name: 'MCP Integration', icon: CpuChipIcon },
];

// Example content section to add
{activeTab === 'mcp' && (
  <div>
    <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--color-text)' }}>
      MCP Integration Settings
    </h2>
    <MCPSettings isAdmin={user?.role === 'admin'} />
  </div>
)}
```

### Phase 6: Chat Interface Integration
- [ ] **Implement MCP toggle in chat interface:**
  - [ ] Add MCP toggle button in `client/src/components/chat/ChatHeader.tsx`
  - [ ] Connect toggle to MCPContext's toggleMCPEnabled function
  - [ ] Add visual indicators for MCP connection status in chat interface
  - [ ] Add tooltip with connection details

- [ ] **Implement command approval workflow:**
  - [x] Add command parsing in chat messages
  - [x] Create approval UI for commands
  - [ ] Modify `client/src/components/chat/ChatInterface.tsx` to detect command requests
  - [ ] Implement command extraction from AI responses
  - [ ] Connect MCPCommandApproval component to MCPContext
  - [ ] Implement approval/rejection handling with callbacks
  - [ ] Add command history tracking

- [ ] **Implement result display:**
  - [ ] Create special message type for MCP results in `client/src/types/chat.ts`
  - [ ] Update `client/src/components/chat/Message.tsx` to render MCP tool results
  - [ ] Add formatting for different result types (text, JSON, error)
  - [ ] Implement error handling for failed commands
  - [ ] Add copy-to-clipboard functionality for results

### Phase 7: Security Audit and Testing
- [ ] **Security review:**
  - [ ] Audit SSH credential management with focus on:
    - Password encryption implementation
    - Storage of sensitive data
    - Transmission security (HTTPS)
  - [ ] Review MCP connection security:
    - Command sanitization
    - Input validation
    - Authentication for all endpoints
  - [ ] Check for possible injection vulnerabilities:
    - SQL injection in database operations
    - Command injection in SSH commands
    - Input validation in all API endpoints
  - [ ] Ensure secure password storage and transmission:
    - Use of proper encryption algorithms
    - Secure key management
    - No logging of sensitive data

- [ ] **User acceptance testing:**
  - [ ] Test MCP installation flow:
    - Manual installation instructions
    - SSH-based installation
    - Installation error handling
  - [ ] Test connection management:
    - Adding/editing/deleting MCP servers
    - Adding/editing/deleting SSH configurations
    - Connection testing
  - [ ] Test command execution and approval:
    - Command parsing from AI responses
    - Command approval UI
    - Command execution and result display
  - [ ] Test error handling and recovery:
    - Connection failures
    - Command execution failures
    - Authentication failures

### Phase 8: Integration with Main Application
- [ ] **Integrate with AI chat system:**
  - [ ] Update AI prompt templates to include MCP capabilities
  - [ ] Create special handling for MCP tool requests in chat
  - [ ] Implement MCP command detection in AI responses
  - [ ] Add MCP status to chat context for AI awareness

- [ ] **Update application initialization:**
  - [ ] Add MCPProvider to main application component
  - [ ] Initialize MCP connection on application start
  - [ ] Implement connection status monitoring
  - [ ] Add MCP status to application health checks

- [ ] **Create documentation:**
  - [ ] Update user documentation with MCP features
  - [ ] Create administrator documentation for MCP setup
  - [ ] Create developer documentation for MCP integration
  - [ ] Add troubleshooting guides for common issues

### Phase 9: SPA Routing and Frontend-Backend Connection
- [x] **Add SPA route handling for MCP pages:**
  - [x] Create `src/routes/mcp-pages.js` to handle MCP frontend routes:
    ```javascript
    const express = require('express');
    const path = require('path');
    const router = express.Router();
    const { requireAuth } = require('./auth');

    // MCP SSH Setup page
    router.get('/ssh-setup', requireAuth, (req, res) => {
      // Return the SPA for handling on frontend
      res.sendFile(path.resolve(__dirname, '../../client/build/index.html'));
    });

    // MCP Manual Install page
    router.get('/manual-install', requireAuth, (req, res) => {
      // Return the SPA for handling on frontend
      res.sendFile(path.resolve(__dirname, '../../client/build/index.html'));
    });

    module.exports = router;
    ```

  - [x] Update `src/server.js` to include the MCP pages router:
    ```javascript
    // Add this to the existing routes section
    app.use('/settings/mcp', mcpPagesRoutes);
    ```

  - [ ] Create new React component `client/src/pages/MCPSSHSetup.tsx`:
    ```typescript
    import React, { useState, useEffect } from 'react';
    import { useNavigate } from 'react-router-dom';
    import SSHConfigList from '../components/settings/SSHConfigList';
    import SSHConfigForm from '../components/settings/SSHConfigForm';
    import axios from 'axios';

    const MCPSSHSetup: React.FC = () => {
      const [configurations, setConfigurations] = useState([]);
      const [loading, setLoading] = useState(true);
      const [error, setError] = useState(null);
      const [showAddForm, setShowAddForm] = useState(false);
      const navigate = useNavigate();

      useEffect(() => {
        fetchSSHConfigs();
      }, []);

      const fetchSSHConfigs = async () => {
        setLoading(true);
        try {
          const response = await axios.get('/api/mcp/ssh/config');
          setConfigurations(response.data.configurations || []);
        } catch (err) {
          setError('Failed to load SSH configurations');
          console.error(err);
        } finally {
          setLoading(false);
        }
      };

      // Implementation of onSave, onDelete, onTest handlers

      return (
        <div className="py-6 px-4 md:px-8">
          <div className="flex items-center mb-6">
            <button
              onClick={() => navigate('/settings')}
              className="mr-4 text-sm"
              style={{ color: 'var(--color-primary)' }}
            >
              ← Back to Settings
            </button>
            <h1 className="text-2xl font-bold" style={{ color: 'var(--color-text)' }}>
              SSH Setup for MCP Installation
            </h1>
          </div>

          <div className="space-y-6">
            <p style={{ color: 'var(--color-text-secondary)' }}>
              Configure SSH connections to install MCP on remote machines.
            </p>

            <SSHConfigList
              configurations={configurations}
              onSave={handleSaveConfig}
              onDelete={handleDeleteConfig}
              onTest={handleTestConnection}
            />

            <div className="mt-8">
              <button
                onClick={() => navigate('/settings/mcp/install-mcp')}
                className="px-4 py-2 rounded"
                style={{
                  backgroundColor: 'var(--color-primary)',
                  color: 'white'
                }}
              >
                Continue to MCP Installation
              </button>
            </div>
          </div>
        </div>
      );
    };

    export default MCPSSHSetup;
    ```

  - [ ] Update `client/src/App.tsx` to include the new routes:
    ```typescript
    <Routes>
      {/* Existing routes... */}
      <Route path="/settings/mcp/ssh-setup" element={<MCPSSHSetup />} />
      <Route path="/settings/mcp/manual-install" element={<MCPManualInstall />} />
    </Routes>
    ```

## Technical Implementation Details

### SSH Authentication Security
- Password encryption will use AES-256-GCM with a secure key management strategy
- Private keys will be handled securely, never stored in plaintext
- All credentials will be validated server-side before use

### MCP Connection Management
- The application will establish a persistent SSE connection to the MCP server
- Connection status will be monitored and reported to the user
- Reconnection will be handled automatically when possible

### Command Approval Security
- All commands will require explicit user approval before execution
- The UI will clearly show what commands will be executed
- Users can modify or reject commands before execution

### Error Handling Strategy
- Clear error messages for connection issues
- Detailed logging for troubleshooting
- Graceful fallbacks when MCP is unavailable

## Interactive MCP Installation Implementation Plan

### Overview
This section outlines the implementation plan for an interactive MCP installation process that allows users to navigate the remote filesystem and select an installation directory before executing the MCP installation command.

### Phase 1: Remote Filesystem Explorer Component

#### 1. Create Remote Filesystem Explorer Component
- [ ] Create `client/src/components/ssh/RemoteFilesystemExplorer.tsx` with the following features:
  - Directory navigation (double-click to enter directories)
  - Directory selection (single-click to select)
  - Breadcrumb navigation for current path
  - "Go up" button to navigate to parent directory
  - Loading indicators for directory contents
  - Error handling for permission issues or connection problems
  - Responsive design for different screen sizes

```typescript
interface FileItem {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size?: number;
  permissions?: string;
  owner?: string;
  group?: string;
  lastModified?: string;
}

interface RemoteFilesystemExplorerProps {
  sshConfigId: string;
  initialPath?: string;
  onPathSelect: (path: string) => void;
  onCancel: () => void;
}
```

#### 2. Create Backend API for Remote Filesystem Operations
- [ ] Add new endpoints to `src/routes/mcp.js`:
  - `GET /api/mcp/ssh/fs/list` - List directory contents
  - `GET /api/mcp/ssh/fs/info` - Get information about a file or directory
  - `POST /api/mcp/ssh/fs/mkdir` - Create a new directory (optional)

#### 3. Implement SSH Filesystem Service
- [ ] Add new functions to `src/services/sshService.js`:
  - `listDirectory(sshConfig, path)` - List contents of a directory
  - `getFileInfo(sshConfig, path)` - Get information about a file
  - `createDirectory(sshConfig, path)` - Create a new directory (optional)
  - Implement proper error handling for filesystem operations
  - Add security checks to prevent unauthorized access

### Phase 2: Interactive MCP Installation Flow

#### 1. Enhance MCPInstall Component
- [ ] Update `client/src/pages/MCPInstall.tsx` to include:
  - Multi-step installation wizard UI
  - Step 1: Select SSH configuration and MCP tool (existing)
  - Step 2: Navigate and select installation directory
  - Step 3: Confirm installation details
  - Step 4: Installation progress and results
  - Back/Next/Cancel buttons for navigation between steps
  - Proper state management for the installation process

#### 2. Create Installation Directory Selection Step
- [ ] Implement directory selection UI:
  - Integrate RemoteFilesystemExplorer component
  - Add current path display
  - Add directory creation option (optional)
  - Validate selected directory for write permissions
  - Show available disk space (optional)

#### 3. Enhance Backend Installation Process
- [ ] Update `src/services/sshService.js` to support:
  - Installation in a specific directory
  - Real-time output streaming
  - Proper error handling for installation failures
  - Verification of successful installation
  - Cleanup on installation failure

```javascript
/**
 * Install MCP via SSH in a specific directory
 *
 * @param {Object} sshConfig - SSH connection configuration
 * @param {string} installDir - Directory to install MCP in
 * @param {string} installCommand - Command to execute for installation
 * @param {function} outputCallback - Callback for real-time output
 * @returns {Promise<Object>} - Installation result
 */
async function installMCPViaSSH(sshConfig, installDir, installCommand, outputCallback) {
  // Implementation details
}
```

#### 4. Implement Real-time Installation Output
- [ ] Create WebSocket or Server-Sent Events endpoint for real-time output:
  - `POST /api/mcp/ssh/install/stream` - Stream installation output
  - Implement proper authentication for the streaming endpoint
  - Handle connection errors and reconnection
  - Format output for display in the UI

#### 5. Add Installation Status Tracking
- [ ] Create installation status tracking:
  - Track installation progress in the database
  - Store installation logs for troubleshooting
  - Allow resuming failed installations (optional)
  - Provide installation history (optional)

### Phase 3: User Experience Enhancements

#### 1. Add Installation Confirmation Dialog
- [ ] Create confirmation dialog with:
  - Summary of installation details (SSH config, directory, command)
  - Warning about potential risks
  - Option to modify installation command
  - Clear confirmation button

#### 2. Implement Error Recovery
- [ ] Add error recovery mechanisms:
  - Detect common installation failures
  - Provide troubleshooting guidance
  - Option to retry failed steps
  - Detailed error logs for support

#### 3. Add Post-Installation Verification
- [ ] Implement verification steps:
  - Verify MCP server is running after installation
  - Test basic MCP functionality
  - Add automatic MCP server configuration
  - Show connection details for the installed server

### Phase 4: Security Considerations

#### 1. Secure Filesystem Access
- [ ] Implement security measures:
  - Restrict filesystem access to authenticated users
  - Validate all paths to prevent directory traversal attacks
  - Limit access to sensitive system directories
  - Log all filesystem operations for audit purposes

#### 2. Secure Command Execution
- [ ] Enhance command execution security:
  - Validate and sanitize all user inputs
  - Restrict command execution to authorized users
  - Prevent command injection attacks
  - Set appropriate timeouts for long-running commands

#### 3. Error Handling and Logging
- [ ] Improve error handling:
  - Implement detailed error logging
  - Sanitize error messages to prevent information disclosure
  - Provide user-friendly error messages
  - Add support contact information for unresolvable errors

### Phase 5: Testing and Documentation

#### 1. Comprehensive Testing
- [ ] Implement testing strategy:
  - Unit tests for filesystem operations
  - Integration tests for installation flow
  - Security testing for filesystem access
  - Performance testing for large directories
  - Error handling tests for various failure scenarios

#### 2. User Documentation
- [ ] Create user documentation:
  - Step-by-step installation guide
  - Troubleshooting guide for common issues
  - Security best practices
  - FAQ for installation process

#### 3. Developer Documentation
- [ ] Create developer documentation:
  - API documentation for filesystem operations
  - Component documentation for RemoteFilesystemExplorer
  - Security considerations for future development
  - Performance optimization guidelines

## Folder Selection GUI Implementation Plan

### Overview
This detailed implementation plan outlines the steps required to add a folder selection GUI to the MCP installation process. This will allow users to browse directories on the remote machine and select an installation location before running the MCP installation command.

### Key Components and Files Involved

#### Frontend Components
1. `client/src/components/ssh/RemoteFilesystemExplorer.tsx` (new)
   - Main component for browsing remote directories
   - Displays directory contents in a user-friendly interface
   - Handles directory navigation and selection

2. `client/src/pages/MCPInstall.tsx` (update)
   - Add a step for directory selection in the installation flow
   - Integrate the RemoteFilesystemExplorer component
   - Update the installation process to use the selected directory

3. `client/src/components/ssh/DirectoryBrowser.tsx` (new)
   - Reusable component for directory browsing
   - Includes breadcrumb navigation and directory listing
   - Handles user interactions for directory selection

4. `client/src/components/ssh/FileSystemItem.tsx` (new)
   - Component for rendering individual files/directories
   - Displays file/directory information (name, size, permissions)
   - Handles click events for navigation and selection

#### Backend API Endpoints
1. `src/routes/mcp.js` (update)
   - Add new endpoints for filesystem operations:
     - `GET /api/mcp/ssh/fs/list` - List directory contents
     - `GET /api/mcp/ssh/fs/info` - Get file/directory information
     - `POST /api/mcp/ssh/fs/mkdir` - Create new directory (optional)

2. `src/services/sshService.js` (update)
   - Add new functions for filesystem operations:
     - `listDirectory(sshConfig, path)` - List directory contents
     - `getFileInfo(sshConfig, path)` - Get file/directory information
     - `createDirectory(sshConfig, path)` - Create new directory (optional)
     - `installMCPInDirectory(sshConfig, directory, command)` - Install MCP in specified directory

### Implementation Steps

#### Step 1: Backend API Implementation
1. Update `src/services/sshService.js`:
   - [x] Implement `authenticateSSH(sshConfig, password)` function for initial authentication
   - [x] Implement `listDirectory(sshConfig, path, password)` function with authentication
   - [x] Implement `getFileInfo(sshConfig, path, password)` function with authentication
   - [x] Implement `createDirectory(sshConfig, path, password)` function with authentication (optional)
   - [x] Update `installMCPViaSSH` to accept a target directory parameter and require password confirmation
   - [x] Implement secure password handling for all operations

2. Update `src/routes/mcp.js`:
   - [x] Add `POST /api/mcp/ssh/authenticate` endpoint for initial authentication
   - [x] Add `GET /api/mcp/ssh/fs/list` endpoint with authentication
   - [x] Add `GET /api/mcp/ssh/fs/info` endpoint with authentication
   - [x] Add `POST /api/mcp/ssh/fs/mkdir` endpoint with authentication (optional)
   - [x] Add `POST /api/mcp/ssh/install-in-directory` endpoint to accept a target directory and require password confirmation

#### Step 2: Frontend Component Implementation
1. Create `client/src/components/ssh/SSHPasswordPrompt.tsx`:
   - [x] Implement password input form with validation
   - [x] Add secure password handling
   - [x] Implement authentication status feedback
   - [x] Add error handling for authentication failures

2. Create `client/src/components/ssh/FileSystemItem.tsx`:
   - [x] Implement file/directory item display
   - [x] Add click handlers for navigation and selection
   - [x] Style the component for different file types

3. Create `client/src/components/ssh/DirectoryBrowser.tsx`:
   - [x] Implement directory listing functionality
   - [x] Add breadcrumb navigation
   - [x] Implement directory navigation (up, down, home)
   - [x] Add loading indicators and error handling

4. Create `client/src/components/ssh/RemoteFilesystemExplorer.tsx`:
   - [x] Implement the main explorer component
   - [x] Integrate DirectoryBrowser component
   - [x] Add path selection functionality
   - [x] Implement modal dialog behavior

5. Update `client/src/pages/MCPInstall.tsx`:
   - [x] Implement multi-step installation wizard with password authentication steps
   - [x] Add initial password authentication step before directory browsing
   - [x] Add confirmation password authentication step before installation
   - [x] Integrate SSHPasswordPrompt component
   - [x] Integrate RemoteFilesystemExplorer component
   - [x] Update installation flow to use selected directory
   - [x] Add UI for displaying selected directory

#### Step 3: Bug Fixes and Improvements
1. Fix directory listing functionality:
   - [x] Improve the `listDirectory` function in `sshService.js` to handle different output formats
   - [x] Use more reliable commands (`find`) for directory listing
   - [x] Add better error handling and logging for debugging
   - [x] Fix parsing of directory contents

2. Fix MCP installation process:
   - [x] Update `installMCPInDirectory` to use commands from config.ini
   - [x] Add proper start and status checking after installation
   - [x] Improve port detection from command output
   - [x] Add warning handling for partial success scenarios
   - [x] Update UI to show port information after successful installation

3. Fix database schema compatibility issues:
   - [x] Fix column name mismatch in `user_mcp_server_configurations` table
   - [x] Update `createMCPServerConfiguration` to handle different schema versions
   - [x] Add dynamic column detection to support both `server_name` and `mcp_nickname` columns
   - [x] Improve error handling and logging for database operations
   - [x] Update all MCP database functions to handle schema differences:
     - [x] `getUserMCPServerConfigurations`
     - [x] `getMCPServerConfiguration`
     - [x] `getDefaultMCPServerConfiguration`
     - [x] `updateMCPConnectionStatus`
   - [x] Handle unique constraint violations for MCP server configurations:
     - [x] Update existing configurations instead of creating duplicates
     - [x] Generate unique names for new configurations
     - [x] Add fallback mechanism when database operations fail during installation
   - [x] Fix frontend MCP connection management:
     - [x] Update `MCPSettings.tsx` to use correct API endpoints
     - [x] Fix `handleSaveConnection` to use `/api/mcp/server/config` endpoint
     - [x] Update data mapping between frontend and backend
     - [x] Fix server deletion functionality
   - [x] Improve MCP connection status checking:
     - [x] Add automatic connection status checking on page load
     - [x] Add manual connection status checking with refresh button
     - [x] Use `/info` endpoint to check if MCP server is running
     - [x] Add visual indicators for connection status (connected, checking, error)
     - [x] Add server version and tools count display

#### Step 4: WebSocket Integration for Real-time Updates
1. Update WebSocket handlers:
   - [ ] Add support for directory listing updates
   - [ ] Implement real-time feedback during directory operations
   - [ ] Add error handling for WebSocket communication

2. Update frontend components to use WebSockets:
   - [ ] Modify RemoteFilesystemExplorer to use WebSockets for updates
   - [ ] Add reconnection logic for WebSocket failures
   - [ ] Implement fallback to REST API when WebSockets are unavailable

#### Step 5: Security Implementation
1. Add security measures to backend:
   - [ ] Implement secure password handling with encryption in transit
   - [ ] Never store passwords in plaintext or in session storage
   - [ ] Validate all path inputs to prevent directory traversal attacks
   - [ ] Add authentication checks for all filesystem operations
   - [ ] Implement proper error handling and logging
   - [ ] Add rate limiting for filesystem operations
   - [ ] Implement session timeout for authenticated SSH sessions

2. Add security measures to frontend:
   - [ ] Ensure password fields are properly secured (no autocomplete, masked input)
   - [ ] Clear password from memory after authentication
   - [ ] Sanitize all user inputs
   - [ ] Add confirmation dialogs for potentially dangerous operations
   - [ ] Implement proper error handling and user feedback
   - [ ] Add visual indicators for secure connections

#### Step 5: Testing and Documentation
1. Testing:
   - [ ] Write unit tests for filesystem operations
   - [ ] Test directory navigation and selection
   - [ ] Test error handling and edge cases
   - [ ] Test security measures

2. Documentation:
   - [ ] Update user documentation with directory selection instructions
   - [ ] Add developer documentation for the new components
   - [ ] Document security considerations and best practices
   - [ ] Create troubleshooting guide for common issues

### Expected User Flow
1. User selects an SSH configuration and MCP installation tool
2. User clicks "Next" and is prompted to enter SSH password (if using password authentication)
3. After successful authentication, the RemoteFilesystemExplorer opens, showing the home directory
4. User navigates to the desired installation directory
5. User selects the directory and clicks "Select"
6. The selected directory is displayed in the installation form
7. User confirms the installation details and clicks "Install"
8. User is prompted again to enter SSH password for confirmation
9. After authentication, the installation command runs in the selected directory
10. Real-time updates are shown during installation
11. User receives confirmation when installation is complete

### Technical Considerations
1. Authentication Flow:
   - Implement secure password handling throughout the process
   - Use temporary authentication tokens for the directory browsing session
   - Require re-authentication for the final installation step
   - Implement proper timeout for authentication sessions
   - Never store passwords in browser storage or cookies

2. Performance:
   - Implement pagination for large directories
   - Add caching for frequently accessed directories
   - Optimize WebSocket communication for real-time updates
   - Maintain SSH connection during directory browsing to avoid reconnection delays

3. Error Handling:
   - Handle network failures gracefully
   - Provide clear error messages for authentication failures
   - Provide clear error messages for permission issues
   - Add retry mechanisms for transient failures
   - Implement proper recovery from authentication timeouts

4. Security:
   - Validate all user inputs server-side
   - Implement proper authentication and authorization
   - Prevent directory traversal and command injection attacks
   - Use HTTPS for all communications
   - Implement proper session management
   - Log authentication attempts for security auditing

5. User Experience:
   - Add keyboard shortcuts for navigation
   - Implement drag-and-drop for directory selection (optional)
   - Add search functionality for large directories (optional)
   - Provide clear visual feedback for loading states and errors
   - Show clear progress indicators during authentication steps
   - Provide helpful error messages for authentication failures

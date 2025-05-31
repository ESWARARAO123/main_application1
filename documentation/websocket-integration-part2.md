## Phase 4: MCP Installation with WebSockets

1. **Create Remote Filesystem Explorer with WebSockets**
   ```javascript
   // src/services/sshService.js
   async listDirectory(sshConfig, path, wsConnection) {
     const ssh = new NodeSSH();
     
     try {
       // Connect to SSH
       await ssh.connect({
         host: sshConfig.ssh_host,
         port: sshConfig.ssh_port,
         username: sshConfig.ssh_user,
         // Authentication details...
       });
       
       // Execute ls command
       const result = await ssh.execCommand(`ls -la ${path}`);
       
       // Parse directory listing
       const files = parseDirectoryListing(result.stdout);
       
       // Send result via WebSocket if provided
       if (wsConnection) {
         wsConnection.send(JSON.stringify({
           type: 'directory_listing',
           path,
           files
         }));
       }
       
       return {
         success: true,
         files
       };
     } catch (error) {
       // Send error via WebSocket if provided
       if (wsConnection) {
         wsConnection.send(JSON.stringify({
           type: 'error',
           error: error.message
         }));
       }
       
       return {
         success: false,
         error: error.message
       };
     } finally {
       ssh.dispose();
     }
   }
   ```

2. **Implement MCP Installation with Real-time Updates**
   ```javascript
   // src/services/sshService.js
   async installMCPViaSSH(sshConfig, installDir, installCommand, userId) {
     const ssh = new NodeSSH();
     const installId = uuidv4();
     const wsServer = app.get('wsServer');
     
     try {
       // Send initial status
       wsServer.broadcast(userId, {
         type: 'mcp_install',
         installId,
         status: 'connecting',
         message: `Connecting to ${sshConfig.ssh_host}...`
       });
       
       // Connect to SSH
       await ssh.connect({
         host: sshConfig.ssh_host,
         port: sshConfig.ssh_port,
         username: sshConfig.ssh_user,
         // Authentication details...
       });
       
       // Send connected status
       wsServer.broadcast(userId, {
         type: 'mcp_install',
         installId,
         status: 'connected',
         message: `Connected to ${sshConfig.ssh_host}`
       });
       
       // Change to installation directory
       wsServer.broadcast(userId, {
         type: 'mcp_install',
         installId,
         status: 'preparing',
         message: `Changing to directory: ${installDir}`
       });
       
       await ssh.execCommand(`cd ${installDir}`);
       
       // Execute installation command with real-time output
       wsServer.broadcast(userId, {
         type: 'mcp_install',
         installId,
         status: 'installing',
         message: `Executing: ${installCommand}`
       });
       
       // Stream command output
       const installProcess = await ssh.execCommand(installCommand, {
         onStdout: (chunk) => {
           wsServer.broadcast(userId, {
             type: 'mcp_install_output',
             installId,
             output: chunk.toString(),
             stream: 'stdout'
           });
         },
         onStderr: (chunk) => {
           wsServer.broadcast(userId, {
             type: 'mcp_install_output',
             installId,
             output: chunk.toString(),
             stream: 'stderr'
           });
         }
       });
       
       // Check installation result
       if (installProcess.code === 0) {
         wsServer.broadcast(userId, {
           type: 'mcp_install',
           installId,
           status: 'success',
           message: 'MCP installation completed successfully'
         });
         
         return {
           success: true,
           message: 'MCP installation completed successfully'
         };
       } else {
         wsServer.broadcast(userId, {
           type: 'mcp_install',
           installId,
           status: 'failed',
           message: `Installation failed: ${installProcess.stderr}`
         });
         
         return {
           success: false,
           error: installProcess.stderr
         };
       }
     } catch (error) {
       wsServer.broadcast(userId, {
         type: 'mcp_install',
         installId,
         status: 'error',
         message: `Error: ${error.message}`
       });
       
       return {
         success: false,
         error: error.message
       };
     } finally {
       ssh.dispose();
     }
   }
   ```

## Phase 5: Authentication and Session Management

1. **WebSocket Authentication Middleware**
   ```javascript
   // src/websocket/auth.js
   const cookie = require('cookie');
   const { verifySession } = require('../services/authService');

   async function authenticateWebSocket(req) {
     try {
       // Parse cookies from request headers
       const cookies = cookie.parse(req.headers.cookie || '');
       
       // Get session ID from cookie
       const sessionId = cookies['connect.sid'];
       
       if (!sessionId) {
         return null;
       }
       
       // Verify session and get user ID
       const userId = await verifySession(sessionId);
       
       return userId;
     } catch (error) {
       console.error('WebSocket authentication error:', error);
       return null;
     }
   }
   ```

2. **Session Expiration Notifications**
   ```javascript
   // src/services/authService.js
   function setupSessionExpirationNotifications(wsServer) {
     // Check for expiring sessions every minute
     setInterval(async () => {
       const expiringSessionsResult = await db.query(
         'SELECT user_id, expires FROM sessions WHERE expires > NOW() AND expires < NOW() + INTERVAL \'5 minutes\''
       );
       
       for (const session of expiringSessionsResult.rows) {
         const expiresIn = Math.floor((new Date(session.expires) - new Date()) / 1000);
         
         wsServer.broadcast(session.user_id, {
           type: 'session_update',
           status: 'expiring_soon',
           expiresIn
         });
       }
     }, 60000); // Check every minute
   }
   ```

## Pros and Cons of WebSocket Integration

### Pros

1. **Real-time Updates**: Provides immediate updates without polling, reducing latency
2. **Reduced Server Load**: Eliminates frequent polling requests, reducing server load
3. **Bandwidth Efficiency**: More efficient than polling for frequent updates
4. **Bidirectional Communication**: Enables server-initiated messages and client responses
5. **Enhanced User Experience**: Creates a more responsive and interactive application
6. **Unified API**: Consolidates real-time features under a single communication protocol
7. **Scalability**: Can be scaled with proper architecture (Redis pub/sub, etc.)

### Cons

1. **Implementation Complexity**: More complex than REST APIs to implement correctly
2. **Connection Management**: Requires handling connection state, reconnection, etc.
3. **Security Considerations**: Needs careful authentication and authorization
4. **Resource Usage**: Persistent connections consume server resources
5. **Firewall Issues**: Some networks may block WebSocket connections
6. **Debugging Challenges**: More difficult to debug than standard HTTP requests
7. **Browser Compatibility**: Older browsers may require fallbacks

## Conclusion

WebSocket integration would significantly enhance the real-time capabilities of the application, particularly for document processing, RAG queries, chat interactions, and MCP installation. The implementation should be phased, starting with core infrastructure and gradually extending to specific features.

The most immediate benefits would be seen in:

1. Document processing status updates
2. RAG query processing feedback
3. MCP installation with real-time terminal output
4. Chat message streaming

By implementing WebSockets, the application would provide a more responsive, interactive experience while potentially reducing server load from polling requests.

/**
 * SSH Service
 * Manages SSH connections and installation of MCP via SSH
 */

const { NodeSSH } = require('node-ssh');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const config = require('../utils/config');
const { logger } = require('../utils/logger');

// Get encryption key from environment or generate one
const ENCRYPTION_KEY = process.env.SSH_ENCRYPTION_KEY ||
  crypto.randomBytes(32).toString('hex');

/**
 * Test SSH connection with provided configuration
 *
 * @param {Object} sshConfig - SSH connection configuration
 * @param {string} sshConfig.ssh_host - SSH server hostname or IP
 * @param {number} sshConfig.ssh_port - SSH server port
 * @param {string} sshConfig.ssh_user - SSH username
 * @param {string} sshConfig.ssh_auth_method - Authentication method ('password' or 'key')
 * @param {string} [sshConfig.ssh_password] - SSH password (if ssh_auth_method is 'password')
 * @param {string} [sshConfig.ssh_key_path] - Path to private key (if ssh_auth_method is 'key')
 * @returns {Promise<Object>} - Connection test result
 */
async function testSSHConnection(sshConfig) {
  const ssh = new NodeSSH();

  try {
    // Map the frontend field names to what the SSH library expects
    const connectionConfig = {
      host: sshConfig.ssh_host || sshConfig.host,
      port: sshConfig.ssh_port || sshConfig.port || 22, // Default to port 22 if not specified
      username: sshConfig.ssh_user || sshConfig.username,
      // Set auth method based on config
      ...(
        (sshConfig.ssh_auth_method === 'password' || sshConfig.authMethod === 'password')
          ? { password: sshConfig.ssh_password || sshConfig.password }
          : { privateKey: sshConfig.ssh_key_path || sshConfig.keyPath }
      )
    };

    logger.info(`Attempting SSH connection to ${connectionConfig.host}:${connectionConfig.port} as ${connectionConfig.username}`);

    // Try to connect
    await ssh.connect(connectionConfig);

    // Test a simple command
    const result = await ssh.execCommand('echo "SSH connection successful"');

    if (result.code !== 0) {
      throw new Error(`Command execution failed: ${result.stderr}`);
    }

    // Disconnect
    ssh.dispose();

    logger.info(`SSH connection test successful to ${connectionConfig.host}`);

    return {
      success: true,
      message: 'SSH connection test successful'
    };
  } catch (err) {
    // Make sure SSH connection is closed
    ssh.dispose();

    logger.error(`SSH connection test failed: ${err.message}`);

    return {
      success: false,
      error: err.message || 'SSH connection test failed'
    };
  }
}

/**
 * Execute a command over SSH
 *
 * @param {Object} sshConfig - SSH connection configuration
 * @param {string} command - Command to execute
 * @param {Object} [options] - Command execution options
 * @param {boolean} [options.cwd] - Working directory
 * @param {boolean} [options.stream] - Whether to stream output
 * @returns {Promise<Object>} - Command execution result
 */
async function executeSSHCommand(sshConfig, command, options = {}) {
  const ssh = new NodeSSH();

  try {
    // If password is encrypted, decrypt it
    let password = sshConfig.ssh_password || sshConfig.password;
    const authMethod = sshConfig.ssh_auth_method || sshConfig.authMethod;

    if (authMethod === 'password') {
      if (sshConfig.ssh_password_encrypted || sshConfig.passwordEncrypted) {
        password = decryptPassword(sshConfig.ssh_password_encrypted || sshConfig.passwordEncrypted);
      }
    }

    const connectionConfig = {
      host: sshConfig.ssh_host || sshConfig.host,
      port: sshConfig.ssh_port || sshConfig.port,
      username: sshConfig.ssh_user || sshConfig.username,
      // Set auth method based on config
      ...(authMethod === 'password'
        ? { password }
        : { privateKey: sshConfig.ssh_key_path || sshConfig.keyPath })
    };

    // Connect to the SSH server
    await ssh.connect(connectionConfig);

    // Execute the command
    const result = await ssh.execCommand(command, {
      cwd: options.cwd || '.',
      stream: options.stream || false,
      onStdout: options.onStdout,
      onStderr: options.onStderr
    });

    // Disconnect
    ssh.dispose();

    return {
      success: result.code === 0,
      code: result.code,
      stdout: result.stdout,
      stderr: result.stderr
    };
  } catch (err) {
    // Make sure SSH connection is closed
    ssh.dispose();

    logger.error(`SSH command execution failed: ${err.message}`);

    return {
      success: false,
      error: err.message || 'SSH command execution failed'
    };
  }
}

/**
 * Install MCP server via SSH
 *
 * @param {Object} sshConfig - SSH connection configuration
 * @returns {Promise<Object>} - Installation result
 */
async function installMCPViaSSH(sshConfig) {
  // Get all MCP commands from config
  const mcpName = config.get('mcp-server.mcp_terminal_name_1') || 'mcp-terminal_executor';
  const mcpInstallCommand = config.get('mcp-server.mcp_terminal_command_1');
  const startCommand = config.get('mcp-server.mcp_terminal_command_1_install_cmd') || `${mcpName} start`;
  const stopCommand = config.get('mcp-server.mcp_terminal_command_1_stop_cmd') || `${mcpName} stop`;
  const restartCommand = config.get('mcp-server.mcp_terminal_command_1_restart_cmd') || `${mcpName} restart`;
  const statusCommand = config.get('mcp-server.mcp_terminal_command_1_status_cmd') || `${mcpName} status`;
  const uninstallCommand = config.get('mcp-server.mcp_terminal_command_1_uninstall_cmd') || `${mcpName} uninstall`;
  const defaultPort = config.get('mcp-server.mcp_terminal_command_default_port_1') || 8080;

  logger.info(`Using MCP commands from config:
    - Install: ${mcpInstallCommand}
    - Start: ${startCommand}
    - Stop: ${stopCommand}
    - Restart: ${restartCommand}
    - Status: ${statusCommand}
    - Uninstall: ${uninstallCommand}
    - Default Port: ${defaultPort}`);

  if (!mcpInstallCommand) {
    return {
      success: false,
      error: 'MCP installation command not found in configuration'
    };
  }

  try {
    // First test SSH connection
    const testResult = await testSSHConnection(sshConfig);

    if (!testResult.success) {
      throw new Error(`SSH connection failed: ${testResult.error}`);
    }

    // Get the host from the config, supporting both naming conventions
    const host = sshConfig.ssh_host || sshConfig.host;

    // Execute installation command
    logger.info(`Installing MCP on ${host} via SSH...`);

    const installResult = await executeSSHCommand(sshConfig, mcpInstallCommand, {
      stream: true
    });

    if (!installResult.success) {
      throw new Error(`MCP installation failed: ${installResult.stderr}`);
    }

    // Start the MCP server
    logger.info(`Starting MCP server with command: ${startCommand}`);
    const startResult = await executeSSHCommand(sshConfig, startCommand);

    if (!startResult.success) {
      logger.warn(`Failed to start MCP server: ${startResult.stderr}`);
      // Don't fail the installation if start fails, just log it
    } else {
      logger.info(`MCP server started successfully: ${startResult.stdout}`);
    }

    // Check the status to verify installation
    logger.info(`Checking MCP server status with command: ${statusCommand}`);
    const statusResult = await executeSSHCommand(sshConfig, statusCommand);

    // We'll skip the initial verification with default port and directly detect the actual port

    // Parse the output to get the port - this should be detected from the actual output
    // rather than assuming a default
    let port = null;

    // Try to find port in the output of start or status commands
    const portMatch = startResult.stdout?.match(/port (\d+)/i) ||
                      statusResult.stdout?.match(/port (\d+)/i) ||
                      startResult.stdout?.match(/listening on .*:(\d+)/i) ||
                      statusResult.stdout?.match(/listening on .*:(\d+)/i);

    if (portMatch && portMatch[1]) {
      port = parseInt(portMatch[1], 10);
      logger.info(`Detected MCP server running on port: ${port}`);
    } else {
      // If we can't detect the port, try to find it using netstat
      logger.info(`Port not found in command output, trying to detect using netstat...`);
      const netstatCommand = `netstat -tulpn | grep "${mcpName}" | awk '{print $4}' | awk -F: '{print $NF}'`;
      const netstatResult = await executeSSHCommand(sshConfig, netstatCommand);

      if (netstatResult.success && netstatResult.stdout.trim()) {
        const ports = netstatResult.stdout.trim().split('\n');
        if (ports.length > 0 && /^\d+$/.test(ports[0])) {
          port = parseInt(ports[0], 10);
          logger.info(`Detected MCP server running on port: ${port} (via netstat)`);
        }
      }

      // If we still can't find the port, use the default as a last resort
      if (!port) {
        port = defaultPort;
        logger.warn(`Could not detect actual port, using default: ${port}`);
      }
    }

    // Verify the server is running on the detected port
    const specificVerifyCommand = `netstat -tulpn | grep ${port}`;
    const specificVerifyResult = await executeSSHCommand(sshConfig, specificVerifyCommand);

    if (!specificVerifyResult.success || !specificVerifyResult.stdout.includes(port.toString())) {
      logger.warn(`MCP server installation verification failed: Server not listening on port ${port}`);
      return {
        success: true,
        message: 'MCP server installed successfully, but verification failed. You may need to start it manually.',
        host: host,
        port: port,
        warning: "Server verification failed. You may need to start it manually.",
        portDetectionMethod: port !== defaultPort ? "detected" : "default"
      };
    }

    return {
      success: true,
      message: 'MCP server installed successfully',
      host: host,
      port: port,
      portDetectionMethod: port !== defaultPort ? "detected" : "default"
    };
  } catch (err) {
    logger.error(`MCP installation failed: ${err.message}`);

    return {
      success: false,
      error: err.message || 'MCP installation failed'
    };
  }
}

/**
 * Encrypt password for storage
 *
 * @param {string} password - Plain text password
 * @returns {string} - Encrypted password
 */
function encryptPassword(password) {
  // Generate a random initialization vector
  const iv = crypto.randomBytes(16);

  // Create cipher using AES-256-GCM
  const cipher = crypto.createCipheriv(
    'aes-256-gcm',
    Buffer.from(ENCRYPTION_KEY, 'hex'),
    iv
  );

  // Encrypt the password
  let encrypted = cipher.update(password, 'utf8', 'hex');
  encrypted += cipher.final('hex');

  // Get the auth tag
  const authTag = cipher.getAuthTag().toString('hex');

  // Combine IV, encrypted password, and auth tag for storage
  return `${iv.toString('hex')}:${encrypted}:${authTag}`;
}

/**
 * Decrypt password for use
 *
 * @param {string} encryptedPassword - Encrypted password
 * @returns {string} - Decrypted password
 */
function decryptPassword(encryptedPassword) {
  // Split the stored data
  const [ivHex, encrypted, authTagHex] = encryptedPassword.split(':');

  // Convert hex strings to buffers
  const iv = Buffer.from(ivHex, 'hex');
  const authTag = Buffer.from(authTagHex, 'hex');

  // Create decipher using AES-256-GCM
  const decipher = crypto.createDecipheriv(
    'aes-256-gcm',
    Buffer.from(ENCRYPTION_KEY, 'hex'),
    iv
  );

  // Set auth tag
  decipher.setAuthTag(authTag);

  // Decrypt the password
  let decrypted = decipher.update(encrypted, 'hex', 'utf8');
  decrypted += decipher.final('utf8');

  return decrypted;
}

/**
 * Authenticate SSH connection with password
 *
 * @param {Object} sshConfig - SSH connection configuration
 * @param {string} password - SSH password
 * @returns {Promise<Object>} - Authentication result
 */
async function authenticateSSH(sshConfig, password) {
  const ssh = new NodeSSH();

  try {
    const connectionConfig = {
      host: sshConfig.ssh_host || sshConfig.host,
      port: sshConfig.ssh_port || sshConfig.port || 22,
      username: sshConfig.ssh_user || sshConfig.username,
      password: password
    };

    logger.info(`Authenticating SSH connection to ${connectionConfig.host}:${connectionConfig.port} as ${connectionConfig.username}`);

    // Try to connect
    await ssh.connect(connectionConfig);

    // Test a simple command
    const result = await ssh.execCommand('echo "SSH authentication successful"');

    if (result.code !== 0) {
      throw new Error(`Authentication failed: ${result.stderr}`);
    }

    // Disconnect
    ssh.dispose();

    logger.info(`SSH authentication successful to ${connectionConfig.host}`);

    return {
      success: true,
      message: 'SSH authentication successful'
    };
  } catch (err) {
    // Make sure SSH connection is closed
    ssh.dispose();

    logger.error(`SSH authentication failed: ${err.message}`);

    return {
      success: false,
      error: err.message || 'SSH authentication failed'
    };
  }
}

/**
 * List directory contents via SSH
 *
 * @param {Object} sshConfig - SSH connection configuration
 * @param {string} directoryPath - Path to list
 * @param {string} password - SSH password
 * @returns {Promise<Object>} - Directory listing result
 */
async function listDirectory(sshConfig, directoryPath, password) {
  const ssh = new NodeSSH();

  try {
    const connectionConfig = {
      host: sshConfig.ssh_host || sshConfig.host,
      port: sshConfig.ssh_port || sshConfig.port || 22,
      username: sshConfig.ssh_user || sshConfig.username,
      password: password
    };

    // Connect to SSH
    await ssh.connect(connectionConfig);

    // Sanitize the path to prevent command injection
    const sanitizedPath = directoryPath.replace(/[;&|"`'$\\]/g, '\\$&');

    logger.info(`Listing directory contents for ${sanitizedPath}`);

    // Use a more reliable approach: get directories and files separately with simpler commands
    // First, get directories
    const dirCommand = `find "${sanitizedPath}" -maxdepth 1 -type d -not -path "${sanitizedPath}" | sort`;
    const dirResult = await ssh.execCommand(dirCommand);

    if (dirResult.code !== 0) {
      throw new Error(`Failed to list directories: ${dirResult.stderr}`);
    }

    // Then, get files
    const fileCommand = `find "${sanitizedPath}" -maxdepth 1 -type f | sort`;
    const fileResult = await ssh.execCommand(fileCommand);

    if (fileResult.code !== 0) {
      throw new Error(`Failed to list files: ${fileResult.stderr}`);
    }

    // Parse the results
    const items = [];

    // Process directories
    const directories = dirResult.stdout.split('\n').filter(line => line.trim() !== '');
    for (const dir of directories) {
      const name = dir.split('/').pop();
      items.push({
        name,
        path: dir,
        type: 'directory',
        permissions: 'drwxr-xr-x', // Default permissions
        owner: connectionConfig.username,
        group: connectionConfig.username,
        size: 4096, // Default size for directories
        lastModified: new Date().toISOString().split('T')[0]
      });
    }

    // Process files
    const files = fileResult.stdout.split('\n').filter(line => line.trim() !== '');
    for (const file of files) {
      const name = file.split('/').pop();
      items.push({
        name,
        path: file,
        type: 'file',
        permissions: '-rw-r--r--', // Default permissions
        owner: connectionConfig.username,
        group: connectionConfig.username,
        size: 0, // Default size
        lastModified: new Date().toISOString().split('T')[0]
      });
    }

    logger.info(`Found ${items.length} items in directory (${directories.length} directories, ${files.length} files)`);

    // Disconnect
    ssh.dispose();

    return {
      success: true,
      items: items
    };
  } catch (err) {
    // Make sure SSH connection is closed
    ssh.dispose();

    logger.error(`Directory listing failed: ${err.message}`);

    return {
      success: false,
      error: err.message || 'Failed to list directory'
    };
  }
}

/**
 * Get information about a file or directory
 *
 * @param {Object} sshConfig - SSH connection configuration
 * @param {string} path - Path to get info for
 * @param {string} password - SSH password
 * @returns {Promise<Object>} - File/directory info result
 */
async function getFileInfo(sshConfig, path, password) {
  const ssh = new NodeSSH();

  try {
    const connectionConfig = {
      host: sshConfig.ssh_host || sshConfig.host,
      port: sshConfig.ssh_port || sshConfig.port || 22,
      username: sshConfig.ssh_user || sshConfig.username,
      password: password
    };

    // Connect to SSH
    await ssh.connect(connectionConfig);

    // Sanitize the path to prevent command injection
    const sanitizedPath = path.replace(/[;&|"`'$\\]/g, '\\$&');

    // Execute stat command to get file info
    const command = `stat -c "%F:%a:%U:%G:%s:%Y:%n" "${sanitizedPath}"`;
    const result = await ssh.execCommand(command);

    if (result.code !== 0) {
      throw new Error(`Failed to get file info: ${result.stderr}`);
    }

    // Parse the stat output
    const parts = result.stdout.trim().split(':');
    const type = parts[0];
    const permissions = parts[1];
    const owner = parts[2];
    const group = parts[3];
    const size = parts[4];
    const modTime = parts[5];

    const isDirectory = type.includes('directory');

    // Disconnect
    ssh.dispose();

    return {
      success: true,
      info: {
        name: path.split('/').pop(),
        path: path,
        type: isDirectory ? 'directory' : 'file',
        size: parseInt(size, 10),
        permissions: permissions,
        owner: owner,
        group: group,
        lastModified: new Date(parseInt(modTime, 10) * 1000).toISOString()
      }
    };
  } catch (err) {
    // Make sure SSH connection is closed
    ssh.dispose();

    logger.error(`File info retrieval failed: ${err.message}`);

    return {
      success: false,
      error: err.message || 'Failed to get file info'
    };
  }
}

/**
 * Create a directory via SSH
 *
 * @param {Object} sshConfig - SSH connection configuration
 * @param {string} path - Path to create
 * @param {string} password - SSH password
 * @returns {Promise<Object>} - Directory creation result
 */
async function createDirectory(sshConfig, path, password) {
  const ssh = new NodeSSH();

  try {
    const connectionConfig = {
      host: sshConfig.ssh_host || sshConfig.host,
      port: sshConfig.ssh_port || sshConfig.port || 22,
      username: sshConfig.ssh_user || sshConfig.username,
      password: password
    };

    // Connect to SSH
    await ssh.connect(connectionConfig);

    // Sanitize the path to prevent command injection
    const sanitizedPath = path.replace(/[;&|"`'$\\]/g, '\\$&');

    // Execute mkdir command
    const command = `mkdir -p "${sanitizedPath}"`;
    const result = await ssh.execCommand(command);

    if (result.code !== 0) {
      throw new Error(`Failed to create directory: ${result.stderr}`);
    }

    // Disconnect
    ssh.dispose();

    return {
      success: true,
      message: `Directory ${path} created successfully`
    };
  } catch (err) {
    // Make sure SSH connection is closed
    ssh.dispose();

    logger.error(`Directory creation failed: ${err.message}`);

    return {
      success: false,
      error: err.message || 'Failed to create directory'
    };
  }
}

/**
 * Install MCP via SSH in a specific directory
 *
 * @param {Object} sshConfig - SSH connection configuration
 * @param {string} installDir - Directory to install MCP in
 * @param {string} installCommand - Command to execute for installation
 * @param {string} password - SSH password
 * @returns {Promise<Object>} - Installation result
 */
async function installMCPInDirectory(sshConfig, installDir, installCommand, password) {
  const ssh = new NodeSSH();

  try {
    const connectionConfig = {
      host: sshConfig.ssh_host || sshConfig.host,
      port: sshConfig.ssh_port || sshConfig.port || 22,
      username: sshConfig.ssh_user || sshConfig.username,
      password: password
    };

    // Connect to SSH
    await ssh.connect(connectionConfig);

    // Sanitize the directory path to prevent command injection
    const sanitizedDir = installDir.replace(/[;&|"`'$\\]/g, '\\$&');

    // First check if directory exists
    const checkDirCommand = `[ -d "${sanitizedDir}" ] && echo "Directory exists" || echo "Directory does not exist"`;
    const checkResult = await ssh.execCommand(checkDirCommand);

    if (checkResult.stdout.includes("Directory does not exist")) {
      // Create the directory
      const mkdirCommand = `mkdir -p "${sanitizedDir}"`;
      const mkdirResult = await ssh.execCommand(mkdirCommand);

      if (mkdirResult.code !== 0) {
        throw new Error(`Failed to create installation directory: ${mkdirResult.stderr}`);
      }
    }

    // Change to the installation directory and run the command
    const fullCommand = `cd "${sanitizedDir}" && ${installCommand}`;
    logger.info(`Installing MCP in directory ${installDir}`);

    const installResult = await ssh.execCommand(fullCommand);

    if (installResult.code !== 0) {
      throw new Error(`MCP installation failed: ${installResult.stderr}`);
    }

    logger.info(`MCP installation output: ${installResult.stdout}`);

    // Get all MCP commands from config
    const mcpName = config.get('mcp-server.mcp_terminal_name_1') || 'mcp-terminal_executor';
    const startCommand = config.get('mcp-server.mcp_terminal_command_1_install_cmd') || `${mcpName} start`;
    const stopCommand = config.get('mcp-server.mcp_terminal_command_1_stop_cmd') || `${mcpName} stop`;
    const restartCommand = config.get('mcp-server.mcp_terminal_command_1_restart_cmd') || `${mcpName} restart`;
    const statusCommand = config.get('mcp-server.mcp_terminal_command_1_status_cmd') || `${mcpName} status`;
    const uninstallCommand = config.get('mcp-server.mcp_terminal_command_1_uninstall_cmd') || `${mcpName} uninstall`;

    logger.info(`Using MCP commands from config:
      - Start: ${startCommand}
      - Stop: ${stopCommand}
      - Restart: ${restartCommand}
      - Status: ${statusCommand}
      - Uninstall: ${uninstallCommand}`);

    // Run the start command to start the MCP server
    logger.info(`Starting MCP server with command: ${startCommand}`);
    const startResult = await ssh.execCommand(startCommand);

    if (startResult.code !== 0) {
      logger.error(`Failed to start MCP server: ${startResult.stderr}`);
      // Don't fail the installation if start fails, just log it
    } else {
      logger.info(`MCP server started successfully: ${startResult.stdout}`);
    }

    // Check the status to verify installation
    logger.info(`Checking MCP server status with command: ${statusCommand}`);
    const statusResult = await ssh.execCommand(statusCommand);

    // Disconnect
    ssh.dispose();

    // Parse the output to get the port - this should be detected from the actual output
    // rather than assuming a default
    let port = null;

    // Try to find port in the output of start or status commands
    const portMatch = startResult.stdout?.match(/port (\d+)/i) ||
                      statusResult.stdout?.match(/port (\d+)/i) ||
                      startResult.stdout?.match(/listening on .*:(\d+)/i) ||
                      statusResult.stdout?.match(/listening on .*:(\d+)/i);

    if (portMatch && portMatch[1]) {
      port = parseInt(portMatch[1], 10);
      logger.info(`Detected MCP server running on port: ${port}`);
    } else {
      // If we can't detect the port, try to find it using netstat
      logger.info(`Port not found in command output, trying to detect using netstat...`);
      const netstatCommand = `netstat -tulpn | grep "${mcpName}" | awk '{print $4}' | awk -F: '{print $NF}'`;
      const netstatResult = await ssh.execCommand(netstatCommand);

      if (netstatResult.code === 0 && netstatResult.stdout.trim()) {
        const ports = netstatResult.stdout.trim().split('\n');
        if (ports.length > 0 && /^\d+$/.test(ports[0])) {
          port = parseInt(ports[0], 10);
          logger.info(`Detected MCP server running on port: ${port} (via netstat)`);
        }
      }

      // If we still can't find the port, use the default as a last resort
      if (!port) {
        port = config.get('mcp-server.mcp_terminal_command_default_port_1') || 8080;
        logger.warn(`Could not detect actual port, using default: ${port}`);
      }
    }

    // If status command fails, installation might still be successful but server not running
    if (statusResult.code !== 0) {
      logger.warn(`MCP server status check failed: ${statusResult.stderr}`);
      return {
        success: true,
        message: `MCP installed successfully in ${installDir}, but server status check failed. You may need to start it manually.`,
        host: sshConfig.ssh_host || sshConfig.host,
        port: port,
        installDir: installDir,
        warning: "Server status check failed. You may need to start it manually.",
        portDetectionMethod: port ? "detected" : "default"
      };
    }

    return {
      success: true,
      message: `MCP server installed successfully in ${installDir}`,
      host: sshConfig.ssh_host || sshConfig.host,
      port: port,
      installDir: installDir,
      portDetectionMethod: port ? "detected" : "default"
    };
  } catch (err) {
    // Make sure SSH connection is closed
    ssh.dispose();

    logger.error(`MCP installation failed: ${err.message}`);

    return {
      success: false,
      error: err.message || 'MCP installation failed'
    };
  }
}

module.exports = {
  testSSHConnection,
  executeSSHCommand,
  installMCPViaSSH,
  encryptPassword,
  decryptPassword,
  authenticateSSH,
  listDirectory,
  getFileInfo,
  createDirectory,
  installMCPInDirectory
};
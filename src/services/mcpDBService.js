/**
 * MCP Database Service
 * Database access layer for MCP configurations
 */

const { pool } = require('../database');
const { v4: uuidv4 } = require('uuid');
const { logger } = require('../utils/logger');
const { encryptPassword, decryptPassword } = require('./sshService');

/**
 * Get all SSH configurations for a user
 *
 * @param {string} userId - User ID
 * @returns {Promise<Array>} - Array of SSH configurations
 */
async function getUserSSHConfigurations(userId) {
  try {
    const query = `
      SELECT id, user_id, machine_nickname, ssh_host, ssh_port, ssh_user,
             ssh_auth_method, ssh_key_path, last_ssh_connection_status,
             last_ssh_error_message, created_at, updated_at
      FROM user_ssh_configurations
      WHERE user_id = $1
      ORDER BY created_at DESC
    `;

    const result = await pool.query(query, [userId]);
    return result.rows;
  } catch (err) {
    logger.error(`Failed to get SSH configurations for user ${userId}: ${err.message}`);
    throw err;
  }
}

/**
 * Get a specific SSH configuration
 *
 * @param {string} configId - Configuration ID
 * @param {string} userId - User ID
 * @returns {Promise<Object>} - SSH configuration
 */
async function getSSHConfiguration(configId, userId) {
  try {
    const query = `
      SELECT id, user_id, machine_nickname, ssh_host, ssh_port, ssh_user,
             ssh_auth_method, ssh_key_path, ssh_password_encrypted,
             last_ssh_connection_status, last_ssh_error_message,
             created_at, updated_at
      FROM user_ssh_configurations
      WHERE id = $1 AND user_id = $2
    `;

    const result = await pool.query(query, [configId, userId]);

    if (result.rows.length === 0) {
      return null;
    }

    const config = result.rows[0];

    // Don't return encrypted password to clients
    delete config.ssh_password_encrypted;

    return config;
  } catch (err) {
    logger.error(`Failed to get SSH configuration ${configId}: ${err.message}`);
    throw err;
  }
}

/**
 * Create a new SSH configuration
 *
 * @param {Object} config - SSH configuration
 * @param {string} userId - User ID
 * @returns {Promise<Object>} - Created SSH configuration
 */
async function createSSHConfiguration(config, userId) {
  const client = await pool.connect();

  logger.info(`Creating SSH configuration for user ${userId}: ${JSON.stringify({
    machine_nickname: config.machine_nickname,
    ssh_host: config.ssh_host,
    ssh_port: config.ssh_port,
    ssh_user: config.ssh_user,
    ssh_auth_method: config.ssh_auth_method,
    has_key_path: !!config.ssh_key_path,
    has_password: !!config.ssh_password
  })}`);

  try {
    await client.query('BEGIN');
    logger.debug('Transaction started');

    // Encrypt password if provided and auth method is password
    let passwordEncrypted = null;
    if (config.ssh_auth_method === 'password' && config.ssh_password) {
      passwordEncrypted = encryptPassword(config.ssh_password);
      logger.debug('Password encrypted successfully');
    }

    const query = `
      INSERT INTO user_ssh_configurations (
        id, user_id, machine_nickname, ssh_host, ssh_port, ssh_user,
        ssh_auth_method, ssh_key_path, ssh_password_encrypted
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
      RETURNING id, user_id, machine_nickname, ssh_host, ssh_port, ssh_user,
                ssh_auth_method, ssh_key_path, last_ssh_connection_status,
                created_at, updated_at
    `;

    const configId = config.id || uuidv4();
    logger.debug(`Using config ID: ${configId}`);

    const values = [
      configId,
      userId,
      config.machine_nickname,
      config.ssh_host,
      config.ssh_port,
      config.ssh_user,
      config.ssh_auth_method,
      config.ssh_key_path || null,
      passwordEncrypted
    ];

    logger.debug(`Executing query with values: ${JSON.stringify(values.map((v, i) => i === 8 ? 'ENCRYPTED' : v))}`);
    const result = await client.query(query, values);
    logger.debug(`Query executed, got ${result.rows.length} rows`);

    await client.query('COMMIT');
    logger.info(`SSH configuration created successfully with ID: ${configId}`);

    return result.rows[0];
  } catch (err) {
    await client.query('ROLLBACK');
    logger.error(`Failed to create SSH configuration: ${err.message}`);
    logger.error(`Error stack: ${err.stack}`);
    throw err;
  } finally {
    client.release();
    logger.debug('Database client released');
  }
}

/**
 * Update an existing SSH configuration
 *
 * @param {Object} config - SSH configuration
 * @param {string} userId - User ID
 * @returns {Promise<Object>} - Updated SSH configuration
 */
async function updateSSHConfiguration(config, userId) {
  const client = await pool.connect();

  try {
    await client.query('BEGIN');

    // Check if the config exists and belongs to the user
    const checkQuery = `
      SELECT id FROM user_ssh_configurations
      WHERE id = $1 AND user_id = $2
    `;

    const checkResult = await client.query(checkQuery, [config.id, userId]);

    if (checkResult.rows.length === 0) {
      throw new Error('SSH configuration not found or access denied');
    }

    // Encrypt password if provided and auth method is password
    let passwordEncrypted = null;
    if (config.ssh_auth_method === 'password') {
      if (config.ssh_password) {
        // User provided a new password
        passwordEncrypted = encryptPassword(config.ssh_password);
      } else {
        // No new password provided, keep the existing one
        const existingConfig = await getSSHConfiguration(config.id, userId);
        if (existingConfig && existingConfig.ssh_password_encrypted) {
          passwordEncrypted = existingConfig.ssh_password_encrypted;
        }
      }
    }

    const query = `
      UPDATE user_ssh_configurations
      SET machine_nickname = $1,
          ssh_host = $2,
          ssh_port = $3,
          ssh_user = $4,
          ssh_auth_method = $5,
          ssh_key_path = $6,
          ssh_password_encrypted = $7,
          updated_at = NOW()
      WHERE id = $8 AND user_id = $9
      RETURNING id, user_id, machine_nickname, ssh_host, ssh_port, ssh_user,
                ssh_auth_method, ssh_key_path, last_ssh_connection_status,
                created_at, updated_at
    `;

    const values = [
      config.machine_nickname,
      config.ssh_host,
      config.ssh_port,
      config.ssh_user,
      config.ssh_auth_method,
      config.ssh_key_path || null,
      passwordEncrypted,
      config.id,
      userId
    ];

    const result = await client.query(query, values);
    await client.query('COMMIT');

    return result.rows[0];
  } catch (err) {
    await client.query('ROLLBACK');
    logger.error(`Failed to update SSH configuration ${config.id}: ${err.message}`);
    throw err;
  } finally {
    client.release();
  }
}

/**
 * Delete an SSH configuration
 *
 * @param {string} configId - Configuration ID
 * @param {string} userId - User ID
 * @returns {Promise<boolean>} - Whether the configuration was deleted
 */
async function deleteSSHConfiguration(configId, userId) {
  try {
    const query = `
      DELETE FROM user_ssh_configurations
      WHERE id = $1 AND user_id = $2
      RETURNING id
    `;

    const result = await pool.query(query, [configId, userId]);
    return result.rowCount > 0;
  } catch (err) {
    logger.error(`Failed to delete SSH configuration ${configId}: ${err.message}`);
    throw err;
  }
}

/**
 * Update SSH connection status
 *
 * @param {string} configId - Configuration ID
 * @param {string} status - Connection status ('successful', 'failed', 'unknown')
 * @param {string} [errorMessage] - Error message if status is 'failed'
 * @returns {Promise<boolean>} - Whether the status was updated
 */
async function updateSSHConnectionStatus(configId, status, errorMessage = null) {
  try {
    const query = `
      UPDATE user_ssh_configurations
      SET last_ssh_connection_status = $1,
          last_ssh_error_message = $2,
          updated_at = NOW()
      WHERE id = $3
      RETURNING id
    `;

    const result = await pool.query(query, [status, errorMessage, configId]);
    return result.rowCount > 0;
  } catch (err) {
    logger.error(`Failed to update SSH connection status for ${configId}: ${err.message}`);
    throw err;
  }
}

/**
 * Get all MCP server configurations for a user
 *
 * @param {string} userId - User ID
 * @returns {Promise<Array>} - Array of MCP server configurations
 */
async function getUserMCPServerConfigurations(userId) {
  try {
    // First, check which columns exist in the table
    const checkColumnsQuery = `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'user_mcp_server_configurations'
    `;

    const columnsResult = await pool.query(checkColumnsQuery);
    const columns = columnsResult.rows.map(row => row.column_name);

    logger.info(`Available columns in user_mcp_server_configurations for getUserMCPServerConfigurations: ${columns.join(', ')}`);

    // Use the correct column names based on what's available in the database
    const hasServerName = columns.includes('server_name');
    const hasMcpNickname = columns.includes('mcp_nickname');
    const hasLastConnectionStatus = columns.includes('last_connection_status');
    const hasMcpConnectionStatus = columns.includes('mcp_connection_status');
    const hasLastErrorMessage = columns.includes('last_error_message');
    const hasMcpLastErrorMessage = columns.includes('mcp_last_error_message');

    let query;

    if (hasServerName) {
      query = `
        SELECT id, user_id, server_name as name, mcp_host, mcp_port,
               is_default, ${hasLastConnectionStatus ? 'last_connection_status' : 'NULL'} as connection_status,
               ${hasLastErrorMessage ? 'last_error_message' : 'NULL'} as error_message,
               created_at, updated_at
        FROM user_mcp_server_configurations
        WHERE user_id = $1
        ORDER BY is_default DESC, created_at DESC
      `;
    } else if (hasMcpNickname) {
      query = `
        SELECT id, user_id, mcp_nickname as name, mcp_host, mcp_port,
               is_default, ${hasMcpConnectionStatus ? 'mcp_connection_status' : 'NULL'} as connection_status,
               ${hasMcpLastErrorMessage ? 'mcp_last_error_message' : 'NULL'} as error_message,
               created_at, updated_at
        FROM user_mcp_server_configurations
        WHERE user_id = $1
        ORDER BY is_default DESC, created_at DESC
      `;
    } else {
      throw new Error('Could not determine the correct column name for MCP server name');
    }

    const result = await pool.query(query, [userId]);

    // Map the results to a consistent format
    const mappedResults = result.rows.map(row => ({
      id: row.id,
      user_id: row.user_id,
      server_name: row.name,
      mcp_host: row.mcp_host,
      mcp_port: row.mcp_port,
      is_default: row.is_default,
      last_connection_status: row.connection_status,
      last_error_message: row.error_message,
      created_at: row.created_at,
      updated_at: row.updated_at
    }));

    return mappedResults;
  } catch (err) {
    logger.error(`Failed to get MCP server configurations for user ${userId}: ${err.message}`);
    throw err;
  }
}

/**
 * Get a specific MCP server configuration
 *
 * @param {string} configId - Configuration ID
 * @param {string} userId - User ID
 * @returns {Promise<Object>} - MCP server configuration
 */
async function getMCPServerConfiguration(configId, userId) {
  try {
    // First, check which columns exist in the table
    const checkColumnsQuery = `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'user_mcp_server_configurations'
    `;

    const columnsResult = await pool.query(checkColumnsQuery);
    const columns = columnsResult.rows.map(row => row.column_name);

    // Use the correct column names based on what's available in the database
    const hasServerName = columns.includes('server_name');
    const hasMcpNickname = columns.includes('mcp_nickname');
    const hasLastConnectionStatus = columns.includes('last_connection_status');
    const hasMcpConnectionStatus = columns.includes('mcp_connection_status');
    const hasLastErrorMessage = columns.includes('last_error_message');
    const hasMcpLastErrorMessage = columns.includes('mcp_last_error_message');

    let query;

    if (hasServerName) {
      query = `
        SELECT id, user_id, server_name as name, mcp_host, mcp_port,
               is_default, ${hasLastConnectionStatus ? 'last_connection_status' : 'NULL'} as connection_status,
               ${hasLastErrorMessage ? 'last_error_message' : 'NULL'} as error_message,
               created_at, updated_at
        FROM user_mcp_server_configurations
        WHERE id = $1 AND user_id = $2
      `;
    } else if (hasMcpNickname) {
      query = `
        SELECT id, user_id, mcp_nickname as name, mcp_host, mcp_port,
               is_default, ${hasMcpConnectionStatus ? 'mcp_connection_status' : 'NULL'} as connection_status,
               ${hasMcpLastErrorMessage ? 'mcp_last_error_message' : 'NULL'} as error_message,
               created_at, updated_at
        FROM user_mcp_server_configurations
        WHERE id = $1 AND user_id = $2
      `;
    } else {
      throw new Error('Could not determine the correct column name for MCP server name');
    }

    const result = await pool.query(query, [configId, userId]);

    if (result.rows.length === 0) {
      return null;
    }

    // Map the result to a consistent format
    const row = result.rows[0];
    return {
      id: row.id,
      user_id: row.user_id,
      server_name: row.name,
      mcp_host: row.mcp_host,
      mcp_port: row.mcp_port,
      is_default: row.is_default,
      last_connection_status: row.connection_status,
      last_error_message: row.error_message,
      created_at: row.created_at,
      updated_at: row.updated_at
    };
  } catch (err) {
    logger.error(`Failed to get MCP server configuration ${configId}: ${err.message}`);
    throw err;
  }
}

/**
 * Create a new MCP server configuration
 *
 * @param {Object} config - MCP server configuration
 * @param {string} userId - User ID
 * @returns {Promise<Object>} - Created MCP server configuration
 */
async function createMCPServerConfiguration(config, userId) {
  const client = await pool.connect();

  try {
    await client.query('BEGIN');

    // If this is set as default, unset any existing default
    if (config.is_default) {
      await client.query(`
        UPDATE user_mcp_server_configurations
        SET is_default = false
        WHERE user_id = $1 AND is_default = true
      `, [userId]);
    }

    // Log the configuration for debugging
    logger.info(`Creating MCP server configuration: ${JSON.stringify({
      mcp_host: config.mcp_host,
      mcp_port: config.mcp_port,
      is_default: config.is_default,
      server_name: config.server_name
    })}`);

    // Check if the table has server_name or mcp_nickname column
    const checkColumnsQuery = `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'user_mcp_server_configurations'
    `;

    const columnsResult = await client.query(checkColumnsQuery);
    const columns = columnsResult.rows.map(row => row.column_name);

    logger.info(`Available columns in user_mcp_server_configurations: ${columns.join(', ')}`);

    // Use the correct column name based on what's available in the database
    const hasServerName = columns.includes('server_name');
    const hasMcpNickname = columns.includes('mcp_nickname');

    // Generate a server name/nickname
    const serverName = config.server_name || `MCP on ${config.mcp_host}:${config.mcp_port}`;

    // Check if a configuration with this name already exists for this user
    let checkExistingQuery;
    if (hasServerName) {
      checkExistingQuery = `
        SELECT id FROM user_mcp_server_configurations
        WHERE user_id = $1 AND server_name = $2
      `;
    } else if (hasMcpNickname) {
      checkExistingQuery = `
        SELECT id FROM user_mcp_server_configurations
        WHERE user_id = $1 AND mcp_nickname = $2
      `;
    } else {
      throw new Error('Could not determine the correct column name for MCP server name');
    }

    const existingResult = await client.query(checkExistingQuery, [userId, serverName]);

    // If a configuration with this name already exists, update it instead of creating a new one
    if (existingResult.rows.length > 0) {
      const existingId = existingResult.rows[0].id;
      logger.info(`Found existing MCP configuration with ID ${existingId}, updating instead of creating new`);

      let updateQuery;
      if (hasServerName) {
        updateQuery = `
          UPDATE user_mcp_server_configurations
          SET mcp_host = $1,
              mcp_port = $2,
              is_default = $3,
              updated_at = NOW()
          WHERE id = $4
          RETURNING id, user_id, server_name, mcp_host, mcp_port,
                    is_default, last_connection_status, created_at, updated_at
        `;
      } else if (hasMcpNickname) {
        updateQuery = `
          UPDATE user_mcp_server_configurations
          SET mcp_host = $1,
              mcp_port = $2,
              is_default = $3,
              updated_at = NOW()
          WHERE id = $4
          RETURNING id, user_id, mcp_nickname, mcp_host, mcp_port,
                    is_default, mcp_connection_status, created_at, updated_at
        `;
      }

      const updateResult = await client.query(updateQuery, [
        config.mcp_host,
        config.mcp_port,
        config.is_default || false,
        existingId
      ]);

      await client.query('COMMIT');
      return updateResult.rows[0];
    }

    // If no existing configuration found, create a new one
    // Generate a unique name if needed
    let uniqueServerName = serverName;
    let counter = 1;
    let isUnique = false;

    while (!isUnique) {
      const checkUniqueResult = await client.query(checkExistingQuery, [userId, uniqueServerName]);
      if (checkUniqueResult.rows.length === 0) {
        isUnique = true;
      } else {
        uniqueServerName = `${serverName} (${counter})`;
        counter++;
      }
    }

    // Now create the new configuration with the unique name
    const configId = config.id || uuidv4();
    let query;
    let values;

    if (hasServerName) {
      query = `
        INSERT INTO user_mcp_server_configurations (
          id, user_id, server_name, mcp_host, mcp_port, is_default
        ) VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id, user_id, server_name, mcp_host, mcp_port,
                  is_default, last_connection_status, created_at, updated_at
      `;

      values = [
        configId,
        userId,
        uniqueServerName,
        config.mcp_host,
        config.mcp_port,
        config.is_default || false
      ];
    } else if (hasMcpNickname) {
      query = `
        INSERT INTO user_mcp_server_configurations (
          id, user_id, mcp_nickname, mcp_host, mcp_port, is_default
        ) VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id, user_id, mcp_nickname, mcp_host, mcp_port,
                  is_default, mcp_connection_status, created_at, updated_at
      `;

      values = [
        configId,
        userId,
        uniqueServerName,
        config.mcp_host,
        config.mcp_port,
        config.is_default || false
      ];
    }

    const result = await client.query(query, values);
    await client.query('COMMIT');

    return result.rows[0];
  } catch (err) {
    await client.query('ROLLBACK');
    logger.error(`Failed to create MCP server configuration: ${err.message}`);
    throw err;
  } finally {
    client.release();
  }
}

/**
 * Update an existing MCP server configuration
 *
 * @param {Object} config - MCP server configuration
 * @param {string} userId - User ID
 * @returns {Promise<Object>} - Updated MCP server configuration
 */
async function updateMCPServerConfiguration(config, userId) {
  const client = await pool.connect();

  try {
    await client.query('BEGIN');

    // Check if the config exists and belongs to the user
    const checkQuery = `
      SELECT id FROM user_mcp_server_configurations
      WHERE id = $1 AND user_id = $2
    `;

    const checkResult = await client.query(checkQuery, [config.id, userId]);

    if (checkResult.rows.length === 0) {
      throw new Error('MCP server configuration not found or access denied');
    }

    // If this is set as default, unset any existing default
    if (config.is_default) {
      await client.query(`
        UPDATE user_mcp_server_configurations
        SET is_default = false
        WHERE user_id = $1 AND id != $2 AND is_default = true
      `, [userId, config.id]);
    }

    const query = `
      UPDATE user_mcp_server_configurations
      SET server_name = $1,
          mcp_host = $2,
          mcp_port = $3,
          is_default = $4,
          updated_at = NOW()
      WHERE id = $5 AND user_id = $6
      RETURNING id, user_id, server_name, mcp_host, mcp_port,
                is_default, last_connection_status, created_at, updated_at
    `;

    const values = [
      config.server_name,
      config.mcp_host,
      config.mcp_port,
      config.is_default || false,
      config.id,
      userId
    ];

    const result = await client.query(query, values);
    await client.query('COMMIT');

    return result.rows[0];
  } catch (err) {
    await client.query('ROLLBACK');
    logger.error(`Failed to update MCP server configuration ${config.id}: ${err.message}`);
    throw err;
  } finally {
    client.release();
  }
}

/**
 * Delete an MCP server configuration
 *
 * @param {string} configId - Configuration ID
 * @param {string} userId - User ID
 * @returns {Promise<boolean>} - Whether the configuration was deleted
 */
async function deleteMCPServerConfiguration(configId, userId) {
  try {
    const query = `
      DELETE FROM user_mcp_server_configurations
      WHERE id = $1 AND user_id = $2
      RETURNING id
    `;

    const result = await pool.query(query, [configId, userId]);
    return result.rowCount > 0;
  } catch (err) {
    logger.error(`Failed to delete MCP server configuration ${configId}: ${err.message}`);
    throw err;
  }
}

/**
 * Update MCP connection status
 *
 * @param {string} configId - Configuration ID
 * @param {string} status - Connection status ('connected', 'error', 'disconnected')
 * @param {string} [errorMessage] - Error message if status is 'error'
 * @returns {Promise<boolean>} - Whether the status was updated
 */
async function updateMCPConnectionStatus(configId, status, errorMessage = null) {
  try {
    // First, check which columns exist in the table
    const checkColumnsQuery = `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'user_mcp_server_configurations'
    `;

    const columnsResult = await pool.query(checkColumnsQuery);
    const columns = columnsResult.rows.map(row => row.column_name);

    // Use the correct column names based on what's available in the database
    const hasLastConnectionStatus = columns.includes('last_connection_status');
    const hasMcpConnectionStatus = columns.includes('mcp_connection_status');
    const hasLastErrorMessage = columns.includes('last_error_message');
    const hasMcpLastErrorMessage = columns.includes('mcp_last_error_message');

    let query;

    if (hasLastConnectionStatus && hasLastErrorMessage) {
      query = `
        UPDATE user_mcp_server_configurations
        SET last_connection_status = $1,
            last_error_message = $2,
            updated_at = NOW()
        WHERE id = $3
        RETURNING id
      `;
    } else if (hasMcpConnectionStatus && hasMcpLastErrorMessage) {
      query = `
        UPDATE user_mcp_server_configurations
        SET mcp_connection_status = $1,
            mcp_last_error_message = $2,
            updated_at = NOW()
        WHERE id = $3
        RETURNING id
      `;
    } else {
      // If we can't determine the correct columns, log a warning and return success
      logger.warn(`Could not determine the correct column names for MCP connection status. Available columns: ${columns.join(', ')}`);
      return true;
    }

    const result = await pool.query(query, [status, errorMessage, configId]);
    return result.rowCount > 0;
  } catch (err) {
    logger.error(`Failed to update MCP connection status for ${configId}: ${err.message}`);
    throw err;
  }
}

/**
 * Get the default MCP server configuration for a user
 *
 * @param {string} userId - User ID
 * @returns {Promise<Object>} - Default MCP server configuration
 */
async function getDefaultMCPServerConfiguration(userId) {
  try {
    // First, check which columns exist in the table
    const checkColumnsQuery = `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'user_mcp_server_configurations'
    `;

    const columnsResult = await pool.query(checkColumnsQuery);
    const columns = columnsResult.rows.map(row => row.column_name);

    // Use the correct column names based on what's available in the database
    const hasServerName = columns.includes('server_name');
    const hasMcpNickname = columns.includes('mcp_nickname');
    const hasLastConnectionStatus = columns.includes('last_connection_status');
    const hasMcpConnectionStatus = columns.includes('mcp_connection_status');
    const hasLastErrorMessage = columns.includes('last_error_message');
    const hasMcpLastErrorMessage = columns.includes('mcp_last_error_message');

    let query;

    if (hasServerName) {
      query = `
        SELECT id, user_id, server_name as name, mcp_host, mcp_port,
               is_default, ${hasLastConnectionStatus ? 'last_connection_status' : 'NULL'} as connection_status,
               ${hasLastErrorMessage ? 'last_error_message' : 'NULL'} as error_message,
               created_at, updated_at
        FROM user_mcp_server_configurations
        WHERE user_id = $1 AND is_default = true
        LIMIT 1
      `;
    } else if (hasMcpNickname) {
      query = `
        SELECT id, user_id, mcp_nickname as name, mcp_host, mcp_port,
               is_default, ${hasMcpConnectionStatus ? 'mcp_connection_status' : 'NULL'} as connection_status,
               ${hasMcpLastErrorMessage ? 'mcp_last_error_message' : 'NULL'} as error_message,
               created_at, updated_at
        FROM user_mcp_server_configurations
        WHERE user_id = $1 AND is_default = true
        LIMIT 1
      `;
    } else {
      throw new Error('Could not determine the correct column name for MCP server name');
    }

    const result = await pool.query(query, [userId]);

    if (result.rows.length === 0) {
      return null;
    }

    // Map the result to a consistent format
    const row = result.rows[0];
    return {
      id: row.id,
      user_id: row.user_id,
      server_name: row.name,
      mcp_host: row.mcp_host,
      mcp_port: row.mcp_port,
      is_default: row.is_default,
      last_connection_status: row.connection_status,
      last_error_message: row.error_message,
      created_at: row.created_at,
      updated_at: row.updated_at
    };
  } catch (err) {
    logger.error(`Failed to get default MCP server configuration for user ${userId}: ${err.message}`);
    throw err;
  }
}

/**
 * Get SSH configuration with decrypted password (for internal use only)
 *
 * @param {string} configId - Configuration ID
 * @param {string} userId - User ID
 * @returns {Promise<Object>} - SSH configuration with decrypted password
 */
async function getSSHConfigurationWithPassword(configId, userId) {
  try {
    const query = `
      SELECT id, user_id, machine_nickname, ssh_host, ssh_port, ssh_user,
             ssh_auth_method, ssh_key_path, ssh_password_encrypted,
             last_ssh_connection_status, last_ssh_error_message
      FROM user_ssh_configurations
      WHERE id = $1 AND user_id = $2
    `;

    const result = await pool.query(query, [configId, userId]);

    if (result.rows.length === 0) {
      return null;
    }

    const config = result.rows[0];

    // Decrypt password if auth method is password
    if (config.ssh_auth_method === 'password' && config.ssh_password_encrypted) {
      config.ssh_password = decryptPassword(config.ssh_password_encrypted);
    }

    // Don't return encrypted password
    delete config.ssh_password_encrypted;

    return config;
  } catch (err) {
    logger.error(`Failed to get SSH configuration with password ${configId}: ${err.message}`);
    throw err;
  }
}

module.exports = {
  // SSH configurations
  getUserSSHConfigurations,
  getSSHConfiguration,
  createSSHConfiguration,
  updateSSHConfiguration,
  deleteSSHConfiguration,
  updateSSHConnectionStatus,
  getSSHConfigurationWithPassword,

  // MCP server configurations
  getUserMCPServerConfigurations,
  getMCPServerConfiguration,
  createMCPServerConfiguration,
  updateMCPServerConfiguration,
  deleteMCPServerConfiguration,
  updateMCPConnectionStatus,
  getDefaultMCPServerConfiguration
};
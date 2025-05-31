/**
 * Migration to create user_mcp_server_configurations table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Enable UUID extension if not already enabled
    await client.query(`CREATE EXTENSION IF NOT EXISTS "uuid-ossp"`);
    
    // Create the user_mcp_server_configurations table
    await client.query(`
      CREATE TABLE IF NOT EXISTS user_mcp_server_configurations (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id UUID NOT NULL,
        ssh_configuration_id UUID,
        mcp_nickname VARCHAR(255) NOT NULL,
        mcp_host VARCHAR(255) NOT NULL,
        mcp_port INTEGER NOT NULL,
        mcp_connection_status VARCHAR(50) DEFAULT 'unknown',
        mcp_last_error_message TEXT,
        mcp_discovered_tools_schema JSONB,
        mcp_server_version VARCHAR(100),
        is_default BOOLEAN DEFAULT false,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (ssh_configuration_id) REFERENCES user_ssh_configurations(id) ON DELETE SET NULL,
        UNIQUE(user_id, mcp_nickname)
      )
    `);
    
    // Create indexes
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_user_mcp_server_configurations_user_id ON user_mcp_server_configurations(user_id)
    `);

    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_user_mcp_server_configurations_ssh_config_id ON user_mcp_server_configurations(ssh_configuration_id)
    `);

    await client.query('COMMIT');
    console.log('Migration 016: User MCP server configurations table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 016 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TABLE IF EXISTS user_mcp_server_configurations CASCADE');

    await client.query('COMMIT');
    console.log('Migration 016: User MCP server configurations table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 016 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
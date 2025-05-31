/**
 * Migration to create user_mcp_server_configurations table
 */

const { pool } = require('../database');

exports.up = async function() {
  return pool.query(`
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
    CREATE TABLE IF NOT EXISTS user_mcp_server_configurations (
      id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
      user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      server_name VARCHAR(255) NOT NULL,
      mcp_host VARCHAR(255) NOT NULL,
      mcp_port INTEGER NOT NULL,
      is_default BOOLEAN NOT NULL DEFAULT false,
      last_connection_status VARCHAR(50),
      last_error_message TEXT,
      created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_user_mcp_server_configurations_user_id ON user_mcp_server_configurations(user_id);
    CREATE INDEX IF NOT EXISTS idx_user_mcp_server_configurations_user_default ON user_mcp_server_configurations(user_id, is_default);
  `);
};

exports.down = async function() {
  return pool.query('DROP TABLE IF EXISTS user_mcp_server_configurations;');
}; 
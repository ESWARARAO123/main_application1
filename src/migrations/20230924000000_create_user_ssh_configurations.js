/**
 * Migration to create user_ssh_configurations table
 */

const { pool } = require('../database');

exports.up = async function() {
  return pool.query(`
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
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
      last_ssh_connection_status VARCHAR(50),
      last_ssh_error_message TEXT,
      created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_user_ssh_configurations_user_id ON user_ssh_configurations(user_id);
  `);
};

exports.down = async function() {
  return pool.query('DROP TABLE IF EXISTS user_ssh_configurations;');
}; 
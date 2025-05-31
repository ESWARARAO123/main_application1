/**
 * Migration to create user_ssh_configurations table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Enable UUID extension if not already enabled
    await client.query(`CREATE EXTENSION IF NOT EXISTS "uuid-ossp"`);
    
    // Create the user_ssh_configurations table
    await client.query(`
      CREATE TABLE IF NOT EXISTS user_ssh_configurations (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id UUID NOT NULL,
        machine_nickname VARCHAR(255) NOT NULL,
        ssh_host VARCHAR(255) NOT NULL,
        ssh_port INTEGER NOT NULL DEFAULT 22,
        ssh_user VARCHAR(255) NOT NULL,
        ssh_auth_method VARCHAR(50) NOT NULL,
        ssh_password_encrypted TEXT,
        ssh_key_path TEXT,
        last_ssh_connection_status VARCHAR(50) DEFAULT 'unknown',
        last_ssh_error_message TEXT,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, machine_nickname)
      )
    `);
    
    // Create indexes
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_user_ssh_configurations_user_id ON user_ssh_configurations(user_id)
    `);

    await client.query('COMMIT');
    console.log('Migration 015: User SSH configurations table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 015 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TABLE IF EXISTS user_ssh_configurations CASCADE');

    await client.query('COMMIT');
    console.log('Migration 015: User SSH configurations table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 015 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
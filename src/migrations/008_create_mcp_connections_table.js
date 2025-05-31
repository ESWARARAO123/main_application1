/**
 * Migration to create mcp_connections table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the mcp_connections table
    await client.query(`
      CREATE TABLE IF NOT EXISTS mcp_connections (
        id BIGSERIAL PRIMARY KEY,
        user_id UUID NOT NULL,
        mcp_host VARCHAR(100) NOT NULL,
        mcp_port INTEGER NOT NULL,
        status VARCHAR(20) DEFAULT 'disconnected',
        last_file_path TEXT,
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, mcp_host, mcp_port)
      )
    `);

    await client.query('COMMIT');
    console.log('Migration 008: MCP connections table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 008 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TABLE IF EXISTS mcp_connections CASCADE');

    await client.query('COMMIT');
    console.log('Migration 008: MCP connections table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 008 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
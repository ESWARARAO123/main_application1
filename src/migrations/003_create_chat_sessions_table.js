/**
 * Migration to create chat_sessions table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the chat_sessions table
    await client.query(`
      CREATE TABLE IF NOT EXISTS chat_sessions (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        user_id UUID NOT NULL,
        title VARCHAR(255) NOT NULL,
        last_message_timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT true,
        use_rag BOOLEAN DEFAULT true,
        rag_collections JSONB,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
      )
    `);

    // Create indexes for performance
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)
    `);

    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_chat_sessions_timestamp ON chat_sessions(last_message_timestamp)
    `);

    await client.query('COMMIT');
    console.log('Migration 003: Chat sessions table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 003 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TABLE IF EXISTS chat_sessions CASCADE');

    await client.query('COMMIT');
    console.log('Migration 003: Chat sessions table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 003 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
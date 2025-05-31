/**
 * Migration to create messages table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the messages table
    await client.query(`
      CREATE TABLE IF NOT EXISTS messages (
        id BIGSERIAL PRIMARY KEY,
        user_id UUID NOT NULL,
        message TEXT NOT NULL,
        response TEXT,
        timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        session_id UUID,
        model_id UUID,
        retrieved_chunks JSONB,
        document_id INTEGER,
        file_path TEXT,
        is_context_update BOOLEAN DEFAULT false,
        is_context_flag BOOLEAN DEFAULT false,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
        FOREIGN KEY (model_id) REFERENCES ai_models(id)
      )
    `);

    // Add comments on context columns
    await client.query(`
      COMMENT ON COLUMN messages.is_context_update IS 'Flag indicating if this message is a context update'
    `);

    await client.query(`
      COMMENT ON COLUMN messages.is_context_flag IS 'Flag indicating if this message contains context information'
    `);

    // Create indexes for performance
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)
    `);

    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_messages_context_flags ON messages(is_context_update, is_context_flag)
    `);

    await client.query('COMMIT');
    console.log('Migration 006: Messages table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 006 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TABLE IF EXISTS messages CASCADE');

    await client.query('COMMIT');
    console.log('Migration 006: Messages table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 006 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
/**
 * Migration to add documents table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create documents table with complete structure (without document_collections FK for now)
    await client.query(`
      CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        user_id UUID NOT NULL,
        filename VARCHAR(255) DEFAULT '' NOT NULL,
        original_name VARCHAR(255) NOT NULL,
        file_path TEXT NOT NULL,
        file_type VARCHAR(100) NOT NULL,
        file_size BIGINT NOT NULL,
        status VARCHAR(50) DEFAULT 'UPLOADED',
        processing_error TEXT,
        collection_id UUID,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        last_activity_timestamp TIMESTAMP WITH TIME ZONE,
        mime_type VARCHAR(255),
        session_id UUID,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE SET NULL
      )
    `);

    // Create index on session_id
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id)
    `);

    await client.query('COMMIT');
    console.log('Migration 005: Documents table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 005 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Drop documents table
    await client.query('DROP TABLE IF EXISTS documents');

    await client.query('COMMIT');
    console.log('Migration 005: Documents table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 005 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down };

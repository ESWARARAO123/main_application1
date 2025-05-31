/**
 * Migration to create document_chunks table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the document_chunks table
    await client.query(`
      CREATE TABLE IF NOT EXISTS document_chunks (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        document_id UUID NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        vector_id VARCHAR(255),
        embedding BYTEA,
        token_count INTEGER,
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB
      )
    `);

    // Create indexes for performance
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id)
    `);

    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_document_chunks_vector_id ON document_chunks(vector_id)
    `);

    await client.query('COMMIT');
    console.log('Migration 013: Document chunks table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 013 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TABLE IF EXISTS document_chunks CASCADE');

    await client.query('COMMIT');
    console.log('Migration 013: Document chunks table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 013 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
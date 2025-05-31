/**
 * Migration to create rag_settings table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the rag_settings table
    await client.query(`
      CREATE TABLE IF NOT EXISTS rag_settings (
        id SERIAL PRIMARY KEY,
        embedding_model VARCHAR(255) DEFAULT 'ollama/nomic-embed-text',
        chunk_size INTEGER DEFAULT 1000,
        chunk_overlap INTEGER DEFAULT 200,
        similarity_top_k INTEGER DEFAULT 4,
        search_type VARCHAR(20) DEFAULT 'similarity',
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        config JSONB
      )
    `);

    // Create trigger for updated_at timestamp
    await client.query(`
      DROP TRIGGER IF EXISTS update_rag_settings_timestamp ON rag_settings;
      CREATE TRIGGER update_rag_settings_timestamp 
      BEFORE UPDATE ON rag_settings 
      FOR EACH ROW 
      EXECUTE FUNCTION update_timestamp_column();
    `);

    await client.query('COMMIT');
    console.log('Migration 014: RAG settings table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 014 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TRIGGER IF EXISTS update_rag_settings_timestamp ON rag_settings');
    await client.query('DROP TABLE IF EXISTS rag_settings CASCADE');

    await client.query('COMMIT');
    console.log('Migration 014: RAG settings table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 014 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
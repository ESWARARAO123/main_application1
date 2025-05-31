/**
 * Migration to create document_collections table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the document_collections table
    await client.query(`
      CREATE TABLE IF NOT EXISTS document_collections (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        user_id UUID NOT NULL,
        vector_store_id UUID,
        embedding_model VARCHAR(255) DEFAULT 'ollama/nomic-embed-text',
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT true,
        metadata JSONB,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (vector_store_id) REFERENCES vector_stores(id) ON DELETE SET NULL
      )
    `);

    // Create indexes for performance
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_document_collections_user_id ON document_collections(user_id)
    `);

    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_document_collections_is_active ON document_collections(is_active)
    `);

    // Create trigger for updated_at timestamp
    await client.query(`
      DROP TRIGGER IF EXISTS update_document_collections_timestamp ON document_collections;
      CREATE TRIGGER update_document_collections_timestamp 
      BEFORE UPDATE ON document_collections 
      FOR EACH ROW 
      EXECUTE FUNCTION update_timestamp_column();
    `);

    await client.query('COMMIT');
    console.log('Migration 012: Document collections table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 012 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TRIGGER IF EXISTS update_document_collections_timestamp ON document_collections');
    await client.query('DROP TABLE IF EXISTS document_collections CASCADE');

    await client.query('COMMIT');
    console.log('Migration 012: Document collections table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 012 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
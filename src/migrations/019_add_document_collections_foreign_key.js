/**
 * Migration to add foreign key constraint from documents to document_collections
 * This migration runs after document_collections table is created
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Add foreign key constraint from documents.collection_id to document_collections.id
    await client.query(`
      ALTER TABLE documents 
      ADD CONSTRAINT fk_documents_collection_id 
      FOREIGN KEY (collection_id) REFERENCES document_collections(id) ON DELETE SET NULL
    `);

    // Create index on collection_id for performance
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_documents_collection_id ON documents(collection_id)
    `);

    await client.query('COMMIT');
    console.log('Migration 019: Added foreign key constraint from documents to document_collections');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 019 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Remove the foreign key constraint and index
    await client.query(`
      ALTER TABLE documents 
      DROP CONSTRAINT IF EXISTS fk_documents_collection_id
    `);

    await client.query(`
      DROP INDEX IF EXISTS idx_documents_collection_id
    `);

    await client.query('COMMIT');
    console.log('Migration 019: Removed foreign key constraint from documents to document_collections');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 019 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
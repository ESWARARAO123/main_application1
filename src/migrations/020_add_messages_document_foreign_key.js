/**
 * Migration to add foreign key constraint from messages to documents
 * This migration runs after both messages and documents tables are created
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Add foreign key constraint from messages.document_id to documents.id
    await client.query(`
      ALTER TABLE messages 
      ADD CONSTRAINT fk_messages_document_id 
      FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
    `);

    // Create index on document_id for performance
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_messages_document_id ON messages(document_id)
    `);

    await client.query('COMMIT');
    console.log('Migration 020: Added foreign key constraint from messages to documents');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 020 failed:', error);
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
      ALTER TABLE messages 
      DROP CONSTRAINT IF EXISTS fk_messages_document_id
    `);

    await client.query(`
      DROP INDEX IF EXISTS idx_messages_document_id
    `);

    await client.query('COMMIT');
    console.log('Migration 020: Removed foreign key constraint from messages to documents');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 020 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
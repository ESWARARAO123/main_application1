/**
 * Migration to add documents table and update messages table to support file attachments
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create documents table
    await client.query(`
      CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        user_id UUID NOT NULL,
        filename VARCHAR(255) NOT NULL,
        original_name VARCHAR(255) NOT NULL,
        file_path TEXT NOT NULL,
        file_type VARCHAR(100) NOT NULL,
        file_size BIGINT NOT NULL,
        status VARCHAR(50) DEFAULT 'UPLOADED',
        processing_error TEXT,
        collection_id UUID,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
      )
    `);

    // Add file_path and document_id columns to messages table
    await client.query(`
      ALTER TABLE messages
      ADD COLUMN IF NOT EXISTS file_path TEXT,
      ADD COLUMN IF NOT EXISTS document_id INTEGER
    `);

    const checkConstraintExists = await client.query(`
      SELECT constraint_name 
      FROM information_schema.table_constraints 
      WHERE constraint_name = 'fk_document' 
      AND table_name = 'messages'
    `);

    if (checkConstraintExists.rows.length === 0) {
      // Only add the constraint if it doesn't exist
      await client.query(`
        ALTER TABLE messages 
        ADD CONSTRAINT fk_document 
        FOREIGN KEY (document_id) REFERENCES documents(id)
      `);
    }

    await client.query('COMMIT');
    console.log('Migration 005: Documents table created and messages table updated');
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

    // Remove foreign key constraint and columns from messages table
    await client.query(`
      ALTER TABLE messages
      DROP CONSTRAINT IF EXISTS fk_document,
      DROP COLUMN IF EXISTS file_path,
      DROP COLUMN IF EXISTS document_id
    `);

    // Drop documents table
    await client.query('DROP TABLE IF EXISTS documents');

    await client.query('COMMIT');
    console.log('Migration 005: Rolled back documents table and messages table changes');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 005 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down };

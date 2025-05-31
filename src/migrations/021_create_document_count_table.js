/**
 * Migration to create document_count table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Drop the existing document_count table if it exists (it has wrong structure)
    await client.query(`DROP TABLE IF EXISTS document_count CASCADE`);

    // Create the document_count table with correct structure
    await client.query(`
      CREATE TABLE document_count (
        id SERIAL PRIMARY KEY,
        user_id UUID NOT NULL,
        total_documents INTEGER DEFAULT 0,
        total_size BIGINT DEFAULT 0,
        last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
      )
    `);

    // Create unique index on user_id
    await client.query(`
      CREATE UNIQUE INDEX idx_document_count_user_id ON document_count(user_id)
    `);

    // Create trigger to update document count when documents are added/removed
    await client.query(`
      CREATE OR REPLACE FUNCTION update_document_count()
      RETURNS TRIGGER AS $$
      BEGIN
          IF TG_OP = 'INSERT' THEN
              INSERT INTO document_count (user_id, total_documents, total_size)
              VALUES (NEW.user_id, 1, NEW.file_size)
              ON CONFLICT (user_id) 
              DO UPDATE SET 
                  total_documents = document_count.total_documents + 1,
                  total_size = document_count.total_size + NEW.file_size,
                  last_updated = CURRENT_TIMESTAMP;
              RETURN NEW;
          ELSIF TG_OP = 'DELETE' THEN
              UPDATE document_count 
              SET 
                  total_documents = GREATEST(total_documents - 1, 0),
                  total_size = GREATEST(total_size - OLD.file_size, 0),
                  last_updated = CURRENT_TIMESTAMP
              WHERE user_id = OLD.user_id;
              RETURN OLD;
          END IF;
          RETURN NULL;
      END;
      $$ LANGUAGE plpgsql;
    `);

    await client.query(`
      DROP TRIGGER IF EXISTS documents_count_trigger ON documents;
      CREATE TRIGGER documents_count_trigger
      AFTER INSERT OR DELETE ON documents
      FOR EACH ROW
      EXECUTE FUNCTION update_document_count();
    `);

    await client.query('COMMIT');
    console.log('Migration 021: Document count table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 021 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TRIGGER IF EXISTS documents_count_trigger ON documents');
    await client.query('DROP FUNCTION IF EXISTS update_document_count()');
    await client.query('DROP TABLE IF EXISTS document_count CASCADE');

    await client.query('COMMIT');
    console.log('Migration 021: Document count table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 021 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
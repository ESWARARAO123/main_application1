/**
 * Migration to create vector_stores table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the vector_stores table
    await client.query(`
      CREATE TABLE IF NOT EXISTS vector_stores (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        store_type VARCHAR(50) DEFAULT 'chroma' NOT NULL,
        connection_string TEXT,
        is_active BOOLEAN DEFAULT true,
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        config JSONB
      )
    `);

    // Create indexes for performance
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_vector_stores_is_active ON vector_stores(is_active)
    `);

    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_vector_stores_name ON vector_stores(name)
    `);

    // Create trigger for updated_at timestamp
    await client.query(`
      DROP TRIGGER IF EXISTS update_vector_stores_timestamp ON vector_stores;
      CREATE TRIGGER update_vector_stores_timestamp 
      BEFORE UPDATE ON vector_stores 
      FOR EACH ROW 
      EXECUTE FUNCTION update_timestamp_column();
    `);

    await client.query('COMMIT');
    console.log('Migration 011: Vector stores table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 011 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TRIGGER IF EXISTS update_vector_stores_timestamp ON vector_stores');
    await client.query('DROP TABLE IF EXISTS vector_stores CASCADE');

    await client.query('COMMIT');
    console.log('Migration 011: Vector stores table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 011 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
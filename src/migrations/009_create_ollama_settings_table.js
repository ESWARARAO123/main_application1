/**
 * Migration to create ollama_settings table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the ollama_settings table
    await client.query(`
      CREATE TABLE IF NOT EXISTS ollama_settings (
        id SERIAL PRIMARY KEY,
        host VARCHAR(255) DEFAULT 'localhost' NOT NULL,
        port INTEGER DEFAULT 11434 NOT NULL,
        default_model VARCHAR(100) DEFAULT 'llama2' NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create indexes for performance
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_ollama_settings_default_model ON ollama_settings(default_model)
    `);

    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_ollama_settings_updated_at ON ollama_settings(updated_at)
    `);

    await client.query('COMMIT');
    console.log('Migration 009: Ollama settings table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 009 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TABLE IF EXISTS ollama_settings CASCADE');

    await client.query('COMMIT');
    console.log('Migration 009: Ollama settings table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 009 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
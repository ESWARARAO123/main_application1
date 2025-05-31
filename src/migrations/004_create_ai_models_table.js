/**
 * Migration to create ai_models table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the ai_models table
    await client.query(`
      CREATE TABLE IF NOT EXISTS ai_models (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        model_id VARCHAR(100) NOT NULL,
        description TEXT,
        parameters JSONB,
        is_active BOOLEAN DEFAULT true,
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        ollama_model_id VARCHAR(100),
        updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now()
      )
    `);

    // Add comment on ollama_model_id column
    await client.query(`
      COMMENT ON COLUMN ai_models.ollama_model_id IS 'Model identifier in Ollama server'
    `);

    // Create indexes for performance
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_ai_models_ollama_model_id ON ai_models(ollama_model_id)
    `);

    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_ai_models_is_active ON ai_models(is_active)
    `);

    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_ai_models_name ON ai_models(name)
    `);

    await client.query('COMMIT');
    console.log('Migration 004: AI models table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 004 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TABLE IF EXISTS ai_models CASCADE');

    await client.query('COMMIT');
    console.log('Migration 004: AI models table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 004 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
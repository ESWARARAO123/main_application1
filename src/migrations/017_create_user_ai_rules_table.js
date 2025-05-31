/**
 * Migration to create user_ai_rules table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the user_ai_rules table
    await client.query(`
      CREATE TABLE IF NOT EXISTS user_ai_rules (
        id SERIAL PRIMARY KEY,
        user_id UUID NOT NULL,
        rule_content TEXT NOT NULL,
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
      )
    `);

    // Create index for faster lookups by user_id
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_user_ai_rules_user_id ON user_ai_rules(user_id)
    `);

    // Create function and trigger for updated_at timestamp
    await client.query(`
      CREATE OR REPLACE FUNCTION update_user_ai_rules_updated_at()
      RETURNS TRIGGER AS $$
      BEGIN
          NEW.updated_at = CURRENT_TIMESTAMP;
          RETURN NEW;
      END;
      $$ LANGUAGE plpgsql;
    `);

    await client.query(`
      DROP TRIGGER IF EXISTS trigger_update_user_ai_rules_updated_at ON user_ai_rules;
      CREATE TRIGGER trigger_update_user_ai_rules_updated_at
      BEFORE UPDATE ON user_ai_rules
      FOR EACH ROW
      EXECUTE FUNCTION update_user_ai_rules_updated_at();
    `);

    await client.query('COMMIT');
    console.log('Migration 017: User AI rules table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 017 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TRIGGER IF EXISTS trigger_update_user_ai_rules_updated_at ON user_ai_rules');
    await client.query('DROP FUNCTION IF EXISTS update_user_ai_rules_updated_at()');
    await client.query('DROP TABLE IF EXISTS user_ai_rules CASCADE');

    await client.query('COMMIT');
    console.log('Migration 017: User AI rules table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 017 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
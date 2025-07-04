const { pool } = require('../database');

/**
 * Migration to create database_details table (formerly in add_db_info.sql)
 */

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    await client.query(`
      CREATE TABLE IF NOT EXISTS database_details (
        id SERIAL PRIMARY KEY,
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        host VARCHAR(255) NOT NULL,
        database VARCHAR(255) NOT NULL,
        db_user VARCHAR(255) NOT NULL,
        db_password VARCHAR(255) NOT NULL,
        port INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      );
      CREATE UNIQUE INDEX IF NOT EXISTS idx_database_details_user_id ON database_details(user_id);
    `);
    await client.query('COMMIT');
    console.log('Migration 025: database_details table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 025 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    await client.query('DROP TABLE IF EXISTS database_details CASCADE;');
    await client.query('COMMIT');
    console.log('Migration 025: database_details table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 025 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
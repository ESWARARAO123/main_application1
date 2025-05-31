/**
 * Migration to create dashboard_metrics table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the dashboard_metrics table
    await client.query(`
      CREATE TABLE IF NOT EXISTS dashboard_metrics (
        id SERIAL PRIMARY KEY,
        metric_name VARCHAR(50) NOT NULL UNIQUE,
        metric_value JSONB NOT NULL,
        updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
      )
    `);

    await client.query('COMMIT');
    console.log('Migration 007: Dashboard metrics table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 007 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TABLE IF EXISTS dashboard_metrics CASCADE');

    await client.query('COMMIT');
    console.log('Migration 007: Dashboard metrics table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 007 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
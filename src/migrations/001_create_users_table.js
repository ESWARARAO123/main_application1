/**
 * Migration to create users table - base table for the application
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create the users table
    await client.query(`
      CREATE TABLE IF NOT EXISTS users (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        username VARCHAR(50) NOT NULL UNIQUE,
        password TEXT NOT NULL,
        role VARCHAR(20) DEFAULT 'user' NOT NULL,
        email VARCHAR(100),
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        name VARCHAR(100)
      )
    `);

    await client.query('COMMIT');
    console.log('Migration 001: Users table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 001 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TABLE IF EXISTS users CASCADE');

    await client.query('COMMIT');
    console.log('Migration 001: Users table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 001 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
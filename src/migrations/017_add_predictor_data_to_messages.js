const { pool } = require('../database');

async function up() {
  console.log('Adding predictor_data column to messages table...');
  
  try {
    // Add predictor_data column to store predictor-specific data as JSON
    await pool.query(`
      ALTER TABLE messages 
      ADD COLUMN IF NOT EXISTS predictor_data JSONB
    `);
    
    console.log('Successfully added predictor_data column to messages table');
  } catch (error) {
    console.error('Error adding predictor_data column:', error);
    throw error;
  }
}

async function down() {
  console.log('Removing predictor_data column from messages table...');
  
  try {
    await pool.query(`
      ALTER TABLE messages 
      DROP COLUMN IF EXISTS predictor_data
    `);
    
    console.log('Successfully removed predictor_data column from messages table');
  } catch (error) {
    console.error('Error removing predictor_data column:', error);
    throw error;
  }
}

module.exports = { up, down };
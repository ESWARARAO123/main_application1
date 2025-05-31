const fs = require('fs');
const path = require('path');
const { pool } = require('../database');

async function applyChatSessionsSchema() {
  try {
    console.log('Reading chat sessions schema file...');
    // Use a placeholder for the schema for now - we need to create this file
    const schemaFilePath = path.join(__dirname, 'sql/chat_sessions_schema.sql');
    
    // Check if file exists, if not, print error and exit
    if (!fs.existsSync(schemaFilePath)) {
      console.error(`Schema file not found: ${schemaFilePath}`);
      console.error('Please create the schema file or copy it to the correct location.');
      process.exit(1);
    }
    
    const schemaFile = fs.readFileSync(schemaFilePath, 'utf8');
    
    console.log('Applying chat sessions schema...');
    await pool.query(schemaFile);
    
    console.log('Chat sessions schema applied successfully!');
    
    // Close the pool
    await pool.end();
    console.log('Database connection closed');
  } catch (error) {
    console.error('Error applying chat sessions schema:', error);
    process.exit(1);
  }
}

// Run the function
applyChatSessionsSchema(); 
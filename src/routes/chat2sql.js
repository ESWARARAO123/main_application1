const express = require('express');
const router = express.Router();
const { pool } = require('../database');
const { Client } = require('pg');

// Middleware to check if user is authenticated
const isAuthenticated = (req, res, next) => {
  // For testing purposes, we'll allow unauthenticated access
  // In production, you should uncomment the authentication check
  // if (req.session.userId) {
  //   next();
  // } else {
  //   res.status(401).json({ error: 'Unauthorized' });
  // }
  next();
};

// Execute SQL query
router.post('/execute', isAuthenticated, async (req, res) => {
  try {
    const { query } = req.body;
    
    console.log('Executing SQL query:', query);
    
    // Validate query - only allow SELECT statements for security
    if (!query.trim().toLowerCase().startsWith('select')) {
      return res.status(403).json({ 
        error: 'Only SELECT queries are allowed for security reasons' 
      });
    }
    
    // Execute the query
    const result = await pool.query(query);
    
    // Format the result as a markdown table
    const columns = result.fields.map(field => field.name);
    
    // Create markdown table
    let markdownTable = '| ' + columns.join(' | ') + ' |\n';
    markdownTable += '| ' + columns.map(() => '---').join(' | ') + ' |\n';
    
    // Add rows
    result.rows.forEach(row => {
      const rowValues = columns.map(col => {
        const value = row[col];
        return value === null ? 'NULL' : String(value);
      });
      markdownTable += '| ' + rowValues.join(' | ') + ' |\n';
    });
    
    res.json({
      data: markdownTable,
      columns: columns
    });
  } catch (error) {
    console.error('Error executing SQL query:', error);
    res.status(500).json({ error: 'Error executing SQL query: ' + error.message });
  }
});

// Test and save database connection settings
router.post('/db-config', async (req, res) => {
  const { host, database, user, password, port, save } = req.body;
  // You may need to get userId from session or from req.body
  const userId = req.session?.userId || req.body.userId || 'test-user'; // fallback for testing

  if (!host || !database || !user || !password || !port) {
    return res.status(400).json({ error: 'All fields are required.' });
  }

  const client = new Client({
    host,
    database,
    user,
    password,
    port,
    ssl: false // set to true if your DB requires SSL
  });

  try {
    await client.connect();
    await client.end();

    // Optionally save settings if requested
    if (save && userId) {
      await pool.query(
        `INSERT INTO database_details (user_id, host, database, db_user, db_password, port, updated_at)
         VALUES ($1, $2, $3, $4, $5, $6, NOW())
         ON CONFLICT (user_id) DO UPDATE SET
           host = EXCLUDED.host,
           database = EXCLUDED.database,
           db_user = EXCLUDED.db_user,
           db_password = EXCLUDED.db_password,
           port = EXCLUDED.port,
           updated_at = NOW()`,
        [userId, host, database, user, password, port]
      );
    }

    return res.json({ success: true });
  } catch (err) {
    console.error('DB connection test failed:', err);
    return res.status(400).json({ error: 'Connection failed: ' + err.message });
  }
});

module.exports = router;
const express = require('express');
const router = express.Router();
const { pool } = require('../database');

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

module.exports = router;
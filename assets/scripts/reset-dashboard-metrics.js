/**
 * Script to reset the dashboard metrics
 * Run with: node scripts/reset-dashboard-metrics.js
 */

const path = require('path');
const fs = require('fs');
const ini = require('ini');
const { Pool } = require('pg');

// Load configuration
const configPath = path.join(__dirname, '../conf/config.ini');
const config = ini.parse(fs.readFileSync(configPath, 'utf-8'));

// Database connection settings
const dbConfig = {
  host: config.database.host,
  port: config.database.port,
  database: config.database.name,
  user: config.database.user,
  password: config.database.password
};

async function resetDashboardMetrics() {
  const pool = new Pool(dbConfig);
  
  try {
    console.log('Connecting to database...');
    
    // Reset the dashboard_metrics table
    console.log('Resetting dashboard metrics...');
    await pool.query(`
      UPDATE dashboard_metrics 
      SET 
        total_messages = 0,
        recent_messages = 0,
        total_documents = 0,
        avg_response_time = 0
      WHERE id = 1
    `);
    
    console.log('Dashboard metrics reset successfully!');
  } catch (error) {
    console.error('Error resetting dashboard metrics:', error);
  } finally {
    await pool.end();
  }
}

// Run the reset function
resetDashboardMetrics();

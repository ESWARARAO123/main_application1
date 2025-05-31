#!/usr/bin/env node

/**
 * Database Validation Script
 * 
 * This script validates that all required database tables, functions, and triggers exist.
 * Useful for verifying successful installation or troubleshooting.
 * 
 * Usage:
 *   node src/scripts/validate-database.js [--config=path/to/config.ini]
 *   npm run validate:database
 */

const path = require('path');
const fs = require('fs');
const { Pool } = require('pg');
const ini = require('ini');

// ANSI color codes for better output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function colorLog(color, message) {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

const REQUIRED_TABLES = [
  'users',
  'sessions', 
  'chat_sessions',
  'ai_models',
  'messages',
  'dashboard_metrics',
  'mcp_connections',
  'ollama_settings',
  'schema_migrations',
  'vector_stores',
  'document_collections',
  'document_chunks',
  'rag_settings',
  'documents',
  'document_count',
  'message_stats',
  'user_ssh_configurations',
  'user_mcp_server_configurations',
  'user_ai_rules'
];

const REQUIRED_FUNCTIONS = [
  'update_timestamp_column',
  'update_ollama_settings_timestamp',
  'update_chat_session_timestamp',
  'trigger_update_dashboard_metrics',
  'update_dashboard_metrics',
  'update_document_count',
  'update_message_stats'
];

async function main() {
  try {
    colorLog('cyan', '\n🔍 Starting Database Validation...\n');

    // Load configuration
    const configPath = process.argv.find(arg => arg.startsWith('--config='))
      ? process.argv.find(arg => arg.startsWith('--config=')).split('=')[1]
      : './conf/config.ini';

    const fullConfigPath = path.resolve(configPath);
    
    if (!fs.existsSync(fullConfigPath)) {
      colorLog('red', `❌ Configuration file not found: ${fullConfigPath}`);
      process.exit(1);
    }

    const config = ini.parse(fs.readFileSync(fullConfigPath, 'utf-8'));

    // PostgreSQL connection configuration
    const pgConfig = {
      host: config.database['database-host'],
      port: config.database['database-port'],
      user: config.database['database-user'],
      password: config.database['database-password'],
      database: config.database['database-name'],
      max: 5,
      ssl: config.database.ssl === 'true'
    };

    colorLog('blue', `🔌 Connecting to PostgreSQL at ${pgConfig.host}:${pgConfig.port}`);
    
    const pool = new Pool(pgConfig);

    // Test database connection
    try {
      const client = await pool.connect();
      colorLog('green', '✅ Database connection successful');
      client.release();
    } catch (error) {
      colorLog('red', `❌ Database connection failed: ${error.message}`);
      process.exit(1);
    }

    let allValid = true;

    // Validate tables
    colorLog('blue', '\n📋 Checking required tables...');
    const tableResults = await validateTables(pool);
    if (!tableResults.allExist) allValid = false;

    // Validate functions
    colorLog('blue', '\n🔧 Checking required functions...');
    const functionResults = await validateFunctions(pool);
    if (!functionResults.allExist) allValid = false;

    // Validate triggers
    colorLog('blue', '\n⚡ Checking triggers...');
    const triggerResults = await validateTriggers(pool);
    if (!triggerResults.allExist) allValid = false;

    // Check admin user
    colorLog('blue', '\n👤 Checking admin user...');
    const adminResults = await validateAdminUser(pool, config);
    if (!adminResults.exists) allValid = false;

    // Check migrations
    colorLog('blue', '\n📦 Checking migration status...');
    const migrationResults = await validateMigrations(pool);

    // Final result
    colorLog('blue', '\n' + '='.repeat(50));
    if (allValid) {
      colorLog('green', '🎉 Database validation PASSED! All required components exist.');
    } else {
      colorLog('red', '❌ Database validation FAILED! Some components are missing.');
      colorLog('yellow', '💡 Run `npm run install:database` to fix missing components.');
    }

    // Summary
    colorLog('cyan', '\n📊 Validation Summary:');
    colorLog(tableResults.allExist ? 'green' : 'red', `  Tables: ${tableResults.existing}/${tableResults.total}`);
    colorLog(functionResults.allExist ? 'green' : 'red', `  Functions: ${functionResults.existing}/${functionResults.total}`);
    colorLog(triggerResults.allExist ? 'green' : 'red', `  Triggers: ${triggerResults.existing}/${triggerResults.total}`);
    colorLog(adminResults.exists ? 'green' : 'red', `  Admin User: ${adminResults.exists ? 'EXISTS' : 'MISSING'}`);
    colorLog('blue', `  Applied Migrations: ${migrationResults.count}`);

    await pool.end();
    process.exit(allValid ? 0 : 1);
    
  } catch (error) {
    colorLog('red', `\n❌ Validation failed: ${error.message}`);
    console.error(error);
    process.exit(1);
  }
}

async function validateTables(pool) {
  const client = await pool.connect();
  let existing = 0;
  
  try {
    for (const table of REQUIRED_TABLES) {
      const result = await client.query(`
        SELECT EXISTS (
          SELECT FROM information_schema.tables 
          WHERE table_schema = 'public' 
          AND table_name = $1
        )
      `, [table]);
      
      const exists = result.rows[0].exists;
      if (exists) {
        colorLog('green', `  ✅ ${table}`);
        existing++;
      } else {
        colorLog('red', `  ❌ ${table} (MISSING)`);
      }
    }
  } finally {
    client.release();
  }

  return {
    allExist: existing === REQUIRED_TABLES.length,
    existing,
    total: REQUIRED_TABLES.length
  };
}

async function validateFunctions(pool) {
  const client = await pool.connect();
  let existing = 0;
  
  try {
    for (const func of REQUIRED_FUNCTIONS) {
      const result = await client.query(`
        SELECT EXISTS (
          SELECT FROM pg_proc 
          WHERE proname = $1
        )
      `, [func]);
      
      const exists = result.rows[0].exists;
      if (exists) {
        colorLog('green', `  ✅ ${func}()`);
        existing++;
      } else {
        colorLog('red', `  ❌ ${func}() (MISSING)`);
      }
    }
  } finally {
    client.release();
  }

  return {
    allExist: existing === REQUIRED_FUNCTIONS.length,
    existing,
    total: REQUIRED_FUNCTIONS.length
  };
}

async function validateTriggers(pool) {
  const client = await pool.connect();
  let existing = 0;
  const requiredTriggers = [
    'update_ollama_settings_timestamp',
    'update_session_timestamp', 
    'users_update_metrics',
    'messages_update_metrics',
    'documents_count_trigger',
    'messages_stats_trigger'
  ];
  
  try {
    for (const trigger of requiredTriggers) {
      const result = await client.query(`
        SELECT EXISTS (
          SELECT FROM pg_trigger 
          WHERE tgname = $1
        )
      `, [trigger]);
      
      const exists = result.rows[0].exists;
      if (exists) {
        colorLog('green', `  ✅ ${trigger}`);
        existing++;
      } else {
        colorLog('red', `  ❌ ${trigger} (MISSING)`);
      }
    }
  } finally {
    client.release();
  }

  return {
    allExist: existing === requiredTriggers.length,
    existing,
    total: requiredTriggers.length
  };
}

async function validateAdminUser(pool, config) {
  const client = await pool.connect();
  
  try {
    const result = await client.query(
      "SELECT EXISTS (SELECT FROM users WHERE username = $1 AND role = 'admin')",
      [config.admin.default_username]
    );
    
    const exists = result.rows[0].exists;
    if (exists) {
      colorLog('green', `  ✅ Admin user '${config.admin.default_username}' exists`);
    } else {
      colorLog('red', `  ❌ Admin user '${config.admin.default_username}' missing`);
    }

    return { exists };
  } finally {
    client.release();
  }
}

async function validateMigrations(pool) {
  const client = await pool.connect();
  
  try {
    // Check if schema_migrations table exists
    const tableExists = await client.query(`
      SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'schema_migrations'
      )
    `);

    if (!tableExists.rows[0].exists) {
      colorLog('red', '  ❌ schema_migrations table missing');
      return { count: 0 };
    }

    const result = await client.query('SELECT COUNT(*) FROM schema_migrations');
    const count = parseInt(result.rows[0].count);
    
    if (count > 0) {
      colorLog('green', `  ✅ ${count} migrations applied`);
      
      // Show recent migrations
      const recent = await client.query(`
        SELECT name, applied_at 
        FROM schema_migrations 
        ORDER BY applied_at DESC 
        LIMIT 3
      `);
      
      recent.rows.forEach(row => {
        colorLog('blue', `    - ${row.name} (${row.applied_at.toISOString().split('T')[0]})`);
      });
    } else {
      colorLog('yellow', '  ⚠️  No migrations applied yet');
    }

    return { count };
  } finally {
    client.release();
  }
}

// Run the main function
if (require.main === module) {
  main().catch(error => {
    colorLog('red', `\n💥 Fatal error during validation: ${error.message}`);
    console.error(error);
    process.exit(1);
  });
}

module.exports = { main }; 
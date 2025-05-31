#!/usr/bin/env node

/**
 * Database Installation Script
 * 
 * This script sets up the complete database structure for the application.
 * It can be run during application installation to create all necessary tables,
 * functions, triggers, and initial data.
 * 
 * Usage:
 *   node src/scripts/install-database.js [--config=path/to/config.ini]
 *   npm run install:database
 */

const path = require('path');
const fs = require('fs');
const { Pool } = require('pg');
const ini = require('ini');
const bcrypt = require('bcrypt');

// ANSI color codes for better output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

function colorLog(color, message) {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

async function main() {
  try {
    colorLog('cyan', '\n🚀 Starting Database Installation Process...\n');

    // Load configuration
    const configPath = process.argv.find(arg => arg.startsWith('--config='))
      ? process.argv.find(arg => arg.startsWith('--config=')).split('=')[1]
      : './conf/config.ini';

    const fullConfigPath = path.resolve(configPath);
    
    if (!fs.existsSync(fullConfigPath)) {
      colorLog('red', `❌ Configuration file not found: ${fullConfigPath}`);
      colorLog('yellow', 'Please ensure the config.ini file exists with database configuration.');
      process.exit(1);
    }

    colorLog('blue', `📖 Loading configuration from: ${fullConfigPath}`);
    const config = ini.parse(fs.readFileSync(fullConfigPath, 'utf-8'));

    // Validate database configuration
    if (!config.database) {
      colorLog('red', '❌ Database configuration section not found in config file');
      process.exit(1);
    }

    const requiredFields = ['database-host', 'database-port', 'database-user', 'database-password', 'database-name'];
    for (const field of requiredFields) {
      if (!config.database[field]) {
        colorLog('red', `❌ Missing required database configuration: ${field}`);
        process.exit(1);
      }
    }

    // PostgreSQL connection configuration
    const pgConfig = {
      host: config.database['database-host'],
      port: config.database['database-port'],
      user: config.database['database-user'],
      password: config.database['database-password'],
      database: config.database['database-name'],
      max: 10,
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
      colorLog('yellow', 'Please check your database configuration and ensure PostgreSQL is running.');
      process.exit(1);
    }

    // Run migrations
    colorLog('magenta', '\n📋 Running database migrations...\n');
    
    // Create schema_migrations table first
    await createSchemaMigrationsTable(pool);
    
    // Get and run all migrations
    await runAllMigrations(pool);

    // Create admin user
    await createAdminUser(pool, config);

    // Initialize dashboard metrics
    await initializeDashboardMetrics(pool);

    colorLog('green', '\n🎉 Database installation completed successfully!\n');
    
    // Display summary
    colorLog('cyan', '📊 Installation Summary:');
    colorLog('blue', '  ✅ All database tables created');
    colorLog('blue', '  ✅ Functions and triggers installed');
    colorLog('blue', '  ✅ Indexes created for performance');
    colorLog('blue', '  ✅ Admin user configured');
    colorLog('blue', '  ✅ Dashboard metrics initialized');
    
    await pool.end();
    
  } catch (error) {
    colorLog('red', `\n❌ Installation failed: ${error.message}`);
    console.error(error);
    process.exit(1);
  }
}

async function createSchemaMigrationsTable(pool) {
  const client = await pool.connect();
  try {
    await client.query(`
      CREATE TABLE IF NOT EXISTS schema_migrations (
        id SERIAL PRIMARY KEY,
        version VARCHAR(100) NOT NULL UNIQUE,
        applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        description TEXT,
        name VARCHAR(255)
      )
    `);
    
    // Add comment
    await client.query(`
      COMMENT ON TABLE schema_migrations IS 'Tracks applied database migrations'
    `);

    colorLog('green', '✅ Schema migrations table ready');
  } catch (error) {
    colorLog('red', `❌ Failed to create schema_migrations table: ${error.message}`);
    throw error;
  } finally {
    client.release();
  }
}

async function runAllMigrations(pool) {
  // Get all migration files
  const migrationsDir = path.join(__dirname, '..', 'migrations');
  
  if (!fs.existsSync(migrationsDir)) {
    colorLog('yellow', '⚠️  No migrations directory found, skipping migrations');
    return;
  }

  const migrationFiles = fs.readdirSync(migrationsDir)
    .filter(file => file.endsWith('.js'))
    .sort(); // Ensure migrations run in order

  if (migrationFiles.length === 0) {
    colorLog('yellow', '⚠️  No migration files found');
    return;
  }

  // Get applied migrations
  const client = await pool.connect();
  let appliedMigrations = [];
  
  try {
    const result = await client.query('SELECT name FROM schema_migrations WHERE name IS NOT NULL');
    appliedMigrations = result.rows.map(row => row.name);
  } catch (error) {
    colorLog('yellow', '⚠️  Could not fetch applied migrations, assuming none applied');
  } finally {
    client.release();
  }

  // Run each migration
  for (const file of migrationFiles) {
    if (!appliedMigrations.includes(file)) {
      colorLog('blue', `  🔄 Running migration: ${file}`);
      
      try {
        const migrationPath = path.join(migrationsDir, file);
        const migration = require(migrationPath);

        await migration.up();

        // Record migration as applied
        const version = file.match(/^(\d+)_/) ? file.match(/^(\d+)_/)[1] : file;
        const description = file.replace(/\.js$/, '').replace(/^\d+_/, '').replace(/_/g, ' ');
        
        const client = await pool.connect();
        try {
          await client.query(
            'INSERT INTO schema_migrations (name, version, description) VALUES ($1, $2, $3)',
            [file, version, description]
          );
        } finally {
          client.release();
        }
        
        colorLog('green', `  ✅ Migration ${file} completed`);
      } catch (error) {
        colorLog('red', `  ❌ Migration ${file} failed: ${error.message}`);
        throw error;
      }
    } else {
      colorLog('yellow', `  ⏭️  Migration ${file} already applied, skipping`);
    }
  }
}

async function createAdminUser(pool, config) {
  colorLog('blue', '\n👤 Setting up admin user...');
  
  const client = await pool.connect();
  try {
    // Check if admin user exists
    const userQuery = await client.query(
      "SELECT * FROM users WHERE username = $1", 
      [config.admin.default_username]
    );

    if (userQuery.rows.length === 0) {
      const hashedPassword = bcrypt.hashSync(config.admin.default_password, 10);

      await client.query(
        "INSERT INTO users (username, password, email, role, name) VALUES ($1, $2, $3, $4, $5)",
        [
          config.admin.default_username, 
          hashedPassword, 
          config.admin.default_email || 'admin@localhost', 
          'admin', 
          'Administrator'
        ]
      );
      colorLog('green', `✅ Admin user '${config.admin.default_username}' created`);
    } else {
      colorLog('yellow', `⚠️  Admin user '${config.admin.default_username}' already exists`);
    }
  } catch (error) {
    colorLog('red', `❌ Failed to create admin user: ${error.message}`);
    throw error;
  } finally {
    client.release();
  }
}

async function initializeDashboardMetrics(pool) {
  colorLog('blue', '\n📊 Initializing dashboard metrics...');
  
  const client = await pool.connect();
  try {
    // Initialize basic metrics
    const defaultMetrics = [
      {
        name: 'user_stats',
        value: {
          totalUsers: 1,
          adminUsers: 1,
          regularUsers: 0,
          recentUsers: 1
        }
      },
      {
        name: 'message_stats', 
        value: {
          totalMessages: 0,
          recentMessages: 0,
          avgResponseTime: 0,
          totalDocuments: 0
        }
      },
      {
        name: 'license_usage',
        value: {
          totalLicenses: 25,
          activeLicenses: 1,
          expirationDate: '2024-12-31',
          daysRemaining: 245
        }
      }
    ];

    for (const metric of defaultMetrics) {
      await client.query(
        `INSERT INTO dashboard_metrics (metric_name, metric_value) 
         VALUES ($1, $2) 
         ON CONFLICT (metric_name) 
         DO UPDATE SET metric_value = EXCLUDED.metric_value, updated_at = NOW()`,
        [metric.name, JSON.stringify(metric.value)]
      );
    }

    colorLog('green', '✅ Dashboard metrics initialized');
  } catch (error) {
    colorLog('red', `❌ Failed to initialize dashboard metrics: ${error.message}`);
    throw error;
  } finally {
    client.release();
  }
}

// Handle script termination
process.on('SIGINT', () => {
  colorLog('yellow', '\n⚠️  Installation interrupted by user');
  process.exit(1);
});

process.on('SIGTERM', () => {
  colorLog('yellow', '\n⚠️  Installation terminated');
  process.exit(1);
});

// Run the main function
if (require.main === module) {
  main().catch(error => {
    colorLog('red', `\n💥 Fatal error during installation: ${error.message}`);
    console.error(error);
    process.exit(1);
  });
}

module.exports = { main }; 
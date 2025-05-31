#!/usr/bin/env node

/**
 * Combined Database Setup Script
 * 
 * This script performs a complete database setup:
 * 1. Runs migrations
 * 2. Creates admin user and initializes data
 * 3. Validates the installation
 * 
 * Usage:
 *   node src/scripts/setup-database.js [--config=path/to/config.ini]
 *   npm run setup:database
 */

const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

// ANSI color codes for better output
const colors = {
  reset: '\x1b[0m',
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

function runScript(scriptPath, args = []) {
  return new Promise((resolve, reject) => {
    colorLog('blue', `\n🔄 Running: ${scriptPath} ${args.join(' ')}`);
    
    const child = spawn('node', [scriptPath, ...args], {
      stdio: 'inherit',
      cwd: process.cwd()
    });

    child.on('close', (code) => {
      if (code === 0) {
        colorLog('green', `✅ Completed: ${scriptPath}`);
        resolve();
      } else {
        colorLog('red', `❌ Failed: ${scriptPath} (exit code: ${code})`);
        reject(new Error(`Script failed with exit code: ${code}`));
      }
    });

    child.on('error', (error) => {
      colorLog('red', `❌ Error running ${scriptPath}: ${error.message}`);
      reject(error);
    });
  });
}

async function main() {
  try {
    colorLog('cyan', '\n🚀 Starting Complete Database Setup Process...\n');
    colorLog('cyan', '═══════════════════════════════════════════════════════');
    
    // Get config argument if provided
    const configArg = process.argv.find(arg => arg.startsWith('--config='));
    const args = configArg ? [configArg] : [];
    
    // Step 1: Run migrations
    colorLog('magenta', '\n📋 STEP 1: Running Database Migrations');
    colorLog('cyan', '─────────────────────────────────────────────');
    await runScript('src/scripts/migrate.js', args);
    
    // Step 2: Install admin user and initialize data
    colorLog('magenta', '\n👤 STEP 2: Setting Up Admin User & Initial Data');
    colorLog('cyan', '─────────────────────────────────────────────');
    await runScript('src/scripts/install-database.js', args);
    
    // Step 3: Validate installation
    colorLog('magenta', '\n🔍 STEP 3: Validating Database Installation');
    colorLog('cyan', '─────────────────────────────────────────────');
    await runScript('src/scripts/validate-database.js', args);
    
    // Success summary
    colorLog('cyan', '\n═══════════════════════════════════════════════════════');
    colorLog('green', '🎉 COMPLETE DATABASE SETUP SUCCESSFUL! 🎉');
    colorLog('cyan', '═══════════════════════════════════════════════════════');
    colorLog('green', '\n✅ Database migrations applied');
    colorLog('green', '✅ Admin user created and configured');
    colorLog('green', '✅ Dashboard metrics initialized');
    colorLog('green', '✅ Database validation passed');
    colorLog('cyan', '\n🚀 Your database is ready to use!');
    colorLog('yellow', '\n💡 You can now start your application with: npm start');
    
  } catch (error) {
    colorLog('red', '\n❌ Database setup failed!');
    colorLog('red', `Error: ${error.message}`);
    colorLog('yellow', '\n🛠️  Troubleshooting:');
    colorLog('yellow', '1. Check your database connection settings in conf/config.ini');
    colorLog('yellow', '2. Ensure PostgreSQL server is running');
    colorLog('yellow', '3. Verify database permissions');
    colorLog('yellow', '4. Check the logs above for specific error details');
    process.exit(1);
  }
}

// Handle script termination
process.on('SIGINT', () => {
  colorLog('yellow', '\n⚠️  Database setup interrupted by user');
  process.exit(1);
});

process.on('SIGTERM', () => {
  colorLog('yellow', '\n⚠️  Database setup terminated');
  process.exit(1);
});

// Run the main function
if (require.main === module) {
  main().catch(error => {
    colorLog('red', `\n💥 Fatal error during database setup: ${error.message}`);
    console.error(error);
    process.exit(1);
  });
}

module.exports = { main }; 
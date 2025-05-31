# Database Installation Guide

This guide explains how to set up the database for the Platform Dashboard application on a new device or environment.

## 🚀 Quick Installation

For new installations, simply run:

```bash
# Full installation with admin user setup
npm run install:database

# Or just run migrations
npm run db:migrate

# Verify installation
npm run validate:database

# main script which can handle all three
npm run setup:database
```

This will automatically:
- Create all required database tables
- Set up functions, triggers, and indexes
- Create the admin user (with install:database)
- Initialize dashboard metrics (with install:database)

## 📋 Prerequisites

Before running the database installation, ensure you have:

1. **PostgreSQL Database Server** running and accessible
2. **Configuration File** (`conf/config.ini`) with correct database settings
3. **Node.js Dependencies** installed (`npm install`)

### Database Configuration

Your `conf/config.ini` should contain the database section:

```ini
[database]
database-type = postgres
database-host = localhost
database-port = 5432
database-user = postgres
database-password = your_password
database-name = your_database_name
max_connections = 100
log_queries = false
ssl = false
```

### Admin User Configuration

The admin user will be created based on your config:

```ini
[admin]
default_username = admin
default_password = admin
default_email = admin@localhost
```

## 🔧 Installation Methods

### Method 1: Automated Installation (Recommended)

```bash
# Complete setup with admin user and initial data
npm run install:database

# Using custom config location
node src/scripts/install-database.js --config=/path/to/your/config.ini
```

### Method 2: Migrations Only

```bash
# Run just the migrations (no admin user or initial data)
npm run db:migrate

# Using custom config location
node src/scripts/migrate.js --config=/path/to/your/config.ini
```

### Method 3: Manual Migration Run

If you prefer to run migrations manually through the application:

```bash
# Start the application - it will run migrations automatically
npm start
```

The application's `database.js` file includes automatic migration running on startup.

## 📁 Migration System

### Migration Files

All migration files are located in `src/migrations/` and follow this naming convention:
- `001_create_users_table.js`
- `002_create_sessions_table.js`
- `003_create_chat_sessions_table.js`
- etc.

### Migration Structure

Each migration file contains:

```javascript
const { pool } = require('../database');

async function up() {
  // Migration logic to apply changes
}

async function down() {
  // Migration logic to rollback changes
}

module.exports = { up, down };
```

### Complete Migration List (18 Tables)

| Order | File | Description |
|-------|------|-------------|
| 001 | `create_users_table.js` | Creates the base users table |
| 002 | `create_sessions_table.js` | Creates user sessions table |
| 003 | `create_chat_sessions_table.js` | Creates chat sessions table |
| 004 | `create_ai_models_table.js` | Creates AI models configuration table |
| 005 | `add_documents_table.js` | Creates documents table with full structure |
| 006 | `create_messages_table.js` | Creates messages table |
| 007 | `create_dashboard_metrics_table.js` | Creates dashboard metrics table |
| 008 | `create_mcp_connections_table.js` | Creates MCP connections table |
| 009 | `create_ollama_settings_table.js` | Creates Ollama settings table |
| 010 | `create_database_functions_and_triggers.js` | Creates PostgreSQL functions and triggers |
| 011 | `create_vector_stores_table.js` | Creates vector stores table |
| 012 | `create_document_collections_table.js` | Creates document collections table |
| 013 | `create_document_chunks_table.js` | Creates document chunks table for RAG |
| 014 | `create_rag_settings_table.js` | Creates RAG configuration table |
| 015 | `create_user_ssh_configurations_table.js` | Creates SSH connection settings |
| 016 | `create_user_mcp_server_configurations_table.js` | Creates user MCP settings |
| 017 | `create_user_ai_rules_table.js` | Creates user-specific AI rules |
| 018 | `create_document_functions.js` | Creates document processing functions |

## 🛠️ Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```
❌ Database connection failed: Connection refused
```
**Solution:** 
- Ensure PostgreSQL is running
- Check host, port, username, and password in config
- Verify the database exists

#### 2. Permission Denied
```
❌ Failed to create table: permission denied
```
**Solution:**
- Ensure the database user has CREATE privileges
- Grant necessary permissions: `GRANT ALL PRIVILEGES ON DATABASE your_db TO your_user;`

#### 3. Configuration File Not Found
```
❌ Configuration file not found: ./conf/config.ini
```
**Solution:**
- Create the config file from the template
- Specify the correct path with `--config=path/to/config.ini`

#### 4. Migration Already Applied
```
⚠️ Migration 001_create_users_table.js already applied, skipping
```
**Note:** This is normal - migrations track what's already been applied.

### Checking Installation Status

To verify your installation was successful:

```sql
-- Check if all tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check migration status
SELECT * FROM schema_migrations ORDER BY applied_at;

-- Verify admin user
SELECT username, role, created_at FROM users WHERE role = 'admin';

-- Count total tables (should be 18)
SELECT COUNT(*) as table_count 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_type = 'BASE TABLE';
```

## 🔄 Re-running Installation

The installation script is idempotent - you can run it multiple times safely:
- Existing tables won't be recreated
- Applied migrations will be skipped
- Existing admin user won't be duplicated

## 📊 What Gets Created (18 Tables)

### Core Tables
- `users` - User accounts and authentication
- `sessions` - User login sessions
- `schema_migrations` - Migration tracking

### Chat & AI Tables
- `chat_sessions` - Chat conversation sessions
- `messages` - Chat messages and responses
- `ai_models` - AI model configurations
- `user_ai_rules` - User-specific AI rules

### Document & RAG Tables
- `documents` - Uploaded document metadata
- `document_collections` - Document organization
- `document_chunks` - Document text chunks for RAG
- `vector_stores` - Vector storage configurations
- `rag_settings` - RAG configuration settings

### System Tables
- `dashboard_metrics` - Application metrics
- `ollama_settings` - Ollama configuration

### Connection Tables
- `mcp_connections` - MCP server connections
- `user_ssh_configurations` - SSH connection settings
- `user_mcp_server_configurations` - User MCP settings

### Functions & Triggers
- Automatic timestamp updates
- Dashboard metrics computation
- Chat session timestamp tracking
- Document processing functions (RAG system)
- User AI rules management

### Indexes
- Performance indexes on frequently queried columns
- Foreign key indexes
- Unique constraints

## 🚀 Available Commands

```bash
# Complete database setup (recommended for new installations)
npm run install:database

# Run just migrations (good for updates)
npm run db:migrate

# Validate your database setup
npm run validate:database

# With custom config
npm run db:migrate -- --config=/path/to/config.ini
```

## 🔐 Security Notes

1. **Change Default Passwords:** Update the admin password after installation
2. **Database Permissions:** Use a dedicated database user with minimal required permissions
3. **SSL Connections:** Enable SSL in production environments
4. **Network Security:** Restrict database access to authorized hosts only

## 📝 Logs and Monitoring

The installation script provides colored output:
- 🟢 Green: Successful operations
- 🟡 Yellow: Warnings or skipped items
- 🔴 Red: Errors
- 🔵 Blue: Information
- 🟣 Magenta: Process steps

All operations are wrapped in database transactions for safety.

## 🆘 Support

If you encounter issues not covered in this guide:

1. Check the application logs
2. Verify PostgreSQL server status
3. Ensure all Node.js dependencies are installed
4. Check database permissions and connectivity
5. Review the migration files for any environment-specific requirements

For additional help, check the project documentation or contact the development team. 
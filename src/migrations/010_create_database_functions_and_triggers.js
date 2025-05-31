/**
 * Migration to create database functions and triggers
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create update_timestamp_column function
    await client.query(`
      CREATE OR REPLACE FUNCTION update_timestamp_column()
      RETURNS TRIGGER AS $$
      BEGIN
          NEW.updated_at = CURRENT_TIMESTAMP;
          RETURN NEW;
      END;
      $$ LANGUAGE plpgsql;
    `);

    // Create update_ollama_settings_timestamp function
    await client.query(`
      CREATE OR REPLACE FUNCTION update_ollama_settings_timestamp()
      RETURNS TRIGGER AS $$
      BEGIN
          NEW.updated_at = CURRENT_TIMESTAMP;
          RETURN NEW;
      END;
      $$ LANGUAGE plpgsql;
    `);

    // Create update_chat_session_timestamp function
    await client.query(`
      CREATE OR REPLACE FUNCTION update_chat_session_timestamp()
      RETURNS TRIGGER AS $$
      BEGIN
        UPDATE chat_sessions
        SET last_message_timestamp = NEW.timestamp
        WHERE id = NEW.session_id;
        RETURN NEW;
      END;
      $$ LANGUAGE plpgsql;
    `);

    // Create trigger_update_dashboard_metrics function
    await client.query(`
      CREATE OR REPLACE FUNCTION trigger_update_dashboard_metrics()
      RETURNS TRIGGER AS $$
      BEGIN
        PERFORM update_dashboard_metrics();
        RETURN NEW;
      END;
      $$ LANGUAGE plpgsql;
    `);

    // Create update_dashboard_metrics function
    await client.query(`
      CREATE OR REPLACE FUNCTION update_dashboard_metrics()
      RETURNS void AS $$
      DECLARE
        user_stats jsonb;
        message_stats jsonb;
        document_count integer;
      BEGIN
        -- Get user statistics
        SELECT jsonb_build_object(
          'totalUsers', COUNT(*),
          'adminUsers', COUNT(CASE WHEN role = 'admin' THEN 1 END),
          'regularUsers', COUNT(CASE WHEN role = 'user' THEN 1 END),
          'recentUsers', COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END)
        ) INTO user_stats FROM users;

        -- Get message statistics
        SELECT jsonb_build_object(
          'totalMessages', COUNT(*),
          'recentMessages', COUNT(CASE WHEN timestamp > NOW() - INTERVAL '7 days' THEN 1 END),
          'avgResponseTime', 0
        ) INTO message_stats FROM messages;

        -- Get document count (safe check for documents table)
        SELECT COALESCE(COUNT(*), 0) INTO document_count 
        FROM information_schema.tables 
        WHERE table_name = 'documents';
        
        IF document_count > 0 THEN
          SELECT COUNT(*) INTO document_count FROM documents;
        ELSE
          document_count := 0;
        END IF;

        -- Update message_stats with document count
        message_stats := message_stats || jsonb_build_object('totalDocuments', document_count);

        -- Ensure dashboard_metrics table exists and update
        INSERT INTO dashboard_metrics (metric_name, metric_value) 
        VALUES ('user_stats', user_stats)
        ON CONFLICT (metric_name) 
        DO UPDATE SET metric_value = EXCLUDED.metric_value, updated_at = NOW();

        INSERT INTO dashboard_metrics (metric_name, metric_value) 
        VALUES ('message_stats', message_stats)
        ON CONFLICT (metric_name) 
        DO UPDATE SET metric_value = EXCLUDED.metric_value, updated_at = NOW();

        -- Update license usage
        INSERT INTO dashboard_metrics (metric_name, metric_value) 
        VALUES ('license_usage', jsonb_build_object(
          'totalLicenses', 25,
          'activeLicenses', LEAST((user_stats->>'totalUsers')::integer, 12),
          'expirationDate', '2024-12-31',
          'daysRemaining', 245
        ))
        ON CONFLICT (metric_name) 
        DO UPDATE SET metric_value = EXCLUDED.metric_value, updated_at = NOW();
      END;
      $$ LANGUAGE plpgsql;
    `);

    // Create trigger on ollama_settings for timestamp updates
    await client.query(`
      DROP TRIGGER IF EXISTS update_ollama_settings_timestamp ON ollama_settings;
      CREATE TRIGGER update_ollama_settings_timestamp 
      BEFORE UPDATE ON ollama_settings 
      FOR EACH ROW 
      EXECUTE FUNCTION update_ollama_settings_timestamp();
    `);

    // Create trigger on messages for session timestamp updates
    await client.query(`
      DROP TRIGGER IF EXISTS update_session_timestamp ON messages;
      CREATE TRIGGER update_session_timestamp 
      AFTER INSERT ON messages 
      FOR EACH ROW 
      EXECUTE FUNCTION update_chat_session_timestamp();
    `);

    // Create triggers for dashboard metrics updates
    await client.query(`
      DROP TRIGGER IF EXISTS users_update_metrics ON users;
      CREATE TRIGGER users_update_metrics 
      AFTER INSERT OR DELETE OR UPDATE ON users 
      FOR EACH STATEMENT 
      EXECUTE FUNCTION trigger_update_dashboard_metrics();
    `);

    await client.query(`
      DROP TRIGGER IF EXISTS messages_update_metrics ON messages;
      CREATE TRIGGER messages_update_metrics 
      AFTER INSERT OR DELETE OR UPDATE ON messages 
      FOR EACH STATEMENT 
      EXECUTE FUNCTION trigger_update_dashboard_metrics();
    `);

    await client.query('COMMIT');
    console.log('Migration 010: Database functions and triggers created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 010 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Drop triggers first
    await client.query('DROP TRIGGER IF EXISTS update_ollama_settings_timestamp ON ollama_settings');
    await client.query('DROP TRIGGER IF EXISTS update_session_timestamp ON messages');
    await client.query('DROP TRIGGER IF EXISTS users_update_metrics ON users');
    await client.query('DROP TRIGGER IF EXISTS messages_update_metrics ON messages');

    // Drop functions
    await client.query('DROP FUNCTION IF EXISTS update_timestamp_column()');
    await client.query('DROP FUNCTION IF EXISTS update_ollama_settings_timestamp()');
    await client.query('DROP FUNCTION IF EXISTS update_chat_session_timestamp()');
    await client.query('DROP FUNCTION IF EXISTS trigger_update_dashboard_metrics()');
    await client.query('DROP FUNCTION IF EXISTS update_dashboard_metrics()');

    await client.query('COMMIT');
    console.log('Migration 010: Database functions and triggers dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 010 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
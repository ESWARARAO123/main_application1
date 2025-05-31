/**
 * Migration to create message_stats table
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Drop the existing message_stats table if it exists (it has wrong structure)
    await client.query(`DROP TABLE IF EXISTS message_stats CASCADE`);

    // Create the message_stats table with correct structure
    await client.query(`
      CREATE TABLE message_stats (
        id SERIAL PRIMARY KEY,
        user_id UUID NOT NULL,
        session_id UUID,
        total_messages INTEGER DEFAULT 0,
        total_responses INTEGER DEFAULT 0,
        last_message_at TIMESTAMP WITH TIME ZONE,
        last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
      )
    `);

    // Create indexes for performance
    await client.query(`
      CREATE INDEX idx_message_stats_user_id ON message_stats(user_id)
    `);

    await client.query(`
      CREATE INDEX idx_message_stats_session_id ON message_stats(session_id)
    `);

    await client.query(`
      CREATE UNIQUE INDEX idx_message_stats_user_session ON message_stats(user_id, session_id)
    `);

    // Create trigger to update message stats when messages are added
    await client.query(`
      CREATE OR REPLACE FUNCTION update_message_stats()
      RETURNS TRIGGER AS $$
      BEGIN
          IF TG_OP = 'INSERT' THEN
              INSERT INTO message_stats (user_id, session_id, total_messages, total_responses, last_message_at)
              VALUES (
                  NEW.user_id, 
                  NEW.session_id, 
                  1, 
                  CASE WHEN NEW.response IS NOT NULL THEN 1 ELSE 0 END,
                  NEW.timestamp
              )
              ON CONFLICT (user_id, session_id) 
              DO UPDATE SET 
                  total_messages = message_stats.total_messages + 1,
                  total_responses = message_stats.total_responses + CASE WHEN NEW.response IS NOT NULL THEN 1 ELSE 0 END,
                  last_message_at = NEW.timestamp,
                  last_updated = CURRENT_TIMESTAMP;
              RETURN NEW;
          ELSIF TG_OP = 'DELETE' THEN
              UPDATE message_stats 
              SET 
                  total_messages = GREATEST(total_messages - 1, 0),
                  total_responses = GREATEST(total_responses - CASE WHEN OLD.response IS NOT NULL THEN 1 ELSE 0 END, 0),
                  last_updated = CURRENT_TIMESTAMP
              WHERE user_id = OLD.user_id AND session_id = OLD.session_id;
              RETURN OLD;
          END IF;
          RETURN NULL;
      END;
      $$ LANGUAGE plpgsql;
    `);

    await client.query(`
      DROP TRIGGER IF EXISTS messages_stats_trigger ON messages;
      CREATE TRIGGER messages_stats_trigger
      AFTER INSERT OR DELETE ON messages
      FOR EACH ROW
      EXECUTE FUNCTION update_message_stats();
    `);

    await client.query('COMMIT');
    console.log('Migration 022: Message stats table created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 022 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query('DROP TRIGGER IF EXISTS messages_stats_trigger ON messages');
    await client.query('DROP FUNCTION IF EXISTS update_message_stats()');
    await client.query('DROP TABLE IF EXISTS message_stats CASCADE');

    await client.query('COMMIT');
    console.log('Migration 022: Message stats table dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 022 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
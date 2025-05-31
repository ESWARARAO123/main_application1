-- Enable uuid-ossp extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Add missing columns to user_ssh_configurations if they do not exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='user_ssh_configurations' AND column_name='ssh_key_path'
    ) THEN
        ALTER TABLE user_ssh_configurations ADD COLUMN ssh_key_path TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='user_ssh_configurations' AND column_name='ssh_password_encrypted'
    ) THEN
        ALTER TABLE user_ssh_configurations ADD COLUMN ssh_password_encrypted TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='user_ssh_configurations' AND column_name='last_ssh_connection_status'
    ) THEN
        ALTER TABLE user_ssh_configurations ADD COLUMN last_ssh_connection_status VARCHAR(50) DEFAULT 'unknown';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='user_ssh_configurations' AND column_name='last_ssh_error_message'
    ) THEN
        ALTER TABLE user_ssh_configurations ADD COLUMN last_ssh_error_message TEXT;
    END IF;
END$$;

-- Add index if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = 'idx_user_ssh_configurations_user_id'
          AND n.nspname = 'public'
    ) THEN
        CREATE INDEX idx_user_ssh_configurations_user_id ON user_ssh_configurations(user_id);
    END IF;
END$$;
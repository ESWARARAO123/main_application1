-- Add is_context_update column to messages table
-- This migration adds a boolean column to track context update messages
-- Version: 1.0.0
-- Date: 2024-07-10

-- Check if the column already exists
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'messages' AND column_name = 'is_context_update'
    ) THEN
        -- Add the column
        ALTER TABLE messages ADD COLUMN is_context_update BOOLEAN DEFAULT FALSE;
        
        -- Log the migration
        INSERT INTO schema_migrations (version, description, applied_at)
        VALUES ('20240710001', 'add_is_context_update_to_messages', NOW())
        ON CONFLICT (version) DO NOTHING;
        
        RAISE NOTICE 'Added is_context_update column to messages table';
    ELSE
        RAISE NOTICE 'is_context_update column already exists in messages table';
    END IF;
END $$;

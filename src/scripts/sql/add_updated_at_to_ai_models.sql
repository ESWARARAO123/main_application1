-- Add updated_at column to ai_models if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'ai_models' AND column_name = 'updated_at'
    ) THEN
        -- Add the column
        ALTER TABLE ai_models ADD COLUMN updated_at TIMESTAMP DEFAULT NOW();
        
        -- Log the migration
        INSERT INTO schema_migrations (version, description, applied_at)
        VALUES ('20240615003', 'add_updated_at_to_ai_models', NOW())
        ON CONFLICT (version) DO NOTHING;
    END IF;
END $$; 
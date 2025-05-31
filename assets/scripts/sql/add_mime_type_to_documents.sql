-- Add mime_type column to documents table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'mime_type'
    ) THEN
        -- Add the column
        ALTER TABLE documents ADD COLUMN mime_type VARCHAR(255);
        
        -- Log the migration if schema_migrations table exists
        IF EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = 'schema_migrations'
        ) THEN
            INSERT INTO schema_migrations (version, description, applied_at)
            VALUES ('20240502001', 'add_mime_type_to_documents', NOW())
            ON CONFLICT (version) DO NOTHING;
        END IF;
        
        RAISE NOTICE 'Added mime_type column to documents table';
    ELSE
        RAISE NOTICE 'mime_type column already exists in documents table';
    END IF;
END $$;

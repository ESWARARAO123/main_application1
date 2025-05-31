-- Add a default value for the filename column in the documents table
DO $$ 
BEGIN
    -- First check if the column exists
    IF EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'filename'
    ) THEN
        -- Alter the column to set a default value based on original_name
        ALTER TABLE documents 
        ALTER COLUMN filename SET DEFAULT '';
        
        -- Log the migration if schema_migrations table exists
        IF EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = 'schema_migrations'
        ) THEN
            INSERT INTO schema_migrations (version, description, applied_at)
            VALUES ('20240502003', 'update_documents_filename_default', NOW())
            ON CONFLICT (version) DO NOTHING;
        END IF;
        
        RAISE NOTICE 'Updated filename column in documents table to have a default value';
    ELSE
        RAISE NOTICE 'filename column does not exist in documents table';
    END IF;
END $$;

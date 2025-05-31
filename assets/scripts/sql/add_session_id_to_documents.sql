-- Add session_id column to documents table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'session_id'
    ) THEN
        -- Add the column
        ALTER TABLE documents ADD COLUMN session_id UUID;
        
        -- Add foreign key constraint if chat_sessions table exists
        IF EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = 'chat_sessions'
        ) THEN
            ALTER TABLE documents 
            ADD CONSTRAINT fk_documents_session 
            FOREIGN KEY (session_id) 
            REFERENCES chat_sessions(id) 
            ON DELETE SET NULL;
            
            -- Create index for better performance
            CREATE INDEX idx_documents_session_id ON documents(session_id);
        END IF;
        
        -- Log the migration if schema_migrations table exists
        IF EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = 'schema_migrations'
        ) THEN
            INSERT INTO schema_migrations (version, description, applied_at)
            VALUES ('20240502002', 'add_session_id_to_documents', NOW())
            ON CONFLICT (version) DO NOTHING;
        END IF;
        
        RAISE NOTICE 'Added session_id column to documents table';
    ELSE
        RAISE NOTICE 'session_id column already exists in documents table';
    END IF;
END $$;

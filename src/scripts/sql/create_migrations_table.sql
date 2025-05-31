-- Create schema_migrations table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'schema_migrations'
    ) THEN
        CREATE TABLE public.schema_migrations (
            id SERIAL PRIMARY KEY,
            version VARCHAR(50) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT,
            CONSTRAINT schema_migrations_version_key UNIQUE (version)
        );

        CREATE INDEX idx_schema_migrations_version ON public.schema_migrations(version);
        
        RAISE NOTICE 'Created schema_migrations table';
    ELSE
        RAISE NOTICE 'schema_migrations table already exists';
    END IF;
END;
$$; 
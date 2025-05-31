-- Add model_id column to ai_models if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'ai_models' AND column_name = 'model_id'
    ) THEN
        -- Add the column
        ALTER TABLE ai_models ADD COLUMN model_id VARCHAR(255);
        
        -- Update existing records to use ollama_model_id as model_id
        UPDATE ai_models SET model_id = ollama_model_id WHERE model_id IS NULL;
        
        -- Add NOT NULL constraint after populating data
        ALTER TABLE ai_models ALTER COLUMN model_id SET NOT NULL;
        
        -- Log the migration
        INSERT INTO schema_migrations (version, description, applied_at)
        VALUES ('20240823001', 'add_model_id_to_ai_models', NOW())
        ON CONFLICT (version) DO NOTHING;
    END IF;
END $$; 
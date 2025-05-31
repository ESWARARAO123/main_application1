-- Create ai_models table for storing AI model information
CREATE TABLE IF NOT EXISTS ai_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id VARCHAR(255) NOT NULL, -- Model identifier used in chat
    ollama_model_id VARCHAR(255) NOT NULL, -- The ID used by Ollama (e.g., "llama3:latest")
    name VARCHAR(255) NOT NULL, -- Display name
    description TEXT, -- Model description
    parameters JSONB, -- Model parameters (size, etc.)
    is_active BOOLEAN NOT NULL DEFAULT false, -- Whether the model is available for use
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ai_models_ollama_model_id ON ai_models(ollama_model_id);
CREATE INDEX IF NOT EXISTS idx_ai_models_is_active ON ai_models(is_active);
CREATE INDEX IF NOT EXISTS idx_ai_models_name ON ai_models(name);

-- Add to schema_migrations
INSERT INTO schema_migrations (version, description, applied_at)
VALUES ('20240615002', 'create_ai_models', NOW())
ON CONFLICT (version) DO NOTHING; 
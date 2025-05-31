-- Create ollama_settings table for Ollama server connection configuration
CREATE TABLE IF NOT EXISTS ollama_settings (
    id SERIAL PRIMARY KEY,
    host VARCHAR(255) NOT NULL DEFAULT 'localhost',
    port INTEGER NOT NULL DEFAULT 11434,
    default_model VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index on updated_at
CREATE INDEX IF NOT EXISTS idx_ollama_settings_updated_at ON ollama_settings(updated_at);

-- Insert default settings if the table is empty
INSERT INTO ollama_settings (host, port, default_model, created_at, updated_at)
SELECT 'localhost', 11434, '', NOW(), NOW()
WHERE NOT EXISTS (SELECT 1 FROM ollama_settings);

-- Add to schema_migrations
INSERT INTO schema_migrations (version, description, applied_at)
VALUES ('20240615001', 'create_ollama_settings', NOW())
ON CONFLICT (version) DO NOTHING; 
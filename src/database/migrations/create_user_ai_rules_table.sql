-- Create user_ai_rules table to store user-specific AI context rules
CREATE TABLE IF NOT EXISTS user_ai_rules (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rule_content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user_id FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Add index for faster lookups by user_id
CREATE INDEX IF NOT EXISTS idx_user_ai_rules_user_id ON user_ai_rules(user_id);

-- Add a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_user_ai_rules_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to automatically update the updated_at timestamp
DROP TRIGGER IF EXISTS trigger_update_user_ai_rules_updated_at ON user_ai_rules;
CREATE TRIGGER trigger_update_user_ai_rules_updated_at
BEFORE UPDATE ON user_ai_rules
FOR EACH ROW
EXECUTE FUNCTION update_user_ai_rules_updated_at();

-- Add this migration to schema_migrations table if it doesn't exist already
INSERT INTO schema_migrations (version, applied_at, description)
SELECT '20240521001', CURRENT_TIMESTAMP, 'Create user_ai_rules table'
WHERE NOT EXISTS (
    SELECT 1 FROM schema_migrations WHERE version = '20240521001'
);

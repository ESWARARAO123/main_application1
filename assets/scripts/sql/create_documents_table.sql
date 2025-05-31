-- Create documents table if it doesn't exist
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  user_id UUID NOT NULL,
  filename VARCHAR(255) NOT NULL,
  original_name VARCHAR(255) NOT NULL,
  file_path TEXT NOT NULL,
  file_type VARCHAR(100) NOT NULL,
  file_size BIGINT NOT NULL,
  status VARCHAR(50) DEFAULT 'UPLOADED',
  processing_error TEXT,
  collection_id UUID,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Add file_path and document_id columns to messages table if they don't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'messages'
    AND column_name = 'file_path'
  ) THEN
    ALTER TABLE messages ADD COLUMN file_path TEXT;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'messages'
    AND column_name = 'document_id'
  ) THEN
    ALTER TABLE messages ADD COLUMN document_id INTEGER;
  END IF;
END $$;

-- Add foreign key constraint if it doesn't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.table_constraints
    WHERE constraint_name = 'fk_document'
    AND table_name = 'messages'
  ) THEN
    ALTER TABLE messages
    ADD CONSTRAINT fk_document
    FOREIGN KEY (document_id)
    REFERENCES documents(id)
    ON DELETE SET NULL;
  END IF;
END $$;

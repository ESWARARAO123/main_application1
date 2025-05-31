-- Drop the existing documents table if it exists
DROP TABLE IF EXISTS documents CASCADE;

-- Create documents table with integer ID
CREATE TABLE documents (
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

-- Remove existing document_id column from messages table if it exists
ALTER TABLE messages DROP COLUMN IF EXISTS document_id;
ALTER TABLE messages DROP COLUMN IF EXISTS file_path;

-- Add document_id column to messages table
ALTER TABLE messages ADD COLUMN document_id INTEGER;
ALTER TABLE messages ADD COLUMN file_path TEXT;

-- Add foreign key constraint
ALTER TABLE messages
ADD CONSTRAINT fk_document
FOREIGN KEY (document_id)
REFERENCES documents(id)
ON DELETE SET NULL;

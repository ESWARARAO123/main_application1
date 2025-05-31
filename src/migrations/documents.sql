-- Add session_id column to documents table
ALTER TABLE documents 
ADD COLUMN session_id UUID REFERENCES chat_sessions(id) ON DELETE SET NULL;

-- Create index on session_id for faster lookups
CREATE INDEX idx_documents_session_id ON documents(session_id);

-- Add comment explaining the purpose of the column
COMMENT ON COLUMN documents.session_id IS 'Reference to the chat session this document was uploaded in. Used for RAG context isolation.'; 
-- RAG Complete Setup Script
-- Creates all needed tables for RAG integration from scratch
-- Version: 2.0.0
-- Date: 2024-06-30

-- First, check if we need to create the migrations tracking table
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
    END IF;
END;
$$;

-- Check if this migration has already been applied
DO $$
DECLARE
    vector_ext_available BOOLEAN := FALSE;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '2.0.0' AND description = 'RAG Complete Setup') THEN
        RAISE NOTICE 'Starting RAG complete setup...';

        -- Create vector_stores table
        CREATE TABLE IF NOT EXISTS public.vector_stores (
            id UUID DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            store_type VARCHAR(50) NOT NULL DEFAULT 'chroma',
            connection_string TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            config JSONB
        );

        CREATE INDEX IF NOT EXISTS idx_vector_stores_is_active ON public.vector_stores(is_active);
        CREATE INDEX IF NOT EXISTS idx_vector_stores_name ON public.vector_stores(name);

        -- Create default vector store
        INSERT INTO public.vector_stores (name, store_type, connection_string, config)
        VALUES ('Default ChromaDB', 'chroma', 'http://localhost:8000',
                '{"persist_directory": "./DATA/chroma_data", "client_settings": {"anonymized_telemetry": false}}')
        ON CONFLICT DO NOTHING;

        RAISE NOTICE 'Created vector_stores table and default store';

        -- Create document_collections table
        CREATE TABLE IF NOT EXISTS public.document_collections (
            id UUID DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
            vector_store_id UUID REFERENCES public.vector_stores(id) ON DELETE SET NULL,
            embedding_model VARCHAR(255) DEFAULT 'ollama/nomic-embed-text',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB
        );

        CREATE INDEX IF NOT EXISTS idx_document_collections_user_id ON public.document_collections(user_id);
        CREATE INDEX IF NOT EXISTS idx_document_collections_is_active ON public.document_collections(is_active);

        RAISE NOTICE 'Created document_collections table';

        -- Create documents table
        CREATE TABLE IF NOT EXISTS public.documents (
            id UUID DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
            collection_id UUID REFERENCES public.document_collections(id) ON DELETE SET NULL,
            title VARCHAR(255) NOT NULL,
            file_path TEXT NOT NULL,
            file_name VARCHAR(255) NOT NULL,
            file_type VARCHAR(50) NOT NULL,
            content_text TEXT,
            content_hash VARCHAR(64),
            vector_id VARCHAR(255),
            processing_status VARCHAR(20) DEFAULT 'pending' NOT NULL, -- 'pending', 'processing', 'processed', 'failed'
            processing_error TEXT,
            is_indexed BOOLEAN DEFAULT FALSE,
            chunk_count INTEGER DEFAULT 0,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP,
            indexed_at TIMESTAMP,
            last_accessed_at TIMESTAMP,
            metadata JSONB
        );

        CREATE INDEX IF NOT EXISTS idx_documents_user_id ON public.documents(user_id);
        CREATE INDEX IF NOT EXISTS idx_documents_collection_id ON public.documents(collection_id);
        CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON public.documents(content_hash);
        CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON public.documents(processing_status);

        RAISE NOTICE 'Created documents table';

        -- Check if the vector extension is available
        -- Use a safer approach without nested BEGIN-EXCEPTION blocks
        SELECT EXISTS (
            SELECT 1 FROM pg_available_extensions WHERE name = 'vector'
        ) INTO vector_ext_available;

        -- Try to create vector extension if available
        IF vector_ext_available THEN
            CREATE EXTENSION IF NOT EXISTS vector;
            RAISE NOTICE 'Vector extension enabled successfully';

            -- Create document_chunks table with vector type
            CREATE TABLE IF NOT EXISTS public.document_chunks (
                id UUID DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                vector_id VARCHAR(255),
                embedding VECTOR(1536),
                token_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            );

            RAISE NOTICE 'Created document_chunks table with vector type';
        ELSE
            -- Create document_chunks table with BYTEA type as fallback
            RAISE NOTICE 'Vector extension not available, using BYTEA fallback';

            CREATE TABLE IF NOT EXISTS public.document_chunks (
                id UUID DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                vector_id VARCHAR(255),
                embedding BYTEA, -- Store as binary instead of vector
                token_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            );

            RAISE NOTICE 'Created document_chunks table with BYTEA type as fallback';
        END IF;

        CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON public.document_chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_document_chunks_vector_id ON public.document_chunks(vector_id);

        -- Create rag_settings table
        CREATE TABLE IF NOT EXISTS public.rag_settings (
            id SERIAL PRIMARY KEY,
            embedding_model VARCHAR(255) DEFAULT 'ollama/nomic-embed-text',
            chunk_size INTEGER DEFAULT 1000,
            chunk_overlap INTEGER DEFAULT 200,
            similarity_top_k INTEGER DEFAULT 4,
            search_type VARCHAR(20) DEFAULT 'similarity',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            config JSONB
        );

        -- Insert default settings
        INSERT INTO public.rag_settings (
            embedding_model, chunk_size, chunk_overlap, similarity_top_k, search_type, config
        ) VALUES (
            'ollama/nomic-embed-text', 1000, 200, 4, 'similarity',
            '{"use_hybrid_search": false, "rerank_results": false, "max_token_limit": 4000}'
        )
        ON CONFLICT DO NOTHING;

        RAISE NOTICE 'Created rag_settings table and default settings';

        -- Add retrieved_chunks column to messages table if it doesn't exist
        ALTER TABLE public.messages ADD COLUMN IF NOT EXISTS retrieved_chunks JSONB;

        -- Add RAG-related columns to chat_sessions table if they don't exist
        ALTER TABLE public.chat_sessions ADD COLUMN IF NOT EXISTS use_rag BOOLEAN DEFAULT TRUE;
        ALTER TABLE public.chat_sessions ADD COLUMN IF NOT EXISTS rag_collections JSONB;

        RAISE NOTICE 'Updated existing tables with RAG columns';

        -- Create a trigger function to update the updated_at timestamp
        CREATE OR REPLACE FUNCTION update_timestamp_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        -- Add triggers to update the updated_at column
        DROP TRIGGER IF EXISTS update_vector_stores_timestamp ON public.vector_stores;
        CREATE TRIGGER update_vector_stores_timestamp
        BEFORE UPDATE ON public.vector_stores
        FOR EACH ROW
        EXECUTE FUNCTION update_timestamp_column();

        DROP TRIGGER IF EXISTS update_document_collections_timestamp ON public.document_collections;
        CREATE TRIGGER update_document_collections_timestamp
        BEFORE UPDATE ON public.document_collections
        FOR EACH ROW
        EXECUTE FUNCTION update_timestamp_column();

        DROP TRIGGER IF EXISTS update_rag_settings_timestamp ON public.rag_settings;
        CREATE TRIGGER update_rag_settings_timestamp
        BEFORE UPDATE ON public.rag_settings
        FOR EACH ROW
        EXECUTE FUNCTION update_timestamp_column();

        RAISE NOTICE 'Created timestamp update triggers';

        -- Create directories function to make sure directories exist
        CREATE OR REPLACE FUNCTION ensure_document_directories()
        RETURNS TRIGGER AS $$
        DECLARE
            doc_dir TEXT;
        BEGIN
            -- Generate directory path strings
            doc_dir := 'DATA/documents/' || NEW.user_id::text || '/' || NEW.id::text;

            -- Create directories via plpgsql - requires admin privileges
            EXECUTE format('SELECT pg_catalog.pg_file_write(%L, %L, false)',
                           doc_dir || '/.placeholder', '', false);

            RAISE NOTICE 'Created document directory: %', doc_dir;
            RETURN NEW;
        EXCEPTION
            WHEN OTHERS THEN
                RAISE NOTICE 'Could not create document directories: %. This is normal in some environments.', SQLERRM;
                RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        -- Add trigger to create directories when a new document is created
        -- This may not work in all environments due to permissions
        DROP TRIGGER IF EXISTS create_document_directories ON public.documents;
        CREATE TRIGGER create_document_directories
        AFTER INSERT ON public.documents
        FOR EACH ROW
        EXECUTE FUNCTION ensure_document_directories();

        RAISE NOTICE 'Created directory management trigger (may not work in all environments)';

        -- Record migration in schema_migrations table
        INSERT INTO schema_migrations (version, applied_at, description)
        VALUES ('2.0.0', CURRENT_TIMESTAMP, 'RAG Complete Setup');

        RAISE NOTICE 'RAG setup complete and migration recorded';
    ELSE
        RAISE NOTICE 'RAG setup already applied - skipping';
    END IF;
END;
$$;

-- Create file upload function
CREATE OR REPLACE FUNCTION upload_document(
    p_user_id UUID,
    p_title TEXT,
    p_file_name TEXT,
    p_file_type TEXT,
    p_collection_id UUID DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    doc_id UUID;
    doc_path TEXT;
BEGIN
    -- Create new document record
    INSERT INTO public.documents (
        user_id,
        title,
        file_name,
        file_type,
        collection_id,
        file_path,
        processing_status
    ) VALUES (
        p_user_id,
        p_title,
        p_file_name,
        p_file_type,
        p_collection_id,
        '', -- File path will be updated after document ID is known
        'pending'
    ) RETURNING id INTO doc_id;

    -- Set file path now that we have the doc_id
    doc_path := 'DATA/documents/' || p_user_id || '/' || doc_id || '/' || p_file_name;

    -- Update the document with the file path
    UPDATE public.documents
    SET file_path = doc_path
    WHERE id = doc_id;

    RETURN doc_id;
END;
$$ LANGUAGE plpgsql;

-- Create function to update document status
CREATE OR REPLACE FUNCTION update_document_status(
    p_document_id UUID,
    p_status TEXT,
    p_error TEXT DEFAULT NULL,
    p_content_text TEXT DEFAULT NULL,
    p_chunk_count INTEGER DEFAULT NULL
)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE public.documents
    SET
        processing_status = p_status,
        processing_error = p_error,
        content_text = COALESCE(p_content_text, content_text),
        chunk_count = COALESCE(p_chunk_count, chunk_count),
        processed_at = CASE WHEN p_status IN ('processed', 'failed') THEN CURRENT_TIMESTAMP ELSE processed_at END
    WHERE id = p_document_id;

    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Create document chunk storage function
CREATE OR REPLACE FUNCTION store_document_chunk(
    p_document_id UUID,
    p_chunk_index INTEGER,
    p_content TEXT,
    p_vector_id TEXT DEFAULT NULL,
    p_token_count INTEGER DEFAULT NULL,
    p_metadata JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    chunk_id UUID;
BEGIN
    INSERT INTO public.document_chunks (
        document_id,
        chunk_index,
        content,
        vector_id,
        token_count,
        metadata
    ) VALUES (
        p_document_id,
        p_chunk_index,
        p_content,
        p_vector_id,
        p_token_count,
        p_metadata
    ) RETURNING id INTO chunk_id;

    RETURN chunk_id;
END;
$$ LANGUAGE plpgsql;

-- Add a function to quickly retrieve recent documents
CREATE OR REPLACE FUNCTION get_recent_documents(
    p_user_id UUID,
    p_limit INTEGER DEFAULT 20,
    p_offset INTEGER DEFAULT 0
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    file_name TEXT,
    file_type TEXT,
    processing_status TEXT,
    uploaded_at TIMESTAMP,
    collection_id UUID,
    collection_name TEXT,
    chunk_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.title,
        d.file_name,
        d.file_type,
        d.processing_status,
        d.uploaded_at,
        d.collection_id,
        c.name as collection_name,
        d.chunk_count
    FROM
        public.documents d
    LEFT JOIN
        public.document_collections c ON d.collection_id = c.id
    WHERE
        d.user_id = p_user_id
    ORDER BY
        d.uploaded_at DESC
    LIMIT p_limit
    OFFSET p_offset;
END;
$$ LANGUAGE plpgsql;

-- Create function to retrieve document chunks for embedding update
CREATE OR REPLACE FUNCTION get_unembedded_chunks(
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_index INTEGER,
    content TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.chunk_index,
        dc.content
    FROM
        public.document_chunks dc
    WHERE
        dc.embedding IS NULL
    ORDER BY
        dc.created_at ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Create function to update chunk embedding
CREATE OR REPLACE FUNCTION update_chunk_embedding(
    p_chunk_id UUID,
    p_embedding BYTEA,
    p_vector_id TEXT DEFAULT NULL
)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE public.document_chunks
    SET
        embedding = p_embedding,
        vector_id = COALESCE(p_vector_id, vector_id)
    WHERE id = p_chunk_id;

    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;
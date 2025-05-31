/**
 * Migration to create document processing functions
 */

const { pool } = require('../database');

async function up() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create upload_document function
    await client.query(`
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
              filename,
              original_name,
              file_path,
              file_type,
              collection_id,
              status
          ) VALUES (
              p_user_id,
              p_title,
              p_file_name,
              p_file_name,
              '', -- File path will be updated after document ID is known
              p_file_type,
              p_collection_id,
              'UPLOADED'
          ) RETURNING id INTO doc_id;
          
          -- Set file path now that we have the doc_id
          doc_path := 'Documents/' || p_user_id || '/' || doc_id || '/' || p_file_name;
          
          -- Update the document with the file path
          UPDATE public.documents 
          SET file_path = doc_path
          WHERE id = doc_id;
          
          RETURN doc_id;
      END;
      $$ LANGUAGE plpgsql;
    `);

    // Create update_document_status function
    await client.query(`
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
              status = p_status,
              processing_error = p_error,
              updated_at = CURRENT_TIMESTAMP
          WHERE id = p_document_id;
          
          RETURN FOUND;
      END;
      $$ LANGUAGE plpgsql;
    `);

    // Create store_document_chunk function
    await client.query(`
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
    `);

    // Create update_chunk_embedding function
    await client.query(`
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
    `);

    // Create get_unembedded_chunks function
    await client.query(`
      CREATE OR REPLACE FUNCTION get_unembedded_chunks(p_limit INTEGER DEFAULT 100)
      RETURNS TABLE(
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
    `);

    // Create get_recent_documents function
    // Check if function already exists with correct signature
    const existingFunction = await client.query(`
      SELECT 1 FROM pg_proc p 
      JOIN pg_namespace n ON p.pronamespace = n.oid 
      WHERE n.nspname = 'public' AND p.proname = 'get_recent_documents'
    `);
    
    if (existingFunction.rows.length > 0) {
      console.log('Migration 018: get_recent_documents function already exists, skipping creation');
    } else {
      await client.query(`
        CREATE OR REPLACE FUNCTION get_recent_documents(
          p_user_id UUID, 
          p_limit INTEGER DEFAULT 20, 
          p_offset INTEGER DEFAULT 0
        )
        RETURNS TABLE(
          id INTEGER, 
          title TEXT, 
          filename TEXT, 
          file_type TEXT, 
          status TEXT, 
          created_at TIMESTAMP WITH TIME ZONE, 
          collection_id UUID, 
          collection_name TEXT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                d.id,
                d.original_name as title,
                d.filename,
                d.file_type,
                d.status,
                d.created_at,
                d.collection_id,
                c.name as collection_name
            FROM 
                public.documents d
            LEFT JOIN
                public.document_collections c ON d.collection_id = c.id
            WHERE 
                d.user_id = p_user_id
            ORDER BY 
                d.created_at DESC
            LIMIT p_limit
            OFFSET p_offset;
        END;
        $$ LANGUAGE plpgsql;
      `);
    }

    await client.query('COMMIT');
    console.log('Migration 018: Document processing functions created successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 018 failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

async function down() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Drop all document functions
    await client.query('DROP FUNCTION IF EXISTS upload_document(UUID, TEXT, TEXT, TEXT, UUID)');
    await client.query('DROP FUNCTION IF EXISTS update_document_status(UUID, TEXT, TEXT, TEXT, INTEGER)');
    await client.query('DROP FUNCTION IF EXISTS store_document_chunk(UUID, INTEGER, TEXT, TEXT, INTEGER, JSONB)');
    await client.query('DROP FUNCTION IF EXISTS update_chunk_embedding(UUID, BYTEA, TEXT)');
    await client.query('DROP FUNCTION IF EXISTS get_unembedded_chunks(INTEGER)');
    await client.query('DROP FUNCTION IF EXISTS get_recent_documents(UUID, INTEGER, INTEGER)');

    await client.query('COMMIT');
    console.log('Migration 018: Document processing functions dropped successfully');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Migration 018 rollback failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

module.exports = { up, down }; 
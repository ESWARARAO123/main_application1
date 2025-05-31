-- Update Functions After Table Removal
-- This script updates database functions to remove references to deleted tables (pdfs and message_files)
-- Version: 1.0.0
-- Date: 2024-07-02

-- Check if the migration was already applied
SELECT 1 FROM schema_migrations WHERE version = '2.1.0' AND description = 'Update Functions After Table Removal';

-- If the migration hasn't been applied yet, proceed with the changes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '2.1.0' AND description = 'Update Functions After Table Removal') THEN
        RAISE NOTICE 'Applying update_functions_after_table_removal migration...';

        -- 1. Drop and recreate the update_dashboard_metrics function to use documents instead of pdfs
        DROP FUNCTION IF EXISTS public.update_dashboard_metrics();

        -- Create the new function
        RAISE NOTICE 'Creating updated dashboard_metrics function...';
    ELSE
        RAISE NOTICE 'Migration already applied - skipping';
    END IF;
END;
$$;

-- Create the updated function (outside of DO block to avoid nesting issues)
CREATE OR REPLACE FUNCTION public.update_dashboard_metrics() RETURNS void
LANGUAGE plpgsql
AS $func$
DECLARE
  user_stats jsonb;
  message_stats jsonb;
  document_count integer;
BEGIN
  -- Get user statistics
  SELECT jsonb_build_object(
    'totalUsers', COUNT(*),
    'adminUsers', COUNT(CASE WHEN role = 'admin' THEN 1 END),
    'regularUsers', COUNT(CASE WHEN role = 'user' THEN 1 END),
    'recentUsers', COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END)
  ) INTO user_stats FROM users;

  -- Get message statistics
  SELECT jsonb_build_object(
    'totalMessages', COUNT(*),
    'recentMessages', COUNT(CASE WHEN timestamp > NOW() - INTERVAL '7 days' THEN 1 END),
    'avgResponseTime', 0
  ) INTO message_stats FROM messages;

  -- Get document count - query documents table instead of pdfs
  SELECT COALESCE(COUNT(*), 0) INTO document_count FROM documents;

  -- Update message_stats with document count
  message_stats := message_stats || jsonb_build_object('totalDocuments', document_count);

  -- Update the dashboard metrics table
  UPDATE dashboard_metrics SET metric_value = user_stats, updated_at = NOW() WHERE metric_name = 'user_stats';
  UPDATE dashboard_metrics SET metric_value = message_stats, updated_at = NOW() WHERE metric_name = 'message_stats';

  -- Update license usage
  UPDATE dashboard_metrics
  SET metric_value = jsonb_build_object(
    'totalLicenses', 25,
    'activeLicenses', LEAST((user_stats->>'totalUsers')::integer, 12),
    'expirationDate', '2024-12-31',
    'daysRemaining', 245
  ),
  updated_at = NOW()
  WHERE metric_name = 'license_usage';
END;
$func$;

-- Continue with the migration if it hasn't been applied yet
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '2.1.0' AND description = 'Update Functions After Table Removal') THEN
        -- 2. Update existing dashboard_metrics entries to use totalDocuments instead of totalPdfs
        RAISE NOTICE 'Updating dashboard metrics to use totalDocuments instead of totalPdfs...';

        UPDATE dashboard_metrics
        SET metric_value = jsonb_set(
            metric_value,
            '{totalDocuments}',
            COALESCE(metric_value->'totalPdfs', '0')::jsonb
        )
        WHERE metric_name = 'message_stats' AND metric_value ? 'totalPdfs';

        UPDATE dashboard_metrics
        SET metric_value = metric_value - 'totalPdfs'
        WHERE metric_name = 'message_stats' AND metric_value ? 'totalPdfs';

        RAISE NOTICE 'Dashboard metrics updated.';
    END IF;
END;
$$;

-- 3. Drop any constraints referencing pdfs or message_files
DO $$
DECLARE
    r RECORD;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '2.1.0' AND description = 'Update Functions After Table Removal') THEN
        RAISE NOTICE 'Checking for constraints referencing pdfs or message_files...';

        FOR r IN (
            SELECT conname, conrelid::regclass::text as tablename, pg_get_constraintdef(oid) as def
            FROM pg_constraint
            WHERE pg_get_constraintdef(oid) LIKE '%pdfs%' OR pg_get_constraintdef(oid) LIKE '%message_files%'
        ) LOOP
            EXECUTE format('ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I', r.tablename, r.conname);
            RAISE NOTICE 'Dropped constraint % on table %', r.conname, r.tablename;
        END LOOP;
    END IF;
END;
$$;

-- 4. Remove pdf_id column from messages if it still exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '2.1.0' AND description = 'Update Functions After Table Removal') THEN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'messages' AND column_name = 'pdf_id'
        ) THEN
            RAISE NOTICE 'Removing pdf_id column from messages table...';
            ALTER TABLE messages DROP COLUMN pdf_id;
            RAISE NOTICE 'Dropped pdf_id column from messages table';
        ELSE
            RAISE NOTICE 'pdf_id column does not exist in messages table - skipping';
        END IF;
    END IF;
END;
$$;

-- 5. Remove last_pdf_id column from mcp_connections if it still exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '2.1.0' AND description = 'Update Functions After Table Removal') THEN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'mcp_connections' AND column_name = 'last_pdf_id'
        ) THEN
            RAISE NOTICE 'Removing last_pdf_id column from mcp_connections table...';
            ALTER TABLE mcp_connections DROP COLUMN last_pdf_id;
            RAISE NOTICE 'Dropped last_pdf_id column from mcp_connections table';
        ELSE
            RAISE NOTICE 'last_pdf_id column does not exist in mcp_connections table - skipping';
        END IF;
    END IF;
END;
$$;

-- Record migration in schema_migrations table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '2.1.0' AND description = 'Update Functions After Table Removal') THEN
        RAISE NOTICE 'Recording migration in schema_migrations table...';

        INSERT INTO schema_migrations (version, applied_at, description)
        VALUES ('2.1.0', CURRENT_TIMESTAMP, 'Update Functions After Table Removal');

        RAISE NOTICE 'Database functions updated successfully to remove references to deleted tables';
    END IF;
END;
$$;
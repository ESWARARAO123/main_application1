-- Reset dashboard metrics
UPDATE dashboard_metrics 
SET 
  total_messages = 0,
  recent_messages = 0,
  total_documents = 0,
  avg_response_time = 0
WHERE id = 1;

-- Reset document count
DELETE FROM documents;

-- Reset chat messages and sessions
DELETE FROM chat_messages;
DELETE FROM chat_sessions;

-- Commit the changes
COMMIT;

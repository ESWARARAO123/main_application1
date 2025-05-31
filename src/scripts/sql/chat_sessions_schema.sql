-- Chat Sessions Schema
-- This schema creates the tables and indexes for chat sessions functionality

-- Create chat_sessions table if not exists
CREATE TABLE IF NOT EXISTS public.chat_sessions (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    title character varying(255) NOT NULL,
    last_message_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    is_active boolean DEFAULT true
);

-- Add primary key if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chat_sessions_pkey'
    ) THEN
        ALTER TABLE ONLY public.chat_sessions 
        ADD CONSTRAINT chat_sessions_pkey PRIMARY KEY (id);
    END IF;
END
$$;

-- Create messages table if not exists
CREATE TABLE IF NOT EXISTS public.messages (
    id bigint NOT NULL,
    user_id uuid NOT NULL,
    message text NOT NULL,
    response text,
    timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    session_id uuid
);

-- Add primary key if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'messages_pkey'
    ) THEN
        ALTER TABLE ONLY public.messages 
        ADD CONSTRAINT messages_pkey PRIMARY KEY (id);
    END IF;
END
$$;

-- Create indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_chat_sessions_timestamp 
ON public.chat_sessions USING btree (last_message_timestamp);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id 
ON public.chat_sessions USING btree (user_id);

CREATE INDEX IF NOT EXISTS idx_messages_session_id 
ON public.messages USING btree (session_id);

-- Add foreign key constraints if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chat_sessions_user_id_fkey'
    ) THEN
        ALTER TABLE ONLY public.chat_sessions
        ADD CONSTRAINT chat_sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_session'
    ) THEN
        ALTER TABLE ONLY public.messages
        ADD CONSTRAINT fk_session FOREIGN KEY (session_id) REFERENCES public.chat_sessions(id) ON DELETE CASCADE;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'messages_user_id_fkey'
    ) THEN
        ALTER TABLE ONLY public.messages
        ADD CONSTRAINT messages_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;
    END IF;
END
$$;

-- Create function to update chat session timestamp
CREATE OR REPLACE FUNCTION public.update_chat_session_timestamp()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  UPDATE chat_sessions
  SET last_message_timestamp = NEW.timestamp
  WHERE id = NEW.session_id;
  RETURN NEW;
END;
$$;

-- Create trigger if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'update_session_timestamp'
    ) THEN
        CREATE TRIGGER update_session_timestamp
        AFTER INSERT ON public.messages
        FOR EACH ROW
        EXECUTE FUNCTION public.update_chat_session_timestamp();
    END IF;
END
$$; 
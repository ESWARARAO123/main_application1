-- Create the database 'copilot'
CREATE DATABASE copilot;

-- Connect to the newly created database
\c copilot;

-- Set session parameters
SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

-- Create sequences
CREATE SEQUENCE public.dashboard_metrics_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

CREATE SEQUENCE public.mcp_connections_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

CREATE SEQUENCE public.messages_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

CREATE SEQUENCE public.ollama_settings_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

-- Create schema_migrations table for tracking migrations
CREATE TABLE IF NOT EXISTS public.schema_migrations (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) NOT NULL UNIQUE,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Create tables
CREATE TABLE public.users (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    username character varying(50) NOT NULL,
    password text NOT NULL,
    role character varying(20) DEFAULT 'user'::character varying NOT NULL,
    email character varying(100),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    name character varying(100)
);

CREATE TABLE public.sessions (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    token text NOT NULL,
    expires_at timestamp without time zone NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE public.ai_models (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    model_id character varying(255) NOT NULL,
    name character varying(100) NOT NULL,
    description text,
    parameters jsonb,
    is_active boolean DEFAULT false,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    ollama_model_id character varying(255) NOT NULL
);

CREATE TABLE public.ollama_settings (
    id SERIAL PRIMARY KEY,
    host VARCHAR(255) NOT NULL DEFAULT 'localhost',
    port INTEGER NOT NULL DEFAULT 11434,
    default_model VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE public.chat_sessions (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    title character varying(255) NOT NULL,
    last_message_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    is_active boolean DEFAULT true,
    use_rag BOOLEAN DEFAULT TRUE,
    rag_collections JSONB
);

CREATE TABLE public.dashboard_metrics (
    id integer NOT NULL,
    metric_name character varying(50) NOT NULL,
    metric_value jsonb NOT NULL,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE public.mcp_connections (
    id bigint NOT NULL,
    user_id uuid NOT NULL,
    mcp_host character varying(100) NOT NULL,
    mcp_port integer NOT NULL,
    status character varying(20) DEFAULT 'disconnected'::character varying,
    last_file_path text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE public.messages (
    id bigint NOT NULL,
    user_id uuid NOT NULL,
    message text NOT NULL,
    response text,
    file_path text,
    "timestamp" timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    session_id uuid,
    model_id uuid,
    retrieved_chunks JSONB
);

-- Set default values for sequences
ALTER TABLE ONLY public.dashboard_metrics ALTER COLUMN id SET DEFAULT nextval('public.dashboard_metrics_id_seq'::regclass);
ALTER TABLE ONLY public.mcp_connections ALTER COLUMN id SET DEFAULT nextval('public.mcp_connections_id_seq'::regclass);
ALTER TABLE ONLY public.messages ALTER COLUMN id SET DEFAULT nextval('public.messages_id_seq'::regclass);

-- Add constraints
ALTER TABLE ONLY public.ai_models ADD CONSTRAINT ai_models_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.chat_sessions ADD CONSTRAINT chat_sessions_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.dashboard_metrics ADD CONSTRAINT dashboard_metrics_metric_name_key UNIQUE (metric_name);
ALTER TABLE ONLY public.dashboard_metrics ADD CONSTRAINT dashboard_metrics_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.mcp_connections ADD CONSTRAINT mcp_connections_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.mcp_connections ADD CONSTRAINT mcp_connections_user_id_mcp_host_mcp_port_key UNIQUE (user_id, mcp_host, mcp_port);
ALTER TABLE ONLY public.messages ADD CONSTRAINT messages_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.sessions ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.users ADD CONSTRAINT users_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.users ADD CONSTRAINT users_username_key UNIQUE (username);

-- Add indexes
CREATE INDEX idx_chat_sessions_timestamp ON public.chat_sessions USING btree (last_message_timestamp);
CREATE INDEX idx_chat_sessions_user_id ON public.chat_sessions USING btree (user_id);
CREATE INDEX idx_messages_session_id ON public.messages USING btree (session_id);
CREATE INDEX idx_ollama_settings_updated_at ON public.ollama_settings USING btree (updated_at);
CREATE INDEX idx_ai_models_ollama_model_id ON public.ai_models USING btree (ollama_model_id);
CREATE INDEX idx_ai_models_is_active ON public.ai_models USING btree (is_active);
CREATE INDEX idx_ai_models_name ON public.ai_models USING btree (name);

-- Add foreign key constraints
ALTER TABLE ONLY public.chat_sessions ADD CONSTRAINT chat_sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;
ALTER TABLE ONLY public.messages ADD CONSTRAINT fk_session FOREIGN KEY (session_id) REFERENCES public.chat_sessions(id) ON DELETE CASCADE;
ALTER TABLE ONLY public.mcp_connections ADD CONSTRAINT mcp_connections_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;
ALTER TABLE ONLY public.messages ADD CONSTRAINT messages_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.ai_models(id);
ALTER TABLE ONLY public.messages ADD CONSTRAINT messages_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;
ALTER TABLE ONLY public.sessions ADD CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;

-- Create functions
CREATE FUNCTION public.update_dashboard_metrics() RETURNS void
    LANGUAGE plpgsql
    AS $$
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
  -- Get document count
  SELECT COUNT(*) INTO document_count FROM documents;
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
$$;

CREATE FUNCTION public.trigger_update_dashboard_metrics() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  PERFORM public.update_dashboard_metrics();
  RETURN NULL;
END;
$$;

CREATE FUNCTION public.update_chat_session_timestamp() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  UPDATE chat_sessions
  SET last_message_timestamp = NEW.timestamp
  WHERE id = NEW.session_id;
  
  RETURN NEW;
END;
$$;

-- Create triggers
CREATE TRIGGER messages_update_metrics AFTER INSERT OR DELETE OR UPDATE ON public.messages FOR EACH STATEMENT EXECUTE FUNCTION public.trigger_update_dashboard_metrics();
CREATE TRIGGER update_session_timestamp AFTER INSERT ON public.messages FOR EACH ROW EXECUTE FUNCTION public.update_chat_session_timestamp();
CREATE TRIGGER users_update_metrics AFTER INSERT OR DELETE OR UPDATE ON public.users FOR EACH STATEMENT EXECUTE FUNCTION public.trigger_update_dashboard_metrics();
CREATE TRIGGER documents_update_metrics AFTER INSERT OR DELETE OR UPDATE ON public.documents FOR EACH STATEMENT EXECUTE FUNCTION public.trigger_update_dashboard_metrics();

-- Set ownership
ALTER TABLE public.ai_models OWNER TO postgres;
ALTER TABLE public.chat_sessions OWNER TO postgres;
ALTER TABLE public.dashboard_metrics OWNER TO postgres;
ALTER TABLE public.dashboard_metrics_id_seq OWNER TO postgres;
ALTER TABLE public.mcp_connections OWNER TO postgres;
ALTER TABLE public.mcp_connections_id_seq OWNER TO postgres;
ALTER TABLE public.messages OWNER TO postgres;
ALTER TABLE public.messages_id_seq OWNER TO postgres;
ALTER TABLE public.ollama_settings OWNER TO postgres;
ALTER TABLE public.ollama_settings_id_seq OWNER TO postgres;
ALTER TABLE public.sessions OWNER TO postgres;
ALTER TABLE public.users OWNER TO postgres;
ALTER TABLE public.schema_migrations OWNER TO postgres;

-- Insert default values
INSERT INTO public.dashboard_metrics VALUES (1, 'user_stats', '{"totalUsers": 0, "adminUsers": 0, "regularUsers": 0, "recentUsers": 0}', '2023-01-01 00:00:00');
INSERT INTO public.dashboard_metrics VALUES (2, 'message_stats', '{"totalMessages": 0, "recentMessages": 0, "avgResponseTime": 0, "totalDocuments": 0}', '2023-01-01 00:00:00');
INSERT INTO public.dashboard_metrics VALUES (3, 'license_usage', '{"totalLicenses": 25, "activeLicenses": 0, "expirationDate": "2024-12-31", "daysRemaining": 245}', '2023-01-01 00:00:00');
INSERT INTO public.ollama_settings (host, port, default_model) VALUES ('localhost', 11434, 'llama3');

-- Record this as a migration
INSERT INTO schema_migrations (version, applied_at, description) VALUES ('1.0.0', CURRENT_TIMESTAMP, 'Initial database creation');

-- Completed
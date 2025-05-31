# Ollama AI Integration Guide

This document provides comprehensive information about the Ollama AI integration in the Platform Dashboard. It covers setup instructions, troubleshooting steps, and usage guidelines based on the plan to use Ollama's `/api/chat` endpoint.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Database Setup](#database-setup)
4. [Connection Configuration](#connection-configuration)
5. [Model Management](#model-management)
6. [Chat Interaction Workflow](#chat-interaction-workflow)
7. [Endpoint Choice: /api/chat](#endpoint-choice-apichat)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)
10. [Development Notes](#development-notes)

## Overview

The Platform Dashboard integrates with [Ollama](https://ollama.ai/), an open-source tool for running large language models locally or on a server. This integration allows users to:

- Configure connection to an Ollama server
- Test the connection
- Sync available models from the Ollama server to the application's database
- Enable/disable specific models for use in the chat
- Send chat messages (with history) to enabled Ollama models using the `/api/chat` endpoint
- Manage a default model setting

The integration consists of:

- Backend service (`ollamaService.js`) for communicating with the Ollama API
- Backend API routes (`/api/ollama/*`) for settings, models, testing, and chat
- Admin UI (`OllamaSettings.tsx`) for configuring the server connection and managing models
- Model selector component (`ModelSelector.tsx`) for the chat interface
- Database tables (`ollama_settings`, `ai_models`) for storing configuration and model information

## Prerequisites

Before using the Ollama integration, you need:

1. **Ollama Server**: Running locally or on an accessible server
   - Install Ollama: [https://ollama.ai/download](https://ollama.ai/download)
   - Default port is `11434`
2. **Required Models**: At least one model pulled on the Ollama server
   - Example: `ollama pull llama3`
   - Verify models: `ollama list`
3. **Network Access**: The Platform Dashboard backend server needs network access to the Ollama server's host and port
   - Check firewalls if servers are on different machines

## Database Setup

The Ollama integration requires specific database tables (`ollama_settings`, `ai_models`) as defined in `DatabaseStructure.md`. Ensure these tables exist and migrations have been applied.

- Run `node src/scripts/check-migrations.js` to verify
- Apply migrations using `node src/scripts/apply_schema.js <path_to_sql_file>` if needed

## Connection Configuration

### Admin Settings Interface

Admin users configure the Ollama connection in Settings > Ollama Integration:

1. **Host**: Hostname or IP of the Ollama server (e.g., `localhost`, `192.168.1.100`)
2. **Port**: Port number (default: `11434`)
3. **Default Model**: Default model name (e.g., `llama3:latest`)
4. **Test Connection**: Button to verify the backend can reach the Ollama server (calls `/api/version` or similar)
5. **Save Settings**: Stores the configuration in the `ollama_settings` table

## Model Management

### Syncing Models

1. After saving valid settings, click **"Fetch Available Models"** (optional preview) to see models directly from Ollama (`/api/tags`)
2. Click **"Sync Models"**. The backend fetches models from Ollama (`/api/tags`) and updates the `ai_models` table in the database. It stores the full model name (e.g., `llama3:latest`) in the `ollama_model_id` field

### Enabling/Disabling Models

- The admin UI lists models from the `ai_models` database table
- Use the toggle switches to enable (`is_active = true`) or disable (`is_active = false`) models
- Only enabled models will appear in the chat interface's model selector

## Chat Interaction Workflow

1. **User Accesses Chat:** Navigates to the Chat page
2. **Model Selection:** Selects an *enabled* model from the dropdown (populated by `GET /api/ollama/models/active`)
3. **User Sends Message:** Types a prompt and submits
4. **Frontend Request:** Sends the new message, the *entire current chat history*, and the selected model's database ID (`ai_models.id`) to the backend endpoint `POST /api/ollama/chat`
5. **Backend Processing:**
   a. Retrieves the actual Ollama model name (e.g., `llama3:latest`) using the provided database ID
   b. Formats the chat history into the `messages` array structure required by Ollama `/api/chat` (alternating `user` and `assistant` roles)
   c. Calls the configured Ollama server's `/api/chat` endpoint
   d. Streams the response back to the frontend
6. **Frontend Display:** Renders the streamed response in the chat UI
7. **(Persistence):** The backend should ideally save the user message and the full AI response to the `messages` database table after the stream completes

## Endpoint Choice: `/api/chat`

We chose Ollama's `/api/chat` endpoint over `/api/generate` because:

1. **Conversational Context:** It natively handles chat history via the `messages` array, which is essential for chatbots
2. **Simplicity:** Avoids manual context string management required by `/api/generate` (whose `context` parameter is deprecated)
3. **Future RAG/Tools:** Supports `tools` (function calling), providing a robust foundation for future Retrieval-Augmented Generation (RAG) or agentic capabilities

## Troubleshooting

### Connection Issues

- **Verify Ollama Server:** Is it running? (`ollama ps`, Task Manager). Does it have models? (`ollama list`)
- **Test Backend->Ollama:** Use `curl http://<host>:<port>/api/version` from the *backend server* machine. Check firewalls
- **Test App->Ollama:** Use the "Test Connection" button in admin settings
- **Check Settings:** Ensure Host/Port in UI match the running Ollama server
- **Logs:** Check application backend logs (`server.log`) and Ollama server logs for detailed errors

### Model Sync Issues

- **No Models Found:** Ensure models are pulled in Ollama (`ollama pull ...`). Verify connection settings are correct
- **Sync Errors:** Check backend logs for errors during the `/api/tags` call or database update

### Chat Issues

- **Model Not Found:** The selected model might have been deleted from Ollama after syncing. Re-sync models
- **Errors During Generation:** Check Ollama server logs. Ensure the model is compatible with `/api/chat`. Check backend logs for errors calling Ollama or streaming response
- **No Response:** Check network connection, Ollama server status, and logs

## API Reference (Application Backend)

- **Settings:**
  - `GET /api/ollama/settings` (Admin)
  - `PUT /api/ollama/settings` (Admin)
  - `POST /api/ollama/test-connection` (Admin)
- **Models:**
  - `GET /api/ollama/models/available` (Admin) - Fetches from Ollama server
  - `POST /api/ollama/models/sync` (Admin) - Syncs Ollama -> DB
  - `GET /api/ollama/models/db` (Admin) - Fetches from DB
  - `PUT /api/ollama/models/:id/status` (Admin) - Toggles active status in DB
  - `GET /api/ollama/models/active` (Authenticated) - Gets active models from DB
- **Chat:**
  - `POST /api/ollama/chat` (Authenticated)

## Development Notes

### Key Files

- Backend: `src/services/ollamaService.js`, `src/routes/ollama.js`
- Frontend: `client/src/services/ollamaService.ts`, `client/src/components/settings/OllamaSettings.tsx`, `client/src/pages/Chatbot.tsx`, `client/src/components/chat/ModelSelector.tsx`
- Database: `DatabaseStructure.md`, SQL migration scripts

### Security

- Backend API routes must enforce authentication (logged-in user) and authorization (admin role for settings/sync)
- Protect the Ollama server itself at the network level

### Future Improvements

- Implement response streaming
- Allow configuration of Ollama model parameters (`temperature`, `num_ctx`, etc.) [Ollama API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- Integrate RAG using `/api/chat` structure 
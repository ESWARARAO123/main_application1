# Frequently Asked Questions (FAQ)

## General

### Q: How do I reset my password?
A: You can reset your password by clicking "Forgot Password" on the login screen or by contacting an administrator who can reset it for you.

### Q: How do I create a new user account?
A: Only administrators can create new user accounts. Contact your system administrator to request a new account.

### Q: How do I change my display name or email?
A: You can update your profile information from the Profile page, accessible from the dropdown menu in the top-right corner.

## Chat Interface

### Q: Why can't I see my chat history?
A: Chat history is session-based. If you've logged out or cleared browser data, previous chat sessions may not be visible. Administrators can also configure chat retention policies.

### Q: How do I start a new chat session?
A: Click the "New Chat" button at the top of the chat interface to start a fresh conversation.

### Q: Can I export my chat conversations?
A: Yes, look for the "Export" option in the chat options menu (three dots) to download your conversation as text or JSON.

## AI Integration

### Q: What is Ollama and how does it work with the platform?
A: Ollama is an open-source tool for running large language models locally or on a server. Our platform connects to an Ollama server to provide AI chat capabilities while keeping your data within your infrastructure.

### Q: Why can't I connect to the Ollama server?
A: This could be due to several reasons:
1. The Ollama server might not be running
2. Incorrect host/port configuration
3. Network or firewall issues blocking the connection
4. The server might be overloaded or unresponsive

Check the server status, connection settings, and network configuration. Try using the "Test Connection" button in the admin settings.

### Q: Why don't I see any AI models in the dropdown?
A: Models must be:
1. Available on the Ollama server (run `ollama list` to verify)
2. Synchronized to the database (admin needs to run "Sync Models" in the settings)
3. Enabled by an administrator (toggled to "active" in the models list)

Contact your administrator if models are missing.

### Q: How do I add new AI models to the platform?
A: The process has two steps:
1. First, add the model to your Ollama server using the command: `ollama pull <model-name>`
2. Then, in the platform's admin settings, use the "Sync Models" button to update the database with newly available models
3. Enable the models you want to make available to users

### Q: Why is the AI response slow or timing out?
A: Large language model performance depends on:
1. The size and complexity of the selected model
2. Available hardware resources (CPU/GPU/RAM) on the Ollama server
3. Network conditions between the platform and Ollama server
4. Current server load

Try selecting a smaller model for faster responses, or check if the Ollama server has adequate resources.

### Q: Can I use custom models with the platform?
A: Yes, as long as they're compatible with Ollama. Add your custom model to Ollama first, then sync it through the admin interface.

### Q: Is my conversation data private?
A: Yes, conversation data stays within your infrastructure. The platform sends prompts to your Ollama server and receives responses, but no data is sent to external services unless you've explicitly configured external connections.

## Administration

### Q: How do I view system logs?
A: System logs are available to administrators in the Admin Dashboard under the "Logs" section.

### Q: How do I back up the database?
A: Database backup procedures depend on your deployment configuration. Refer to the administration guide or contact your database administrator.

### Q: How do I update the application?
A: Update procedures will be provided with each release. Generally, this involves pulling the latest code, installing dependencies, and running migration scripts.

## Security

### Q: How is user data protected?
A: The platform implements several security measures:
1. Authentication using secure sessions
2. Password hashing using bcrypt
3. Role-based access control
4. Database encryption (if configured)

### Q: Are there any security best practices for deployment?
A: Yes, we recommend:
1. Use HTTPS in production
2. Set up proper firewall rules
3. Consider using a secrets management solution for production
4. Implement proper file permissions on your configuration files 
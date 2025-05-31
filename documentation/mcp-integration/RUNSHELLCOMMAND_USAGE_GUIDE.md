# RunShellCommand Tool - Usage Guide

## 🎉 **Congratulations!** 

Your `runshellcommand` tool is now fully integrated with your AI chat application! The AI can now execute shell commands through your MCP infrastructure using natural language.

## 🚀 **How It Works**

### **Intelligent Command Generation**
The AI now has a comprehensive system prompt that teaches it:
- **When to use shell commands** (file operations, system info, processes, etc.)
- **How to write Linux commands** (ls, ps, df, grep, etc.)
- **Command best practices** (using -h flags, safe operations, etc.)
- **Result interpretation** (explaining output to users)

### **Seamless Integration**
- **MCP Server Detection**: Uses your configured MCP servers automatically
- **User Authentication**: Each user's commands run on their configured servers
- **Context Preservation**: Command history maintained in chat conversations
- **Error Handling**: Comprehensive error handling and user-friendly messages

## 📝 **Example Conversations**

### **File System Operations**

**User:** *"I want to see the list of folders available in this folder"*

**AI Response:**
```
I'll list the files and directories in the current folder for you.

{
  "tool": "runshellcommand",
  "parameters": {
    "command": "ls -la"
  }
}
```

**Result:** The AI executes `ls -la`, gets the output, and explains what files/folders are available.

---

### **System Information**

**User:** *"What's the disk usage on this system?"*

**AI Response:**
```
Let me check the disk usage for you.

{
  "tool": "runshellcommand",
  "parameters": {
    "command": "df -h"
  }
}
```

**Result:** Shows disk usage in human-readable format with explanations.

---

### **Process Management**

**User:** *"Show me what processes are running"*

**AI Response:**
```
I'll show you the currently running processes.

{
  "tool": "runshellcommand",
  "parameters": {
    "command": "ps aux"
  }
}
```

**Result:** Lists all processes with details and highlights important ones.

## 🎯 **Supported Command Categories**

The AI has been trained to handle these types of requests:

### **📁 File & Directory Operations**
- "list files" → `ls -la`
- "show current directory" → `pwd`
- "create a directory" → `mkdir dirname`
- "find files" → `find /path -name "pattern"`

### **💻 System Information**
- "check disk space" → `df -h`
- "show memory usage" → `free -h`
- "system information" → `uname -a`
- "current time" → `date`

### **⚙️ Process Management**
- "running processes" → `ps aux`
- "process tree" → `pstree`
- "top processes" → `top -n 1`

### **🌐 Network Information**
- "network connections" → `netstat -tuln`
- "network interfaces" → `ip addr`
- "ping test" → `ping -c 4 hostname`

### **🔍 File Content & Search**
- "search in files" → `grep "pattern" filename`
- "file contents" → `cat filename`
- "file permissions" → `ls -la filename`

## ⚙️ **Setup Requirements**

### **1. MCP Server Configuration**
✅ Your MCP server must be running and accessible
✅ Configure server details in application UI (**MCP Settings**)
✅ Set one server as default for your user

### **2. Python Environment** 
✅ Virtual environment configured in `conf/config.ini`:
```ini
[python]
interpreter = python\.venv\Scripts\python.exe
```

### **3. Test the Integration**
```bash
# Run the test script
node test_shell_command_integration.js
```

## 🔧 **How to Use**

### **Step 1: Configure Your MCP Server**
1. Go to **MCP Settings** in your application
2. Add your MCP server details (host, port)
3. Test the connection
4. Set as default server

### **Step 2: Start Chatting**
Simply ask the AI for system operations using natural language:

- *"List the files in this directory"*
- *"Check how much disk space is left"*
- *"Show me the running processes"*
- *"What's the current system load?"*

### **Step 3: The AI Will:**
1. **Recognize** your request as a shell command task
2. **Generate** appropriate Linux command
3. **Execute** via your MCP orchestrator
4. **Interpret** results for you
5. **Remember** context for follow-up questions

## 🛡️ **Security Features**

### **✅ Safe by Design**
- Commands run on **MCP server** (not application server)
- **User authentication** required for all operations
- **Server isolation** - users can only access their configured servers
- **Timeout protection** prevents hanging processes

### **✅ Smart Command Generation**
- AI **avoids destructive commands** unless explicitly confirmed
- **Suggests safer alternatives** when appropriate
- **Explains commands** before execution
- **Linux-focused** commands for better compatibility

## 🎭 **AI Behavior Patterns**

### **The AI Will:**
- **Explain what it's doing** before running commands
- **Interpret results** in plain language
- **Suggest follow-up actions** when relevant
- **Ask for confirmation** for potentially destructive operations
- **Remember previous commands** for context

### **Example Interaction Flow:**
```
User: "I want to see the disk usage"

AI: "I'll check the disk usage for you using the df command 
     which shows filesystem disk space usage."

[Executes: df -h]

AI: "Here's your disk usage:
     - Root filesystem: 45% used (23GB of 50GB)
     - /home partition: 67% used (134GB of 200GB)
     - You have good space available on root but /home 
       is getting full. Would you like me to help you 
       find large files?"
```

## 🔄 **Context Preservation**

The AI maintains context across the conversation:

```
User: "List files in current directory"
AI: [executes ls -la, shows results]

User: "Show me details about the largest file"  
AI: [remembers previous output, uses ls -lh on the largest file]

User: "Delete that file"
AI: "Are you sure you want to delete [filename]? This cannot be undone."
```

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. "No MCP server configured"**
**Solution:** Configure MCP server in application settings

#### **2. "Connection failed"**
**Solution:** Check MCP server is running and accessible

#### **3. "Python interpreter not found"**
**Solution:** Verify Python path in `conf/config.ini`

#### **4. "Command timeout"**
**Solution:** Command took too long, check server performance

### **Debug Steps**
```bash
# 1. Test MCP server directly
curl http://your-mcp-server:8080/info

# 2. Test Python environment
python\.venv\Scripts\python.exe --version

# 3. Test orchestrator
python python/terminal-mcp-orchestrator/orchestrator.py --server http://host:port --list

# 4. Check application logs
tail -f logs/app.log
```

## 📊 **Monitoring & Logs**

Monitor AI tool usage through:
- **Application logs** - See command executions
- **Chat history** - Review command results
- **MCP server logs** - Monitor server-side execution

## 🎯 **Best Practices**

### **For Users:**
- Use **natural language** - don't worry about exact command syntax
- Be **specific** in your requests for better results
- **Ask follow-up questions** - the AI remembers context
- **Confirm destructive operations** when prompted

### **For Administrators:**
- **Monitor command usage** through logs
- **Set appropriate MCP server permissions**
- **Consider command whitelisting** for production environments
- **Keep MCP servers updated** and secure

## 🆘 **Support**

If you encounter issues:

1. **Check MCP server status** and connectivity
2. **Verify Python environment** setup
3. **Review application logs** for errors
4. **Test with simple commands** first (like `ls` or `pwd`)
5. **Check user MCP server configuration**

## 🎉 **You're Ready!**

Your AI assistant now has powerful shell command capabilities! Try asking it to:

- "*List files in current directory*"
- "*Check disk usage*"
- "*Show running processes*"
- "*What's the system uptime?*"

The AI will handle the technical details while providing you with clear, interpreted results. Enjoy your enhanced AI assistant! 🚀 
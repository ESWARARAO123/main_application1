const express = require('express');
const session = require('express-session');
const cookieParser = require('cookie-parser');
const ini = require('ini');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const http = require('http');
const { setupWebSocketServer } = require('./websocket/server');
const { registerMCPHandlers } = require('./routes/websocket');
const { initializeDatabase, runModelIdMigration } = require('./database');
const { router: authRoutes } = require('./routes/auth');
const userRoutes = require('./routes/users');
const chatbotRoutes = require('./routes/chatbot');
const runsRoutes = require('./routes/runs');
const settingsRoutes = require('./routes/settings');
const dashboardRoutes = require('./routes/dashboard');
const configRoutes = require('./routes/config');
const ollamaRoutes = require('./routes/ollama');
const mcpRoutes = require('./routes/mcp');
const mcpPagesRoutes = require('./routes/mcp-pages');
const websocketRoutes = require('./routes/websocket').router;
const aiRoutes = require('./routes/ai');
const aiRulesRoutes = require('./routes/aiRules');
const aiContextRoutes = require('./routes/aiContext');
const contextAgentRoutes = require('./routes/contextAgent');
const chat2sqlRoutes = require('./routes/chat2sql');
const { setSessionStore } = require('./services/sessionService');

// Try to require the documents routes, but don't fail if they're not available
let documentsRoutes;
let documentsStatusRoutes;
try {
  documentsRoutes = require('./routes/documents');
} catch (error) {
  console.error('Documents routes not available:', error.message);
  // Create a dummy router that returns 503 for all routes
  documentsRoutes = express.Router();
  documentsRoutes.all('*', (req, res) => {
    res.status(503).json({
      error: 'Document service unavailable. Please install required dependencies: npm install multer uuid'
    });
  });
}

// Try to require the documents-status routes
try {
  documentsStatusRoutes = require('./routes/documents-status');
} catch (error) {
  console.error('Documents status routes not available:', error.message);
  // Create a dummy router that returns 503 for all routes
  documentsStatusRoutes = express.Router();
  documentsStatusRoutes.all('*', (req, res) => {
    res.status(503).json({
      error: 'Document status service unavailable.'
    });
  });
}
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

// Parse command line arguments
const argv = yargs(hideBin(process.argv))
  .option('config', {
    alias: 'c',
    type: 'string',
    description: 'Path to config file',
    default: './conf/config.ini'
  })
  .help()
  .argv;

// Read configuration
const configPath = path.resolve(argv.config);
console.log(`Loading configuration from: ${configPath}`);
const config = ini.parse(fs.readFileSync(configPath, 'utf-8'));

// Initialize Express app
const app = express();

// Make app globally available for services to access
global.app = app;

// Initialize the database and start the server when ready
async function startServer() {
  try {
    // Initialize database - now async
    await initializeDatabase(config);
    console.log('Database initialization completed');

    // Run model_id migration to ensure compatibility
    await runModelIdMigration();

    // Configure CORS based on config
    const corsOptions = {
      origin: config.security.allow_embedding ? true : config.frontend.app_sub_url || true,
      credentials: true
    };
    app.use(cors(corsOptions));

    // Basic middleware
    app.use(cookieParser());
    app.use(express.json());

    // Session configuration
    const sessionOptions = {
      secret: config.security.secret_key,
      resave: false,
      saveUninitialized: false,
      cookie: {
        secure: config.security.cookie_secure === 'true',
        httpOnly: true,
        sameSite: config.security.cookie_samesite,
        maxAge: parseInt(config.security.cookie_max_age)
      },
      rolling: true // Resets the cookie expiration on every response
    };

    // Create the session middleware with a specific store
    // Use MemoryStore for simplicity, but in production you'd use a more robust store
    const MemoryStore = session.MemoryStore;
    const sessionStore = new MemoryStore();
    sessionOptions.store = sessionStore;

    const sessionMiddleware = session(sessionOptions);

    // Store a reference to the session store for WebSocket authentication
    console.log('Setting session store reference for WebSocket authentication');
    setSessionStore(sessionStore);

    // Use the session middleware
    app.use(sessionMiddleware);

    // API Routes - Now all prefixed with /api
    const apiRouter = express.Router();
    apiRouter.use('/auth', authRoutes);
    apiRouter.use('/users', userRoutes);
    apiRouter.use('/chatbot', chatbotRoutes);
    apiRouter.use('/runs', runsRoutes);
    apiRouter.use('/settings', settingsRoutes);
    apiRouter.use('/dashboard', dashboardRoutes);
    apiRouter.use('/ollama', ollamaRoutes(config));
    apiRouter.use('/mcp', mcpRoutes);
    apiRouter.use('/documents', documentsRoutes);
    apiRouter.use('/documents-status', documentsStatusRoutes);
    apiRouter.use('/websocket', websocketRoutes);
    apiRouter.use('/ai', aiRoutes);
    apiRouter.use('/ai-rules', aiRulesRoutes);
    apiRouter.use('/ai-context', aiContextRoutes);
    apiRouter.use('/context-agent', contextAgentRoutes);
    apiRouter.use('/chat2sql', chat2sqlRoutes);
    apiRouter.use('/', configRoutes(config)); // Add config routes at the API root

    // Mount all API routes under /api
    app.use('/api', apiRouter);

    // MCP pages routes
    app.use('/settings/mcp', mcpPagesRoutes);

    // Serve static files from client/build
    const staticPath = config.server.static_root_path || path.join(__dirname, '../client/build');
    console.log(`Serving static files from: ${staticPath}`);

    if (fs.existsSync(staticPath)) {
      // Serve static files
      app.use(express.static(staticPath));

      // For any request that doesn't match an API route or static file,
      // send the React app's index.html (for client-side routing)
      app.get('*', (req, res) => {
        // Use path.resolve to create an absolute path to index.html
        const indexPath = path.resolve(staticPath, 'index.html');
        res.sendFile(indexPath);
      });
    } else {
      console.warn(`Static file path does not exist: ${staticPath}`);
    }

    // Error handling middleware
    app.use((err, req, res, next) => {
      console.error(err.stack);
      res.status(500).json({ error: 'Internal Server Error' });
    });

    // Create HTTP server from Express app
    const port = config.server.port || 5634;
    const host = config.server.domain || 'localhost';

    // Create HTTP server
    const server = http.createServer(app);

    // Setup WebSocket server
    const wsServer = setupWebSocketServer(server);

    // Register MCP WebSocket handlers
    registerMCPHandlers(wsServer);
    console.log('Registered MCP WebSocket handlers');

    // Make WebSocket server available to routes and services
    app.set('wsServer', wsServer);

    // Start HTTP server
    server.listen(port, host, () => {
      console.log(`Server running on ${config.server.protocol}://${host}:${port}`);
      console.log(`WebSocket server running on ${config.server.protocol === 'https' ? 'wss' : 'ws'}://${host}:${port}/ws`);
    }).on('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        console.error(`Port ${port} is already in use. Please use a different port.`);
      } else {
        console.error('Error starting server:', err);
      }
      process.exit(1);
    });

    // Graceful shutdown
    process.on('SIGTERM', () => {
      console.log('Received SIGTERM. Performing graceful shutdown...');
      server.close(() => {
        console.log('Server closed. Exiting process.');
        process.exit(0);
      });
    });

  } catch (error) {
    console.error('Error starting the application:', error);
    process.exit(1);
  }
}

// Start the server
startServer();
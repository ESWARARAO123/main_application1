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
const { getWebSocketService } = require('./services/webSocketService');

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
      origin: config.security?.allow_embedding ? true : (config.frontend && config.frontend.app_sub_url) || true,
      credentials: true
    };
    app.use(cors(corsOptions));

    // Basic middleware
    app.use(cookieParser());
    app.use(express.json());

    // Session configuration
    const sessionOptions = {
      secret: config.security?.secret_key || 'your_session_secret_here',
      resave: false,
      saveUninitialized: false,
      cookie: {
        secure: config.security?.cookie_secure === 'true' || false, // Default to false for development
        httpOnly: true,
        sameSite: config.security?.cookie_samesite || 'lax',
        maxAge: parseInt(config.security?.cookie_max_age) || 86400000
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
    const staticPath = path.join(__dirname, '../client/build');
    console.log(`Serving static files from: ${staticPath}`);

    if (fs.existsSync(staticPath)) {
      // Serve static files
      app.use(express.static(staticPath));

      // For any request that doesn't match an API route or static file,
      // send the React app's index.html (for client-side routing)
      // But exclude API routes from this catch-all
      app.get('*', (req, res, next) => {
        // Skip API routes - let them be handled by the API router
        if (req.path.startsWith('/api/')) {
          return next();
        }

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
    const port = config.server?.port || 5642;
    const host = config.server?.domain || '0.0.0.0';

    // Create HTTP server
    const server = http.createServer(app);

    // Setup WebSocket server
    const wsServer = setupWebSocketServer(server);

    // Initialize WebSocket service
    const webSocketService = getWebSocketService(wsServer);
    console.log('WebSocket service initialized with server instance');

    // Initialize Document Workers for scalable processing
    try {
      const DocumentWorker = require('./workers/documentWorker');
      const workerCount = parseInt(config.document_queue?.worker_count || '3');
      
      console.log(`Starting ${workerCount} document workers...`);
      
      // Start multiple workers for parallel processing
      const workers = [];
      for (let i = 0; i < workerCount; i++) {
        const worker = new DocumentWorker();
        await worker.initialize();
        workers.push(worker);
        console.log(`Document worker ${i + 1} started successfully`);
      }
      
      // Store workers for graceful shutdown
      app.set('documentWorkers', workers);
      console.log(`All ${workerCount} document workers started successfully`);
      
    } catch (error) {
      console.error('Failed to start document workers:', error);
      console.warn('Document processing will not be available');
    }

    // Register MCP WebSocket handlers
    registerMCPHandlers(wsServer);
    console.log('Registered MCP WebSocket handlers');

    // Make WebSocket server available to routes and services
    app.set('wsServer', wsServer);
    app.set('webSocketService', webSocketService);

    // Start HTTP server
    server.listen(port, host, () => {
      console.log(`Server running on http://${host}:${port}`);
      console.log(`WebSocket server running on ws://${host}:${port}/ws`);
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
      
      // Shutdown document workers
      const workers = app.get('documentWorkers');
      if (workers && workers.length > 0) {
        console.log('Shutting down document workers...');
        Promise.all(workers.map(worker => worker.shutdown()))
          .then(() => {
            console.log('All document workers shut down successfully');
          })
          .catch(err => {
            console.error('Error shutting down workers:', err);
          });
      }
      
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
const express = require('express');
const router = express.Router();

/**
 * Configuration API routes
 * Provides frontend configuration for the client application
 */
module.exports = function(config) {
  // Return frontend configuration to the client
  router.get('/frontend-config', (req, res) => {
    // Send only configuration that should be available to the frontend
    res.json({
      title: config.frontend.app_title || 'Product Demo',
      appName: config.frontend.app_name || 'Product Demo',
      apiUrl: config.frontend.api_url || '/api',
      defaultTheme: config.frontend.default_theme || 'light'
    });
  });
  
  return router;
}; 
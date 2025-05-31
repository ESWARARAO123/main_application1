const express = require('express');
const path = require('path');
const router = express.Router();
const { requireAuth } = require('./auth');

// MCP SSH Setup page
router.get('/ssh-setup', requireAuth, (req, res) => {
  // Return the SPA for handling on frontend
  res.sendFile(path.resolve(__dirname, '../../client/build/index.html'));
});

// MCP Manual Install page
router.get('/manual-install', requireAuth, (req, res) => {
  // Return the SPA for handling on frontend
  res.sendFile(path.resolve(__dirname, '../../client/build/index.html'));
});

// MCP Installation page
router.get('/install-mcp', requireAuth, (req, res) => {
  // Return the SPA for handling on frontend
  res.sendFile(path.resolve(__dirname, '../../client/build/index.html'));
});

module.exports = router; 
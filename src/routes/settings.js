const express = require('express');
const fs = require('fs');
const path = require('path');
const ini = require('ini');
const { db } = require('../database');
const router = express.Router();

// Middleware to check if user is authenticated
const isAuthenticated = (req, res, next) => {
  if (req.session.userId) {
    next();
  } else {
    res.status(401).json({ error: 'Unauthorized' });
  }
};

// Get config path
const configPath = process.env.CONFIG_PATH || path.resolve('./conf/config.ini');

// Function to read config.ini
function readConfig() {
  try {
    const configContent = fs.readFileSync(configPath, 'utf-8');
    return ini.parse(configContent);
  } catch (error) {
    console.error('Error reading config.ini:', error);
    return null;
  }
}

// Function to write to config.ini
function writeConfig(config) {
  try {
    const configString = ini.stringify(config);
    fs.writeFileSync(configPath, configString);
    return true;
  } catch (error) {
    console.error('Error writing to config.ini:', error);
    return false;
  }
}

// Update theme in config.ini
router.post('/theme', isAuthenticated, async (req, res) => {
  const { theme } = req.body;
  
  if (!theme) {
    return res.status(400).json({ error: 'Theme is required' });
  }
  
  try {
    const config = readConfig();
    
    if (!config) {
      return res.status(500).json({ error: 'Could not read configuration file' });
    }
    
    // Ensure frontend section exists
    if (!config.frontend) {
      config.frontend = {};
    }
    
    // Update theme setting
    config.frontend.theme = theme;
    
    // Write back to config.ini
    const success = writeConfig(config);
    
    if (!success) {
      return res.status(500).json({ error: 'Failed to save theme to configuration' });
    }
    
    // Also store user preference in database
    try {
      // Check if user_settings table exists
      const tableExists = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").get();
      
      if (!tableExists) {
        // Create user_settings table
        db.exec(`
          CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            theme TEXT,
            api_key TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
          )
        `);
      }
      
      // Update or insert user theme preference
      const existingSettings = db.prepare('SELECT user_id FROM user_settings WHERE user_id = ?').get(req.session.userId);
      
      if (existingSettings) {
        db.prepare('UPDATE user_settings SET theme = ? WHERE user_id = ?').run(theme, req.session.userId);
      } else {
        db.prepare('INSERT INTO user_settings (user_id, theme) VALUES (?, ?)').run(req.session.userId, theme);
      }
    } catch (dbError) {
      console.error('Database error when saving theme:', dbError);
      // Continue even if DB update fails
    }
    
    res.status(200).json({ message: 'Theme updated successfully' });
  } catch (error) {
    console.error('Error updating theme:', error);
    res.status(500).json({ error: 'Failed to update theme setting' });
  }
});

// Get current theme setting
router.get('/theme', isAuthenticated, async (req, res) => {
  try {
    const config = readConfig();
    
    if (!config || !config.frontend) {
      return res.status(200).json({ theme: 'dark' }); // Default theme
    }
    
    // Return theme from config.ini or default to dark
    res.status(200).json({ theme: config.frontend.theme || 'dark' });
  } catch (error) {
    console.error('Error getting theme:', error);
    res.status(500).json({ error: 'Failed to get theme setting' });
  }
});

// Save API key
router.post('/api-key', isAuthenticated, (req, res) => {
  const { apiKey } = req.body;
  
  if (!apiKey) {
    return res.status(400).json({ error: 'API key is required' });
  }
  
  try {
    // Check if user_settings table exists
    const tableExists = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").get();
    
    if (!tableExists) {
      // Create user_settings table
      db.exec(`
        CREATE TABLE IF NOT EXISTS user_settings (
          user_id INTEGER PRIMARY KEY,
          theme TEXT,
          api_key TEXT,
          FOREIGN KEY (user_id) REFERENCES users (id)
        )
      `);
    }
    
    // Update or insert user API key
    const existingSettings = db.prepare('SELECT user_id FROM user_settings WHERE user_id = ?').get(req.session.userId);
    
    if (existingSettings) {
      db.prepare('UPDATE user_settings SET api_key = ? WHERE user_id = ?').run(apiKey, req.session.userId);
    } else {
      db.prepare('INSERT INTO user_settings (user_id, api_key) VALUES (?, ?)').run(req.session.userId, apiKey);
    }
    
    res.json({ message: 'API key saved successfully' });
  } catch (error) {
    console.error('Error saving API key:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get API key
router.get('/api-key', isAuthenticated, (req, res) => {
  try {
    const settings = db.prepare('SELECT api_key FROM user_settings WHERE user_id = ?').get(req.session.userId);
    res.json({ apiKey: settings?.api_key || '' });
  } catch (error) {
    console.error('Error getting API key:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router; 
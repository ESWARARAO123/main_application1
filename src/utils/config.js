/**
 * Config Utility
 * Provides access to application configuration
 */

const ini = require('ini');
const fs = require('fs');
const path = require('path');
const { logger } = require('./logger');

// Load the configuration file
const configPath = process.env.CONFIG_PATH || path.resolve('./conf/config.ini');
let config = {};

try {
  logger.info(`Loading configuration from: ${configPath}`);
  config = ini.parse(fs.readFileSync(configPath, 'utf-8'));
  logger.debug('Configuration loaded successfully');
} catch (err) {
  logger.error(`Failed to load configuration: ${err.message}`);
  throw err;
}

/**
 * Get a configuration value by key
 * 
 * @param {string} key - Dot-notation key (e.g., 'server.port')
 * @param {any} defaultValue - Default value if key not found
 * @returns {any} - The configuration value or default value
 */
function get(key, defaultValue = null) {
  if (!key) {
    return defaultValue;
  }
  
  const parts = key.split('.');
  let value = config;
  
  for (const part of parts) {
    if (value === undefined || value === null || typeof value !== 'object') {
      return defaultValue;
    }
    
    value = value[part];
  }
  
  return value !== undefined ? value : defaultValue;
}

/**
 * Get all configuration values for a section
 * 
 * @param {string} section - Section name
 * @returns {Object} - Section configuration
 */
function getSection(section) {
  return config[section] || {};
}

/**
 * Get all configuration
 * 
 * @returns {Object} - Full configuration
 */
function getAll() {
  return { ...config };
}

/**
 * Check if a configuration key exists
 * 
 * @param {string} key - Dot-notation key
 * @returns {boolean} - Whether the key exists
 */
function has(key) {
  if (!key) {
    return false;
  }
  
  const parts = key.split('.');
  let value = config;
  
  for (const part of parts) {
    if (value === undefined || value === null || typeof value !== 'object') {
      return false;
    }
    
    value = value[part];
  }
  
  return value !== undefined;
}

module.exports = {
  get,
  getSection,
  getAll,
  has
}; 
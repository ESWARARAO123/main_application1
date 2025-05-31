const path = require('path');
const fs = require('fs');
const ini = require('ini');

/**
 * Load paths from config.ini and resolve them relative to the project root
 * @param {string} configPath - Path to the config.ini file
 * @returns {Object} - Object containing resolved paths
 */
function loadPathsFromConfig(configPath = './conf/config.ini') {
  try {
    // Resolve the config path relative to the project root
    const resolvedConfigPath = path.resolve(process.cwd(), configPath);
    
    // Check if the config file exists
    if (!fs.existsSync(resolvedConfigPath)) {
      console.error(`Config file not found at ${resolvedConfigPath}`);
      return getDefaultPaths();
    }
    
    // Read and parse the config file
    const config = ini.parse(fs.readFileSync(resolvedConfigPath, 'utf-8'));
    
    // Get paths from config or use defaults
    const pathsConfig = config.paths || {};
    
    // Resolve paths relative to project root
    const projectRoot = process.cwd();
    
    return {
      documentsDir: path.resolve(projectRoot, pathsConfig.documents || './DATA/documents'),
      embeddingsDir: path.resolve(projectRoot, pathsConfig.embeddings || './DATA/embeddings'),
      chromaDataDir: path.resolve(projectRoot, pathsConfig.chroma_data || './DATA/chroma_data'),
      vectorStoreDir: path.resolve(projectRoot, pathsConfig.vector_store || './DATA/vector_store')
    };
  } catch (error) {
    console.error('Error loading paths from config:', error);
    return getDefaultPaths();
  }
}

/**
 * Get default paths relative to the project root
 * @returns {Object} - Object containing default paths
 */
function getDefaultPaths() {
  const projectRoot = process.cwd();
  return {
    documentsDir: path.resolve(projectRoot, './DATA/documents'),
    embeddingsDir: path.resolve(projectRoot, './DATA/embeddings'),
    chromaDataDir: path.resolve(projectRoot, './DATA/chroma_data'),
    vectorStoreDir: path.resolve(projectRoot, './DATA/vector_store')
  };
}

/**
 * Ensure all required directories exist
 * @param {Object} paths - Object containing paths to ensure
 */
function ensureDirectoriesExist(paths) {
  Object.values(paths).forEach(dirPath => {
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
      console.log(`Created directory: ${dirPath}`);
    }
  });
}

// Export functions
module.exports = {
  loadPathsFromConfig,
  getDefaultPaths,
  ensureDirectoriesExist
};

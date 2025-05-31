/**
 * Script to reset the vector store by deleting all collections and data
 * Run with: node scripts/reset-vector-store.js
 */

const { ChromaClient } = require('chromadb');
const path = require('path');
const fs = require('fs');
const ini = require('ini');

// Load configuration
const configPath = path.join(__dirname, '../conf/config.ini');
const config = ini.parse(fs.readFileSync(configPath, 'utf-8'));

// ChromaDB connection settings
const chromaHost = config.chromadb?.host || 'localhost';
const chromaPort = config.chromadb?.port || 8000;
const chromaUrl = `http://${chromaHost}:${chromaPort}`;

async function resetVectorStore() {
  try {
    console.log(`Connecting to ChromaDB at ${chromaUrl}...`);
    const chromaClient = new ChromaClient({ path: chromaUrl });

    // List all collections
    const collections = await chromaClient.listCollections();
    console.log(`Found ${collections.length} collections in ChromaDB`);

    // Delete each collection
    for (const collection of collections) {
      console.log(`Deleting collection: ${collection.name}`);
      await chromaClient.deleteCollection({ name: collection.name });
    }

    console.log('All collections deleted successfully');

    // Also delete the vector store directory if it exists
    const vectorStoreDir = path.join(__dirname, '../DATA/vector_store');
    if (fs.existsSync(vectorStoreDir)) {
      console.log(`Deleting vector store directory: ${vectorStoreDir}`);
      fs.rmdirSync(vectorStoreDir, { recursive: true });
      console.log('Vector store directory deleted successfully');
    }

    console.log('Vector store reset complete!');
  } catch (error) {
    console.error('Error resetting vector store:', error);
  }
}

// Run the reset function
resetVectorStore();

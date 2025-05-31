/**
 * Script to reset everything - vector store, dashboard metrics, and chat sessions
 * Run with: node scripts/reset-all.js
 */

const path = require('path');
const fs = require('fs');
const ini = require('ini');
const { Pool } = require('pg');
const { ChromaClient } = require('chromadb');

// Load configuration
const configPath = path.join(__dirname, '../conf/config.ini');
const config = ini.parse(fs.readFileSync(configPath, 'utf-8'));

// Database connection settings
const dbConfig = {
  host: config.database.host,
  port: config.database.port,
  database: config.database.name,
  user: config.database.user,
  password: config.database.password,
  ssl: config.database.ssl === 'true'
};

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
      try {
        await chromaClient.deleteCollection({ name: collection.name });
        console.log(`Successfully deleted collection: ${collection.name}`);
      } catch (error) {
        console.error(`Error deleting collection ${collection.name}:`, error.message);
      }
    }

    console.log('All collections deleted successfully');

    // Also delete the vector store directory if it exists
    const vectorStoreDir = path.join(__dirname, '../DATA/vector_store');
    if (fs.existsSync(vectorStoreDir)) {
      console.log(`Deleting vector store directory: ${vectorStoreDir}`);
      fs.rmdirSync(vectorStoreDir, { recursive: true });
      console.log('Vector store directory deleted successfully');
    }

    return true;
  } catch (error) {
    console.error('Error resetting vector store:', error);
    return false;
  }
}

async function resetDashboardMetrics(pool) {
  try {
    console.log('Resetting dashboard metrics...');
    await pool.query(`
      UPDATE dashboard_metrics
      SET
        total_messages = 0,
        recent_messages = 0,
        total_documents = 0,
        avg_response_time = 0
      WHERE id = 1
    `);

    console.log('Dashboard metrics reset successfully!');
    return true;
  } catch (error) {
    console.error('Error resetting dashboard metrics:', error);
    return false;
  }
}

async function resetChatSessions(pool) {
  try {
    console.log('Resetting chat sessions and messages...');

    // Delete all chat messages
    await pool.query('DELETE FROM chat_messages');
    console.log('All chat messages deleted');

    // Delete all chat sessions
    await pool.query('DELETE FROM chat_sessions');
    console.log('All chat sessions deleted');

    return true;
  } catch (error) {
    console.error('Error resetting chat sessions:', error);
    return false;
  }
}

async function resetDocuments(pool) {
  try {
    console.log('Resetting documents...');

    // Delete all documents
    await pool.query('DELETE FROM documents');
    console.log('All documents deleted from database');

    // Delete the documents directory
    const documentsDir = path.join(__dirname, '../DATA/documents');
    if (fs.existsSync(documentsDir)) {
      // We'll just log this but not actually delete it to be safe
      console.log(`Note: You may want to manually delete the documents directory: ${documentsDir}`);
    }

    return true;
  } catch (error) {
    console.error('Error resetting documents:', error);
    return false;
  }
}

async function resetAll() {
  const pool = new Pool(dbConfig);

  try {
    console.log('Starting complete reset of the system...');

    // Reset vector store
    const vectorStoreReset = await resetVectorStore();

    // Reset database tables
    const metricsReset = await resetDashboardMetrics(pool);
    const sessionsReset = await resetChatSessions(pool);
    const documentsReset = await resetDocuments(pool);

    console.log('\nReset Summary:');
    console.log(`- Vector Store: ${vectorStoreReset ? 'SUCCESS' : 'FAILED'}`);
    console.log(`- Dashboard Metrics: ${metricsReset ? 'SUCCESS' : 'FAILED'}`);
    console.log(`- Chat Sessions: ${sessionsReset ? 'SUCCESS' : 'FAILED'}`);
    console.log(`- Documents: ${documentsReset ? 'SUCCESS' : 'FAILED'}`);

    console.log('\nSystem reset complete!');
    console.log('Please restart the server to apply all changes.');
  } catch (error) {
    console.error('Error during reset process:', error);
  } finally {
    await pool.end();
  }
}

// Run the reset function
resetAll();

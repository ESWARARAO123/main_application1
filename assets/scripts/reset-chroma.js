/**
 * Script to reset the ChromaDB collection
 * Run with: node scripts/reset-chroma.js
 */

const { ChromaClient } = require('chromadb');

async function resetChroma() {
  try {
    console.log('Connecting to ChromaDB at http://localhost:8000...');
    const chromaClient = new ChromaClient({ path: 'http://localhost:8000' });
    
    // Create a new collection (this will replace any existing one with the same name)
    console.log('Recreating the rag_docs collection...');
    
    try {
      // Try to delete the collection first
      await chromaClient.deleteCollection({ name: 'rag_docs' });
      console.log('Deleted existing rag_docs collection');
    } catch (error) {
      console.log('No existing collection to delete or error deleting:', error.message);
    }
    
    // Create a new collection
    const collection = await chromaClient.createCollection({ name: 'rag_docs' });
    console.log('Created new rag_docs collection');
    
    console.log('ChromaDB reset complete!');
  } catch (error) {
    console.error('Error resetting ChromaDB:', error);
  }
}

// Run the reset function
resetChroma();

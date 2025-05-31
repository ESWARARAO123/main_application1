/**
 * Test script for RAG functionality
 * This script tests the RAG service by initializing it and performing a simple query
 */

const ragService = require('./src/services/ragService');

async function testRag() {
  try {
    console.log('Testing RAG functionality...');
    
    // Wait a moment for services to initialize
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Check if RAG is available
    const ragAvailable = await ragService.isRagAvailable();
    console.log(`RAG available: ${ragAvailable}`);
    
    if (ragAvailable) {
      // Test a simple query
      const query = 'What is Python?';
      console.log(`Testing query: "${query}"`);
      
      const result = await ragService.retrieveContext(query);
      
      if (result.success) {
        console.log('RAG retrieval successful!');
        console.log(`Found ${result.sources.length} relevant sources`);
        console.log('First source:', result.sources[0]);
      } else {
        console.error('RAG retrieval failed:', result.error);
      }
    } else {
      console.log('RAG is not available. Make sure you have processed some documents first.');
    }
    
    console.log('Test completed.');
  } catch (error) {
    console.error('Error testing RAG:', error);
  }
}

// Run the test
testRag();

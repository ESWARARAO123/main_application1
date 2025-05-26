/**
 * Test script for header-based chunking
 * Run with: node test_header_chunking.js
 */

const { chunkBySection } = require('../../../src/utils/headerChunker');

// Sample document with section headers
const sampleDocument = `Technical Whitepaper: Advanced Processing System

1 Introduction

This document describes the design and implementation of our Advanced Processing System. 
The system is designed to handle large volumes of data efficiently.

1.1 Purpose

The purpose of this document is to provide a technical overview of the system architecture
and components. It serves as a reference for developers and system architects.

1.2 Scope

This document covers the core processing engine, data flow mechanisms, and integration points.
It does not cover deployment considerations or operational procedures.

2 System Architecture

The system is built on a modular architecture that allows for flexible scaling and extension.

2.1 High-Level Components

The system consists of the following high-level components:
- Data Ingestion Layer
- Processing Engine
- Storage Subsystem
- API Gateway

2.1.1 Data Ingestion Layer

The data ingestion layer is responsible for accepting and normalizing input from various sources.
It supports multiple protocols including REST, GraphQL, and message queues.

2.1.2 Processing Engine

The processing engine is the core of the system. It implements the business logic and data transformation rules.
The engine is designed for parallel execution and can scale horizontally.

2.2 Data Flow

Data flows through the system in the following sequence:
1. Input data is received by the ingestion layer
2. Data is validated and normalized
3. Processing engine applies business rules and transformations
4. Results are stored in the storage subsystem
5. Notifications are sent to subscribers via the API gateway

3 Performance Considerations

The system is designed to handle high throughput with minimal latency.

3.1 Scalability

The system can scale horizontally by adding more processing nodes. Each component is designed
to work in a distributed environment and maintains its own scaling characteristics.

3.2 Optimization Techniques

Several optimization techniques are employed:
- Data partitioning
- Caching of frequently accessed data
- Asynchronous processing
- Batch operations for high-volume data

4 Security

Security is a fundamental aspect of the system design.

4.1 Authentication and Authorization

The system implements industry-standard authentication protocols and role-based access control.
All API calls require valid authentication tokens.

4.2 Data Protection

Data is protected both at rest and in transit:
- All network communications use TLS 1.3
- Sensitive data is encrypted using AES-256
- Access logs are maintained for all data operations

5 Conclusion

The Advanced Processing System provides a robust platform for high-volume data processing with
a focus on performance, scalability, and security.`;

// Test the header-based chunking
async function testHeaderChunking() {
  console.log('Testing header-based chunking...\n');
  
  // Use default chunking parameters
  const chunks = chunkBySection(sampleDocument);
  
  console.log(`\nTotal chunks created: ${chunks.length}\n`);
  
  // Display chunk information
  console.log('Chunk summary:');
  chunks.forEach((chunk, index) => {
    console.log(`- Chunk ${index + 1}: "${chunk.sectionTitle}" (${chunk.text.length} chars)`);
    console.log(`  First few words: "${chunk.text.substring(0, 50)}..."`);
    console.log('');
  });
}

testHeaderChunking().catch(console.error); 
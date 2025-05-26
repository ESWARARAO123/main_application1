/**
 * Test script for PDF table extraction
 * Run with: node test_table_extraction.js sample.pdf
 */

const { extractTextAndTablesSmart } = require('../../../src/services/pdfProcessor');
const fs = require('fs');
const path = require('path');

async function testTableExtraction() {
  // Get file path from command line argument
  const pdfPath = process.argv[2];
  
  if (!pdfPath) {
    console.error('Error: Please provide a PDF file path.');
    console.error('Usage: node test_table_extraction.js <path_to_pdf>');
    process.exit(1);
  }
  
  // Check if file exists
  if (!fs.existsSync(pdfPath)) {
    console.error(`Error: File not found: ${pdfPath}`);
    process.exit(1);
  }
  
  console.log(`Testing table extraction with file: ${pdfPath}`);
  
  try {
    // Extract text and tables from the PDF
    const result = await extractTextAndTablesSmart(pdfPath);
    
    if (!result.success) {
      console.error(`Error during extraction: ${result.error}`);
      process.exit(1);
    }
    
    // Output basic stats
    console.log(`\nExtraction successful!`);
    console.log(`Pages: ${result.pageCount}`);
    console.log(`Character count: ${result.text.length}`);
    console.log(`Contains tables: ${result.hasTables ? 'Yes' : 'No'}`);
    
    // Count table occurrences
    const tableMarkerCount = (result.text.match(/### Extracted Table/g) || []).length;
    console.log(`Table count: ${tableMarkerCount}`);
    
    // Save the extracted text to a file
    const outputPath = path.join(path.dirname(pdfPath), `${path.basename(pdfPath, '.pdf')}_extracted.md`);
    fs.writeFileSync(outputPath, result.text);
    console.log(`\nExtracted text saved to: ${outputPath}`);
    
    // Print a sample of the extracted text
    console.log('\nSample of extracted text:');
    console.log('--------------------------------');
    
    // Find a table section in the text if possible
    const tableIndex = result.text.indexOf('### Extracted Table');
    
    if (tableIndex !== -1) {
      // Print some text before and after a table
      const startIndex = Math.max(0, tableIndex - 200);
      const endIndex = Math.min(result.text.length, tableIndex + 800);
      console.log(result.text.substring(startIndex, endIndex));
    } else {
      // Just print the beginning of the text
      console.log(result.text.substring(0, 1000) + '...');
    }
    
    console.log('--------------------------------');
    
  } catch (error) {
    console.error('Error during test:', error);
    process.exit(1);
  }
}

testTableExtraction(); 
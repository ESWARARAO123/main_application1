/**
 * PDF Processor Service
 * Handles advanced PDF processing including text and table extraction
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const ini = require('ini');

/**
 * Get Python interpreter path from config.ini
 * @returns {Object} Configuration object with Python interpreter path
 */
function getPythonConfig() {
  try {
    const configPath = path.resolve(process.cwd(), './conf/config.ini');
    if (fs.existsSync(configPath)) {
      const config = ini.parse(fs.readFileSync(configPath, 'utf-8'));
      return {
        pythonInterpreter: config.python?.interpreter || './python/venv/bin/python',
      };
    }
    return { pythonInterpreter: './python/venv/bin/python' };
  } catch (error) {
    console.warn('Error reading Python config:', error);
    return { pythonInterpreter: './python/venv/bin/python' };
  }
}

/**
 * Extract text and tables from a PDF file using Python pdfplumber
 * @param {string} filePath - Path to the PDF file
 * @returns {Promise<Object>} - Result with extracted text and table data, or error
 */
async function extractTextAndTablesSmart(filePath) {
  return new Promise((resolve) => {
    try {
      // Path to the Python script for table extraction (updated to RAG-MODULE folder)
      const pythonScriptPath = path.resolve(process.cwd(), 'python/RAG-MODULE/extract_text_with_tables.py');
      
      // Check if the Python script exists
      if (!fs.existsSync(pythonScriptPath)) {
        console.warn(`Python table extraction script not found at ${pythonScriptPath}`);
        return resolve({
          success: false,
          error: `Python table extraction script not found at ${pythonScriptPath}`
        });
      }
      
      // Check if the PDF file exists and is accessible
      if (!fs.existsSync(filePath)) {
        console.warn(`PDF file not found at ${filePath}`);
        return resolve({
          success: false,
          error: `PDF file not found at ${filePath}`
        });
      }
      
      // Get Python interpreter from config
      const { pythonInterpreter } = getPythonConfig();
      
      console.log(`Running table extraction with Python interpreter ${pythonInterpreter}`);
      console.log(`Script path: ${pythonScriptPath}`);
      console.log(`PDF path: ${filePath}`);
      
      // Execute the Python script as a child process
      const pythonProcess = spawn(pythonInterpreter, [pythonScriptPath, filePath]);
      
      let stdout = '';
      let stderr = '';
      
      // Collect stdout data
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      // Collect stderr data
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
        console.warn(`Python stderr: ${data.toString()}`);
      });
      
      // Handle process completion
      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.warn(`Python process exited with code ${code}`);
          console.warn(`stderr: ${stderr}`);
          
          // Check if the error indicates missing dependencies
          if (stderr.includes('ModuleNotFoundError: No module named') || 
              stderr.includes('ImportError:')) {
            console.warn('Python module dependency missing. You may need to install required packages.');
            console.warn('Try: pip install --user pdfplumber');
          }
          
          return resolve({
            success: false,
            error: `Python extraction failed with code ${code}: ${stderr}`
          });
        }
        
        try {
          // Try to parse JSON output
          let jsonOutput = stdout.trim();
          
          // Handle case where there might be non-JSON content in stdout
          const jsonStart = jsonOutput.indexOf('{');
          if (jsonStart > 0) {
            console.warn(`Non-JSON prefix in output: ${jsonOutput.substring(0, jsonStart)}`);
            jsonOutput = jsonOutput.substring(jsonStart);
          }
          
          const result = JSON.parse(jsonOutput);
          
          if (!result.success) {
            console.warn(`Python extraction reported failure: ${result.error}`);
            
            // Check for pdfplumber installation issue
            if (result.error && result.error.includes('pdfplumber module not installed')) {
              console.warn('pdfplumber module not installed in Python environment.');
              console.warn('Please install it using: pip install --user pdfplumber');
            }
            
            return resolve({
              success: false,
              error: result.error || 'Unknown error in Python extraction'
            });
          }
          
          console.log(`Successfully extracted ${result.page_count} pages with tables from PDF`);
          
          return resolve({
            success: true,
            text: result.text,
            pageCount: result.page_count,
            hasTables: result.has_tables || false
          });
        } catch (parseError) {
          console.error('Error parsing Python script output:', parseError);
          console.error('Raw output:', stdout);
          
          return resolve({
            success: false,
            error: `Error parsing Python output: ${parseError.message}`,
            rawOutput: stdout.substring(0, 500) // Include part of the raw output for debugging
          });
        }
      });
      
      // Handle process errors
      pythonProcess.on('error', (error) => {
        console.error('Error executing Python script:', error);
        
        // Check for specific errors
        if (error.code === 'ENOENT') {
          console.error(`Python interpreter '${pythonInterpreter}' not found.`);
          console.error('Please check your config.ini file and ensure the python.interpreter path is correct.');
        }
        
        return resolve({
          success: false,
          error: `Failed to execute Python script: ${error.message} (${error.code})`,
          hint: error.code === 'ENOENT' ? 'Python interpreter not found. Check config.ini.' : null
        });
      });
      
      // Set a timeout for the Python process (60 seconds - tables can take longer)
      const timeout = setTimeout(() => {
        console.warn('Python process is taking too long, killing it...');
        pythonProcess.kill();
        
        return resolve({
          success: false,
          error: 'Python extraction timed out after 60 seconds'
        });
      }, 60000);
      
      // Clear the timeout when the process completes
      pythonProcess.on('close', () => {
        clearTimeout(timeout);
      });
    } catch (error) {
      console.error('Error in extractTextAndTablesSmart:', error);
      return resolve({
        success: false,
        error: `Exception in extractTextAndTablesSmart: ${error.message}`
      });
    }
  });
}

module.exports = {
  extractTextAndTablesSmart
}; 
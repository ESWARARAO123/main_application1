// Simple script to run the application in development mode
const { spawn } = require('child_process');
const path = require('path');

// Get the config path
const configPath = path.resolve('../../conf/config.ini');
console.log(`Using config from: ${configPath}`);

// Start the server with the config path
const server = spawn('node', ['../server.js', '--config', configPath], {
  stdio: 'inherit'
});

server.on('close', (code) => {
  console.log(`Server process exited with code ${code}`);
}); 
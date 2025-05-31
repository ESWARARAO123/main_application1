// This file serves as an entry point for React Scripts
// It simply forwards to the main.tsx file which is the actual application entry point
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// Removed StrictMode to prevent double mounting of components
// This helps prevent duplicate WebSocket connections
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
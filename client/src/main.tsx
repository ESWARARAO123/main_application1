import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// Removed StrictMode to prevent double mounting of components
// This helps prevent duplicate WebSocket connections
ReactDOM.createRoot(document.getElementById('root')!).render(
  <App />
)
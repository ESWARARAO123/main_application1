import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ChakraProvider } from '@chakra-ui/react';
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { SidebarProvider } from './contexts/SidebarContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { MCPProvider } from './contexts/MCPContext';
import { MCPAgentProvider } from './contexts/MCPAgentContext';
import { ContextAgentProvider } from './contexts/ContextAgentContext';
import PrivateRoute from './components/PrivateRoute';
import AdminRoute from './components/AdminRoute';
import Layout from './components/Layout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import UserManagement from './pages/UserManagement';
import Chatbot from './pages/Chatbot';
import RunStatus from './pages/RunStatus';
import Settings from './pages/Settings';
import MCPSSHSetup from './pages/MCPSSHSetup';
import MCPManualInstall from './pages/MCPManualInstall';
import MCPInstall from './pages/MCPInstall';
// WebSocketDebug page removed - now integrated in Settings
import LoadingScreen from './components/LoadingScreen';
import chakraTheme from './theme/chakraTheme';

// Import global CSS for theme variables
import './styles/global.css';

function App() {
  return (
    <AuthProvider>
      <ThemeProvider>
        <WebSocketProvider>
          <MCPProvider>
            <MCPAgentProvider>
              <ChakraProvider theme={chakraTheme}>
                <SidebarProvider>
                <Router>
                  <Routes>
                    {/* Public routes */}
                    <Route path="/login" element={<Login />} />

                  {/* Protected routes wrapped in Layout */}
                  <Route element={<PrivateRoute><Layout /></PrivateRoute>}>
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/chatbot" element={<Chatbot />} />
                    <Route path="/runs" element={<RunStatus />} />
                    <Route path="/runs-original" element={<RunStatus />} />
                    <Route path="/settings" element={<Settings />} />
                    <Route path="/settings/mcp/ssh-setup" element={<MCPSSHSetup />} />
                    <Route path="/settings/mcp/manual-install" element={<MCPManualInstall />} />
                    <Route path="/settings/mcp/install-mcp" element={<MCPInstall />} />
                    {/* WebSocket debug page removed - now integrated in Settings */}
                    <Route
                      path="/users"
                      element={
                        <AdminRoute>
                          <UserManagement />
                        </AdminRoute>
                      }
                    />
                  </Route>

                  {/* Root and catch-all routes */}
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
                </Router>
                </SidebarProvider>
              </ChakraProvider>
            </MCPAgentProvider>
          </MCPProvider>
        </WebSocketProvider>
      </ThemeProvider>
    </AuthProvider>
  );
}

export default App;
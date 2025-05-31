import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import axios from 'axios';
import { motion } from 'framer-motion';
import {
  UserIcon,
  PaintBrushIcon,
  ServerIcon,
  CpuChipIcon,
  BugAntIcon,
  ChatBubbleLeftRightIcon
} from '@heroicons/react/24/outline';
import OllamaSettings from '../components/settings/OllamaSettings';
import MCPSettings from '../components/settings/MCPSettings';
import DebuggingSettings from '../components/settings/DebuggingSettings';
import AIRulesSettings from '../components/settings/AIRulesSettings';

// Import ThemeType from the ThemeContext
type ThemeType = 'dark' | 'light' | 'midnight';

export default function Settings() {
  const { user, refreshUser } = useAuth();
  const { currentTheme, setTheme } = useTheme();
  const [activeTab, setActiveTab] = useState(() => {
    // Try to restore the last active tab from localStorage
    const savedTab = localStorage.getItem('settings_active_tab');
    return savedTab || 'profile';
  });
  const [isLoading, setIsLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  // Save active tab to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('settings_active_tab', activeTab);
  }, [activeTab]);

  // Define available themes
  const themes = [
    { id: 'dark' as ThemeType, name: 'Dark Theme', description: 'Dark blue interface for low-light environments' },
    { id: 'light' as ThemeType, name: 'Light Theme', description: 'Bright and clean interface for daytime use' },
    { id: 'midnight' as ThemeType, name: 'Midnight Theme', description: 'Deep black theme with purple accents' }
  ];

  // Profile form state
  const [profileForm, setProfileForm] = useState({
    username: user?.username || '',
    name: user?.name || '',
    email: user?.email || ''
  });



  // Load user data
  useEffect(() => {
    if (user) {
      setProfileForm(prevForm => ({
        username: user.username || prevForm.username || '',
        name: user.name || prevForm.name || '',
        email: user.email || prevForm.email || ''
      }));
    }
  }, [user]);

  // Handle profile form changes
  const handleProfileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setProfileForm({
      ...profileForm,
      [e.target.name]: e.target.value
    });
  };

  // Handle theme selection
  const handleThemeChange = (themeId: ThemeType) => {
    try {
      setTheme(themeId);
      showSuccess('Theme updated successfully');
    } catch (error) {
      console.error('Error changing theme:', error);
      showError('Failed to save theme preferences');
    }
  };

  // Handle profile save
  const handleProfileSave = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!profileForm.username || !profileForm.email) {
      return showError('Username and email are required');
    }

    setIsLoading(true);
    try {
      // Update user profile
      await axios.put('/api/users/profile', {
        username: profileForm.username,
        name: profileForm.name,
        email: profileForm.email
      });

      await refreshUser();
      showSuccess('Profile updated successfully');
    } catch (error: any) {
      console.error('Error updating profile:', error);
      showError(error.response?.data?.error || 'Failed to update profile');
    } finally {
      setIsLoading(false);
    }
  };



  // Show success message
  const showSuccess = (message: string) => {
    setSuccessMessage(message);
    setErrorMessage('');
    setTimeout(() => setSuccessMessage(''), 3000);
  };

  // Show error message
  const showError = (message: string) => {
    setErrorMessage(message);
    setSuccessMessage('');
    setTimeout(() => setErrorMessage(''), 5000);
  };

  // Settings tabs data structure
  const tabs = [
    { id: 'profile', name: 'Profile', icon: UserIcon },
    { id: 'appearance', name: 'Appearance', icon: PaintBrushIcon },
    { id: 'ai_rules', name: 'AI Rules', icon: ChatBubbleLeftRightIcon },
    { id: 'ollama', name: 'LLM Integration', icon: ServerIcon },
    { id: 'mcp', name: 'MCP Integration', icon: CpuChipIcon },
    { id: 'debugging', name: 'Debugging', icon: BugAntIcon },
  ];

  return (
    <div 
      className="min-h-screen p-4 md:p-6 relative overflow-hidden"
      style={{
        background: 'linear-gradient(to-b, var(--color-bg), var(--color-bg-dark))',
      }}
    >
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-10" style={{
        backgroundImage: 'radial-gradient(circle at 2px 2px, var(--color-primary) 1px, transparent 0)',
        backgroundSize: '40px 40px',
      }} />

      <div className="max-w-7xl mx-auto relative z-10 space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="p-6 rounded-xl backdrop-blur-sm"
          style={{
            background: 'linear-gradient(to-r, var(--color-surface), var(--color-surface-dark))',
            border: '1px solid var(--color-border)',
            boxShadow: '0 10px 25px -5px var(--color-shadow), 0 8px 10px -6px var(--color-shadow-light)',
            transform: 'translateZ(0)',
            backfaceVisibility: 'hidden',
          }}
        >
          <h1 className="text-2xl md:text-3xl font-bold mb-2" style={{ color: 'var(--color-text)' }}>Settings</h1>
          <p style={{ color: 'var(--color-text-secondary)' }}>
            Configure your account and application preferences
          </p>
        </motion.div>

      {/* Success/Error messages */}
      {successMessage && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 rounded-xl backdrop-blur-sm"
          style={{
            background: 'var(--color-success)10',
            border: '1px solid var(--color-success)',
            color: 'var(--color-success)',
            boxShadow: '0 4px 6px -1px var(--color-shadow-light)',
          }}
        >
          <div className="flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            {successMessage}
          </div>
        </motion.div>
      )}
      {errorMessage && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 rounded-xl backdrop-blur-sm"
          style={{
            background: 'var(--color-error)10',
            border: '1px solid var(--color-error)',
            color: 'var(--color-error)',
            boxShadow: '0 4px 6px -1px var(--color-shadow-light)',
          }}
        >
          <div className="flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            {errorMessage}
          </div>
        </motion.div>
      )}

      <div className="flex flex-col md:flex-row gap-6">
        {/* Settings tabs */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="w-full md:w-64 flex-shrink-0"
        >
          <div className="rounded-xl overflow-hidden sticky md:top-4 shadow-card border" style={{
            backgroundColor: 'var(--color-surface)',
            borderColor: 'var(--color-border)',
            boxShadow: '0 10px 15px -3px var(--color-shadow), 0 4px 6px -2px var(--color-shadow-light)',
            transform: 'translateZ(0)',
            backfaceVisibility: 'hidden',
          }}>
            <div className="bg-gradient-to-r from-[var(--color-primary-dark)] to-[var(--color-primary)] px-4 py-3">
              <h3 className="text-lg font-semibold text-white">Settings Menu</h3>
            </div>
            <ul>
              {tabs.map((tab) => (
                <li key={tab.id}>
                  <motion.button
                    whileHover={{ 
                      backgroundColor: activeTab === tab.id ? 'var(--color-surface-light)' : 'var(--color-surface-dark)',
                      x: 3
                    }}
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full flex items-center px-4 py-3 transition-all ${
                      activeTab === tab.id ? 'font-medium' : ''
                    }`}
                    style={{
                      color: activeTab === tab.id ? 'var(--color-primary)' : 'var(--color-text)',
                      backgroundColor: activeTab === tab.id ? 'var(--color-surface-light)' : 'transparent',
                      borderLeft: activeTab === tab.id ? `3px solid var(--color-primary)` : '3px solid transparent'
                    }}
                  >
                    <tab.icon className="w-5 h-5 mr-3" />
                    <span>{tab.name}</span>
                  </motion.button>
                </li>
              ))}
            </ul>
          </div>
        </motion.div>

        {/* Settings content */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="flex-1 rounded-xl p-5 overflow-auto shadow-card border" 
          style={{
            backgroundColor: 'var(--color-surface)',
            borderColor: 'var(--color-border)',
            maxHeight: 'calc(100vh - 150px)',
            boxShadow: '0 10px 15px -3px var(--color-shadow), 0 4px 6px -2px var(--color-shadow-light)',
            transform: 'translateZ(0)',
            backfaceVisibility: 'hidden',
          }}
        >
          {activeTab === 'profile' && (
            <div>
              <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--color-text)' }}>Profile Settings</h2>
              <div className="space-y-6">
                <div className="flex flex-col md:flex-row md:items-center gap-4 pb-6" style={{
                  borderBottom: `1px solid var(--color-border)`
                }}>
                  <div className="w-20 h-20 rounded-full flex items-center justify-center text-3xl font-bold"
                    style={{
                      backgroundColor: `var(--color-primary-light)20`,
                      color: 'var(--color-primary)'
                    }}
                  >
                    {user?.username?.charAt(0).toUpperCase() || 'U'}
                  </div>
                  <div>
                    <p style={{ color: 'var(--color-text)' }}>{user?.username || 'User'}</p>
                    <p style={{ color: 'var(--color-text-secondary)' }} className="text-sm">
                      Role: {user?.role === 'admin' ? 'Administrator' : 'User'}
                    </p>
                  </div>
                </div>

                <form onSubmit={handleProfileSave} className="space-y-4">
                  <div>
                    <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>Username</label>
                    <input
                      type="text"
                      name="username"
                      value={profileForm.username}
                      onChange={handleProfileChange}
                      className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                      style={{
                        backgroundColor: 'var(--color-surface-dark)',
                        borderColor: 'var(--color-border)',
                        color: 'var(--color-text)'
                      }}
                    />
                  </div>

                  <div>
                    <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>Full Name</label>
                    <input
                      type="text"
                      name="name"
                      value={profileForm.name}
                      onChange={handleProfileChange}
                      className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                      style={{
                        backgroundColor: 'var(--color-surface-dark)',
                        borderColor: 'var(--color-border)',
                        color: 'var(--color-text)'
                      }}
                    />
                  </div>

                  <div>
                    <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>Email</label>
                    <input
                      type="email"
                      name="email"
                      value={profileForm.email}
                      onChange={handleProfileChange}
                      className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                      style={{
                        backgroundColor: 'var(--color-surface-dark)',
                        borderColor: 'var(--color-border)',
                        color: 'var(--color-text)'
                      }}
                    />
                  </div>

                  <motion.button
                    type="submit"
                    disabled={isLoading}
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    className="px-4 py-2 rounded-lg mt-4 transition-all"
                    style={{
                      background: 'linear-gradient(to-r, var(--color-primary), var(--color-primary-dark))',
                      color: 'white',
                      opacity: isLoading ? 0.7 : 1,
                      boxShadow: '0 4px 6px -1px var(--color-shadow), 0 2px 4px -1px var(--color-shadow-light)',
                      border: '1px solid rgba(255, 255, 255, 0.1)'
                    }}
                  >
                    {isLoading ? (
                      <div className="flex items-center">
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Saving...
                      </div>
                    ) : 'Save Profile'}
                  </motion.button>
                </form>
              </div>
            </div>
          )}

          {activeTab === 'appearance' && (
            <div>
              <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--color-text)' }}>Appearance Settings</h2>
              <p className="mb-4" style={{ color: 'var(--color-text-secondary)' }}>Choose your preferred theme for the application.</p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
                {themes.map((theme) => (
                  <motion.div
                    key={theme.id}
                    onClick={() => handleThemeChange(theme.id)}
                    whileHover={{ 
                      y: -8,
                      boxShadow: '0 20px 25px -5px var(--color-shadow), 0 10px 10px -5px var(--color-shadow-light)'
                    }}
                    initial={{ y: 0 }}
                    animate={{ 
                      y: currentTheme === theme.id ? -5 : 0,
                      boxShadow: currentTheme === theme.id 
                        ? '0 15px 25px -5px var(--color-shadow), 0 10px 10px -5px var(--color-shadow-light)'
                        : '0 4px 6px -1px var(--color-shadow), 0 2px 4px -1px var(--color-shadow-light)'
                    }}
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                    className={`relative rounded-xl p-5 cursor-pointer`}
                    style={{
                      backgroundColor: 'var(--color-surface-dark)',
                      borderWidth: '2px',
                      borderStyle: 'solid',
                      borderColor: currentTheme === theme.id ? 'var(--color-primary)' : 'var(--color-border)',
                      boxShadow: '0 4px 6px -1px var(--color-shadow), 0 2px 4px -1px var(--color-shadow-light)',
                      transform: 'translateZ(0)',
                      backfaceVisibility: 'hidden'
                    }}
                  >
                    <div className="flex items-center mb-4">
                      <div
                        className="w-12 h-12 rounded-full flex items-center justify-center mr-3 shadow-inner"
                        style={{
                          backgroundColor: theme.id === 'dark' ? '#1a1f2d' :
                                           theme.id === 'light' ? '#ffffff' :
                                           '#111827',
                          border: '1px solid var(--color-border)',
                          boxShadow: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.2)'
                        }}
                      >
                        <motion.div
                          animate={{ 
                            scale: [1, 1.2, 1],
                            rotate: currentTheme === theme.id ? [0, 10, -10, 0] : 0
                          }}
                          transition={{ 
                            duration: 0.5,
                            repeat: currentTheme === theme.id ? 0 : 0,
                            repeatType: "reverse"
                          }}
                          className="w-6 h-6 rounded-full shadow-lg"
                          style={{
                            backgroundColor: theme.id === 'dark' ? '#3b82f6' :
                                             theme.id === 'light' ? '#3b82f6' :
                                             '#8b5cf6',
                            boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.3)'
                          }}
                        ></motion.div>
                      </div>
                      <div>
                        <h3 className="font-medium text-lg" style={{ color: 'var(--color-text)' }}>{theme.name}</h3>
                        {currentTheme === theme.id && (
                          <span className="text-xs font-medium px-2 py-0.5 rounded-full" style={{ 
                            backgroundColor: 'var(--color-primary)20',
                            color: 'var(--color-primary)'
                          }}>
                            Active
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>{theme.description}</p>

                    {currentTheme === theme.id && (
                      <div
                        className="absolute top-2 right-2 w-4 h-4 rounded-full flex items-center justify-center"
                        style={{ backgroundColor: 'var(--color-primary)' }}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 text-white" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'ai_rules' && (
            <div>
              <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--color-text)' }}>AI Rules Settings</h2>
              <p className="mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                Define rules and preferences for the AI to follow during your conversations.
              </p>
              <AIRulesSettings />
            </div>
          )}

          {activeTab === 'ollama' && (
            <div>
              <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--color-text)' }}>LLM Integration Settings</h2>
              <OllamaSettings isAdmin={user?.role === 'admin'} />
            </div>
          )}

          {activeTab === 'mcp' && (
            <div>
              <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--color-text)' }}>
                MCP Integration Settings
              </h2>
              <MCPSettings isAdmin={user?.role === 'admin'} />
            </div>
          )}

          {activeTab === 'debugging' && (
            <div>
              <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--color-text)' }}>
                Debugging Tools
              </h2>
              <p className="mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                These tools are for debugging purposes only. Enable them only when needed as they may affect performance.
              </p>
              <DebuggingSettings />
            </div>
          )}
        </motion.div>
      </div>
      </div>
    </div>
  );
}
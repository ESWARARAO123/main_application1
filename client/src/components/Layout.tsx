import React, { useState, useEffect } from 'react';
import { Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { useSidebar } from '../contexts/SidebarContext';
import {
  ArrowRightOnRectangleIcon,
  Bars3Icon,
  BellIcon,
  MoonIcon,
  SunIcon,
} from '@heroicons/react/24/outline';
import Sidebar from './Sidebar';

export default function Layout() {
  const { user, isAdmin, logout } = useAuth();
  const { currentTheme, setTheme } = useTheme();
  const { isExpanded: isSidebarOpen, toggleSidebar: toggleSidebarContext } = useSidebar();
  const navigate = useNavigate();

  const [notifications, setNotifications] = useState([
    { id: 1, text: 'Welcome to the Pinnacleflow Ai!', read: false },
    { id: 2, text: 'New features available', read: false },
  ]);
  const [showNotifications, setShowNotifications] = useState(false);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        // We're using the shared context now, so no direct setting
        if (isSidebarOpen) {
          toggleSidebarContext();
        }
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isSidebarOpen, toggleSidebarContext]);

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const toggleTheme = () => {
    // Cycle between themes: dark -> light -> midnight -> dark
    const nextTheme = currentTheme === 'dark'
      ? 'light'
      : currentTheme === 'light'
        ? 'midnight'
        : 'dark';

    setTheme(nextTheme);
  };

  const markAllAsRead = () => {
    setNotifications(notifications.map(n => ({ ...n, read: true })));
  };

  // Get appropriate theme icon
  const ThemeIcon = currentTheme === 'dark' || currentTheme === 'midnight'
    ? SunIcon
    : MoonIcon;

  return (
    <div className="flex h-screen" style={{ backgroundColor: 'var(--color-bg)' }}>
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => toggleSidebarContext()}
        onToggle={toggleSidebarContext}
        user={user}
        isAdmin={isAdmin}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="h-16 border-b" style={{
          backgroundColor: 'var(--color-surface)',
          borderColor: 'var(--color-border)'
        }}>
          <div className="h-full px-6 flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {/* Only show menu button on mobile */}
              <button
                type="button"
                className="lg:hidden p-2 rounded focus:outline-none transition-colors"
                onClick={toggleSidebarContext}
                aria-label="Toggle sidebar"
                style={{
                  color: 'var(--color-text-secondary)'
                }}
              >
                <Bars3Icon className="w-6 h-6" />
              </button>
              <span className="font-medium hidden md:inline" style={{ color: 'var(--color-text)' }}>
                Welcome, <span style={{ color: 'var(--color-primary)' }}>{user?.username || 'User'}</span>
              </span>
            </div>

            <div className="flex items-center space-x-3">
              {/* Notifications */}
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setShowNotifications(!showNotifications)}
                  className="p-2 rounded-full transition-colors"
                  style={{
                    color: 'var(--color-text-secondary)',
                    backgroundColor: 'transparent'
                  }}
                >
                  <BellIcon className="w-5 h-5" />
                  {notifications.some(n => !n.read) && (
                    <span className="absolute top-1 right-1 w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--color-primary)' }}></span>
                  )}
                </button>

                {showNotifications && (
                  <div className="absolute right-0 mt-2 w-80 rounded-lg shadow-lg z-50"
                    style={{
                      backgroundColor: 'var(--color-surface)',
                      borderColor: 'var(--color-border)',
                      border: '1px solid var(--color-border)'
                    }}
                  >
                    <div className="p-3 flex justify-between items-center" style={{
                      borderBottom: '1px solid var(--color-border)'
                    }}>
                      <h3 className="font-medium" style={{ color: 'var(--color-text)' }}>Notifications</h3>
                      <button
                        type="button"
                        onClick={markAllAsRead}
                        className="text-xs"
                        style={{ color: 'var(--color-primary)' }}
                      >
                        Mark all as read
                      </button>
                    </div>
                    <div className="max-h-80 overflow-y-auto">
                      {notifications.length > 0 ? (
                        notifications.map(notification => (
                          <div
                            key={notification.id}
                            className={`p-3 transition-colors ${notification.read ? 'opacity-60' : ''}`}
                            style={{
                              borderBottom: '1px solid var(--color-border)'
                            }}
                          >
                            <p className="text-sm" style={{ color: 'var(--color-text)' }}>{notification.text}</p>
                            <p className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>Just now</p>
                          </div>
                        ))
                      ) : (
                        <div className="p-4 text-center" style={{ color: 'var(--color-text-muted)' }}>
                          No notifications
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <button
                type="button"
                onClick={handleLogout}
                className="flex items-center px-4 py-2 rounded-lg transition-all duration-200"
                style={{
                  color: 'var(--color-text)',
                  backgroundColor: 'transparent'
                }}
              >
                <ArrowRightOnRectangleIcon className="w-5 h-5 mr-2" />
                <span className="hidden sm:inline">Logout</span>
              </button>
            </div>
          </div>
        </header>
        <main className="flex-1 overflow-auto p-6" style={{ backgroundColor: 'var(--color-bg)' }}>
          <div className="max-w-6xl mx-auto px-4">
            <div className="rounded-lg" style={{
              backgroundColor: 'var(--color-surface)',
              border: '1px solid var(--color-border)'
            }}>
              <Outlet />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
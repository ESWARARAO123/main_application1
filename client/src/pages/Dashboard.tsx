import React, { useState, useEffect, Suspense } from 'react';
import { useAuth } from 'contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { api } from 'services/api';
import MetricCard from 'components/MetricCard';
import QuickActionCard from 'components/QuickActionCard';

interface DashboardMetrics {
  userStats: {
    totalUsers: number;
    adminUsers: number;
    regularUsers: number;
    recentUsers: number;
  };
  messageStats: {
    totalMessages: number;
    recentMessages: number;
    avgResponseTime: number;
    totalDocuments: number;
  };
  licenseUsage?: {
    totalLicenses: number;
    activeLicenses: number;
    expirationDate: string;
    daysRemaining: number;
  };
}

// Components are imported at the top of the file

export default function Dashboard() {
  const { user, isAdmin } = useAuth();
  // Theme context is used for styling via CSS variables
  const navigate = useNavigate();
  // Time period for metrics, currently fixed to 'week'
  const [selectedPeriod] = useState('week');
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isAdmin()) {
      navigate('/chatbot');
    }
  }, [isAdmin, navigate]);

  useEffect(() => {
    if (!isAdmin()) return;

    const fetchMetrics = async () => {
      setLoading(true);
      try {
        const response = await api.get('/dashboard/metrics');
        const metricsData = response.data || {};

        const normalizedData: DashboardMetrics = {
          userStats: {
            totalUsers: Number(metricsData.userStats?.totalUsers ?? 0),
            adminUsers: Number(metricsData.userStats?.adminUsers ?? 0),
            regularUsers: Number(metricsData.userStats?.regularUsers ?? 0),
            recentUsers: Number(metricsData.userStats?.recentUsers ?? 0)
          },
          messageStats: {
            totalMessages: Number(metricsData.messageStats?.totalMessages ?? 0),
            recentMessages: Number(metricsData.messageStats?.recentMessages ?? 0),
            avgResponseTime: Number(metricsData.messageStats?.avgResponseTime ?? 0),
            totalDocuments: Number(metricsData.messageStats?.totalDocuments ?? 0)
          },
          licenseUsage: metricsData.licenseUsage || {
            totalLicenses: 25,
            activeLicenses: 12,
            expirationDate: "2025-12-31",
            daysRemaining: 612
          }
        };

        setMetrics(normalizedData);
        setError(null);
      } catch (err) {
        setError('Failed to load dashboard metrics');
        setMetrics({
          userStats: {
            totalUsers: 0,
            adminUsers: 0,
            regularUsers: 0,
            recentUsers: 0
          },
          messageStats: {
            totalMessages: 0,
            recentMessages: 0,
            avgResponseTime: 0,
            totalDocuments: 0
          },
          licenseUsage: {
            totalLicenses: 25,
            activeLicenses: 0,
            expirationDate: "N/A",
            daysRemaining: 0
          }
        });
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, [isAdmin, selectedPeriod]);

  if (!isAdmin()) {
    return null;
  }

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
        {/* Welcome Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="p-6 rounded-xl backdrop-blur-sm"
          style={{
            background: 'linear-gradient(to-r, var(--color-surface), var(--color-surface-dark))',
            border: '1px solid var(--color-border)',
          }}
        >
          <h2 className="text-2xl md:text-3xl font-bold" style={{ color: 'var(--color-text)' }}>
            Welcome back, <span style={{ color: 'var(--color-primary)' }}>{user?.name || user?.username}</span>
          </h2>
          <p className="mt-2 text-sm md:text-base" style={{ color: 'var(--color-text-secondary)' }}>
            Admin Dashboard - System Overview
          </p>
        </motion.div>

        {/* Loading State */}
        <Suspense fallback={<div>Loading components...</div>}>
          {loading ? (
            <motion.div
              className="flex justify-center items-center py-12"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                className="w-12 h-12 border-4 border-[var(--color-primary)] border-t-transparent rounded-full"
              />
              <p className="ml-4" style={{ color: 'var(--color-text)' }}>Loading dashboard metrics...</p>
            </motion.div>
          ) : error ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-6 rounded-xl backdrop-blur-sm text-center"
              style={{
                background: 'var(--color-error)10',
                border: '1px solid var(--color-error)',
                color: 'var(--color-error)',
              }}
            >
              <div className="flex items-center justify-center gap-2">
                <span className="text-lg">⚠️</span>
                <p>{error}</p>
              </div>
              <button
                className="mt-4 px-4 py-2 rounded-lg transition-all hover:scale-105"
                style={{
                  background: 'linear-gradient(to-r, var(--color-primary), var(--color-primary-dark))',
                  color: 'white',
                }}
                onClick={() => window.location.reload()}
              >
                Retry
              </button>
            </motion.div>
          ) : (
            <>
              {/* Metrics Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <MetricCard
                  title="User Statistics"
                  primaryMetric={metrics?.userStats.totalUsers ?? 0}
                  primaryLabel="Total Users"
                  details={[
                    { label: 'Admin Users', value: metrics?.userStats.adminUsers ?? 0 },
                    { label: 'Regular Users', value: metrics?.userStats.regularUsers ?? 0 },
                    { label: 'New Users (7d)', value: metrics?.userStats.recentUsers ?? 0, highlight: true },
                  ]}
                  link={{ to: '/users', label: 'Manage Users' }}
                />

                <MetricCard
                  title="Message Statistics"
                  primaryMetric={metrics?.messageStats.totalMessages ?? 0}
                  primaryLabel="Total Messages"
                  details={[
                    { label: 'Recent Messages', value: metrics?.messageStats.recentMessages ?? 0 },
                    { label: 'Total Documents', value: metrics?.messageStats.totalDocuments ?? 0 },
                  ]}
                  link={{ to: '/chatbot', label: 'Go to Chatbot' }}
                />

                <MetricCard
                  title="License Usage"
                  primaryMetric={`${metrics?.licenseUsage?.activeLicenses ?? 0}/${metrics?.licenseUsage?.totalLicenses ?? 0}`}
                  primaryLabel="Active/Total"
                  details={[
                    { label: 'Expiration', value: metrics?.licenseUsage?.expirationDate ?? 'N/A' },
                    { label: 'Days Left', value: metrics?.licenseUsage?.daysRemaining ?? 0 },
                  ]}
                  progress={Math.round((metrics?.licenseUsage?.activeLicenses ?? 0) / (metrics?.licenseUsage?.totalLicenses ?? 1) * 100)}
                  link={{ to: '/settings', label: 'Manage Licenses' }}
                  badge="NEW"
                />
              </div>

              {/* Quick Actions & Management */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                  className="lg:col-span-2 p-0 rounded-xl overflow-hidden h-full"
                  style={{
                    background: 'var(--color-surface)',
                    boxShadow: '0 10px 15px -3px var(--color-shadow), 0 4px 6px -2px var(--color-shadow-light)',
                    border: '1px solid var(--color-border)',
                    display: 'flex',
                    flexDirection: 'column',
                  }}
                >
                  <div className="bg-gradient-to-r from-[var(--color-primary-dark)] to-[var(--color-primary)] px-6 py-4">
                    <div className="flex justify-between items-center">
                      <h3 className="text-lg font-semibold text-white flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3zM6 8a2 2 0 11-4 0 2 2 0 014 0zM16 18v-3a5.972 5.972 0 00-.75-2.906A3.005 3.005 0 0119 15v3h-3zM4.75 12.094A5.973 5.973 0 004 15v3H1v-3a3 3 0 013.75-2.906z" />
                        </svg>
                        User Management Tools
                      </h3>
                      <Link
                        to="/users"
                        className="px-4 py-2 rounded-lg transition-all hover:scale-105 flex items-center"
                        style={{
                          background: 'var(--color-primary)',
                          color: 'white',
                          boxShadow: '0 2px 4px var(--color-shadow)',
                          border: '1px solid rgba(255, 255, 255, 0.2)',
                        }}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clipRule="evenodd" />
                        </svg>
                        Add User
                      </Link>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6 flex-grow">
                    <div 
                      className="rounded-xl p-5 transition-shadow h-full flex flex-col"
                      style={{
                        background: 'var(--color-surface)',
                        boxShadow: '0 4px 6px -1px var(--color-shadow-light)',
                        border: '1px solid var(--color-border)',
                      }}
                    >
                      <div className="flex items-center mb-4">
                        <div className="w-10 h-10 rounded-full flex items-center justify-center mr-3" style={{ backgroundColor: 'var(--color-primary)20' }}>
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" style={{ color: 'var(--color-primary)' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                          </svg>
                        </div>
                        <h4 className="text-lg font-semibold" style={{ color: 'var(--color-text)' }}>User Roles</h4>
                      </div>
                      <ul className="space-y-3 flex-grow">
                        <li className="flex items-center justify-between p-3 rounded-lg" style={{ backgroundColor: 'var(--color-surface-light)' }}>
                          <div className="flex items-center">
                            <div className="w-8 h-8 rounded-full flex items-center justify-center mr-3" style={{ backgroundColor: 'var(--color-primary)20' }}>
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" style={{ color: 'var(--color-primary)' }} viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-6-3a2 2 0 11-4 0 2 2 0 014 0zm-2 4a5 5 0 00-4.546 2.916A5.986 5.986 0 005 10a6 6 0 0012 0c0-.35-.035-.691-.1-1.021A5 5 0 0010 11z" clipRule="evenodd" />
                              </svg>
                            </div>
                            <span className="font-medium" style={{ color: 'var(--color-text)' }}>Admin Users</span>
                          </div>
                          <span className="px-3 py-1 rounded-full text-sm font-semibold" style={{
                            backgroundColor: 'var(--color-primary)20',
                            color: 'var(--color-primary)',
                          }}>
                            {metrics?.userStats.adminUsers ?? 0}
                          </span>
                        </li>
                        <li className="flex items-center justify-between p-3 rounded-lg" style={{ backgroundColor: 'var(--color-surface-light)' }}>
                          <div className="flex items-center">
                            <div className="w-8 h-8 rounded-full flex items-center justify-center mr-3" style={{ backgroundColor: 'var(--color-success)20' }}>
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" style={{ color: 'var(--color-success)' }} viewBox="0 0 20 20" fill="currentColor">
                                <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z" />
                              </svg>
                            </div>
                            <span className="font-medium" style={{ color: 'var(--color-text)' }}>Regular Users</span>
                          </div>
                          <span className="px-3 py-1 rounded-full text-sm font-semibold" style={{
                            backgroundColor: 'var(--color-success)20',
                            color: 'var(--color-success)',
                          }}>
                            {metrics?.userStats.regularUsers ?? 0}
                          </span>
                        </li>
                      </ul>
                    </div>

                    <div 
                      className="rounded-xl p-5 transition-shadow h-full flex flex-col"
                      style={{
                        background: 'var(--color-surface)',
                        boxShadow: '0 4px 6px -1px var(--color-shadow-light)',
                        border: '1px solid var(--color-border)',
                      }}
                    >
                      <div className="flex items-center mb-4">
                        <div className="w-10 h-10 rounded-full flex items-center justify-center mr-3" style={{ backgroundColor: 'var(--color-primary)20' }}>
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" style={{ color: 'var(--color-primary)' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                        <h4 className="text-lg font-semibold" style={{ color: 'var(--color-text)' }}>Recent Activity</h4>
                      </div>
                      <ul className="space-y-3 flex-grow">
                        <li className="flex items-center justify-between p-3 rounded-lg" style={{ backgroundColor: 'var(--color-surface-light)' }}>
                          <div className="flex items-center">
                            <div className="w-8 h-8 rounded-full flex items-center justify-center mr-3" style={{ backgroundColor: 'var(--color-success)20' }}>
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" style={{ color: 'var(--color-success)' }} viewBox="0 0 20 20" fill="currentColor">
                                <path d="M8 9a3 3 0 100-6 3 3 0 000 6zM8 11a6 6 0 016 6H2a6 6 0 016-6zM16 7a1 1 0 10-2 0v1h-1a1 1 0 100 2h1v1a1 1 0 102 0v-1h1a1 1 0 100-2h-1V7z" />
                              </svg>
                            </div>
                            <span className="font-medium" style={{ color: 'var(--color-text)' }}>New Users</span>
                          </div>
                          <span className="px-3 py-1 rounded-full text-sm font-semibold" style={{
                            backgroundColor: 'var(--color-success)20',
                            color: 'var(--color-success)',
                          }}>
                            {metrics?.userStats.recentUsers ?? 0}
                          </span>
                        </li>
                        <li className="flex items-center justify-between p-3 rounded-lg" style={{ backgroundColor: 'var(--color-surface-light)' }}>
                          <div className="flex items-center">
                            <div className="w-8 h-8 rounded-full flex items-center justify-center mr-3" style={{ backgroundColor: 'var(--color-primary)20' }}>
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" style={{ color: 'var(--color-primary)' }} viewBox="0 0 20 20" fill="currentColor">
                                <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
                                <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z" />
                              </svg>
                            </div>
                            <span className="font-medium" style={{ color: 'var(--color-text)' }}>New Messages</span>
                          </div>
                          <span className="px-3 py-1 rounded-full text-sm font-semibold" style={{
                            backgroundColor: 'var(--color-primary)20',
                            color: 'var(--color-primary)',
                          }}>
                            {metrics?.messageStats.recentMessages ?? 0}
                          </span>
                        </li>
                        <li className="flex items-center justify-between p-3 rounded-lg" style={{ backgroundColor: 'var(--color-surface-light)' }}>
                          <div className="flex items-center">
                            <div className="w-8 h-8 rounded-full flex items-center justify-center mr-3" style={{ backgroundColor: 'var(--color-warning)20' }}>
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" style={{ color: 'var(--color-warning)' }} viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                              </svg>
                            </div>
                            <span className="font-medium" style={{ color: 'var(--color-text)' }}>Documents Added</span>
                          </div>
                          <span className="px-3 py-1 rounded-full text-sm font-semibold" style={{
                            backgroundColor: 'var(--color-warning)20',
                            color: 'var(--color-warning)',
                          }}>
                            {metrics?.messageStats.totalDocuments ?? 0}
                          </span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                  className="rounded-2xl overflow-hidden h-full"
                  style={{
                    background: 'var(--color-surface)',
                    boxShadow: '0 10px 15px -3px var(--color-shadow), 0 4px 6px -2px var(--color-shadow-light)',
                    border: '1px solid var(--color-border)',
                    display: 'flex',
                    flexDirection: 'column',
                  }}
                >
                  <div className="bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-primary-light)] px-6 py-4">
                    <h3 className="text-lg font-semibold text-white flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
                      </svg>
                      Quick Actions
                    </h3>
                  </div>
                  
                  <div className="p-5 flex-grow flex flex-col">
                    <div className="space-y-4 flex-grow flex flex-col justify-between">
                      <Link 
                        to="/users" 
                        className="flex items-center p-4 rounded-xl transition-all hover:shadow-sm hover:bg-[var(--color-surface-light)]"
                        style={{
                          backgroundColor: 'var(--color-surface)',
                          border: '1px solid var(--color-border)',
                          height: '100%'
                        }}
                      >
                        <div className="w-10 h-10 rounded-full flex items-center justify-center mr-4" style={{ backgroundColor: 'var(--color-primary)20' }}>
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" style={{ color: 'var(--color-primary)' }} viewBox="0 0 20 20" fill="currentColor">
                            <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3zM6 8a2 2 0 11-4 0 2 2 0 014 0zM16 18v-3a5.972 5.972 0 00-.75-2.906A3.005 3.005 0 0119 15v3h-3zM4.75 12.094A5.973 5.973 0 004 15v3H1v-3a3 3 0 013.75-2.906z" />
                          </svg>
                        </div>
                        <div>
                          <h4 className="text-base font-semibold" style={{ color: 'var(--color-text)' }}>Manage Users</h4>
                          <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>Add, edit, remove users</p>
                        </div>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-auto" style={{ color: 'var(--color-text-secondary)' }} viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                      </Link>
                      
                      <Link 
                        to="/settings" 
                        className="flex items-center p-4 rounded-xl transition-all hover:shadow-sm hover:bg-[var(--color-surface-light)]"
                        style={{
                          backgroundColor: 'var(--color-surface)',
                          border: '1px solid var(--color-border)',
                          height: '100%'
                        }}
                      >
                        <div className="w-10 h-10 rounded-full flex items-center justify-center mr-4" style={{ backgroundColor: 'var(--color-primary)20' }}>
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" style={{ color: 'var(--color-primary)' }} viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                          </svg>
                        </div>
                        <div>
                          <h4 className="text-base font-semibold" style={{ color: 'var(--color-text)' }}>License Management</h4>
                          <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>Manage and assign licenses</p>
                        </div>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-auto" style={{ color: 'var(--color-text-secondary)' }} viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                      </Link>
                      
                      <Link 
                        to="/chatbot" 
                        className="flex items-center p-4 rounded-xl transition-all hover:shadow-sm hover:bg-[var(--color-surface-light)]"
                        style={{
                          backgroundColor: 'var(--color-surface)',
                          border: '1px solid var(--color-border)',
                          height: '100%'
                        }}
                      >
                        <div className="w-10 h-10 rounded-full flex items-center justify-center mr-4" style={{ backgroundColor: 'var(--color-primary)20' }}>
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" style={{ color: 'var(--color-primary)' }} viewBox="0 0 20 20" fill="currentColor">
                            <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
                            <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z" />
                          </svg>
                        </div>
                        <div>
                          <h4 className="text-base font-semibold" style={{ color: 'var(--color-text)' }}>Chatbot Interface</h4>
                          <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>Access chat functionality</p>
                        </div>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-auto" style={{ color: 'var(--color-text-secondary)' }} viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                      </Link>
                    </div>
                  </div>
                </motion.div>
              </div>
            </>
          )}
        </Suspense>
      </div>
    </div>
  );
}
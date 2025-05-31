import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { motion } from 'framer-motion';
import { PlusIcon, PencilIcon, TrashIcon, UserCircleIcon, UserGroupIcon, UsersIcon } from '@heroicons/react/24/outline';

interface User {
  id: number;
  name: string;
  username: string;
  email: string;
  role: string;
  created_at: string;
}

interface UserFormData {
  name: string;
  username: string;
  email: string;
  password: string;
  role: string;
}

const defaultUserForm: UserFormData = {
  name: '',
  username: '',
  email: '',
  password: '',
  role: 'user'
};

const AVAILABLE_ROLES = ['admin', 'user', 'viewer'];

export default function UserManagement() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [showRoleModal, setShowRoleModal] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [userForm, setUserForm] = useState<UserFormData>(defaultUserForm);
  const [successMessage, setSuccessMessage] = useState('');

  useEffect(() => {
    fetchUsers();
  }, []);

  useEffect(() => {
    // Clear success message after 3 seconds
    if (successMessage) {
      const timer = setTimeout(() => {
        setSuccessMessage('');
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [successMessage]);

  const fetchUsers = async () => {
    try {
      const response = await api.get('/users');
      setUsers(response.data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to fetch users');
    } finally {
      setLoading(false);
    }
  };

  const handleOpenModal = (user?: User) => {
    if (user) {
      setEditingUser(user);
      setUserForm({
        name: user.name || '',
        username: user.username,
        email: user.email || '',
        password: '', // Don't populate password for editing
        role: user.role
      });
    } else {
      setEditingUser(null);
      setUserForm(defaultUserForm);
    }
    setShowModal(true);
  };

  const handleOpenRoleModal = (user: User) => {
    setEditingUser(user);
    setUserForm({
      ...defaultUserForm,
      role: user.role,
      username: user.username,
      name: user.name || ''
    });
    setShowRoleModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setShowRoleModal(false);
    setEditingUser(null);
    setUserForm(defaultUserForm);
    setError('');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      if (editingUser) {
        // Remove empty fields from update
        const updateData = Object.entries(userForm).reduce((acc, [key, value]) => {
          if (value !== '') {
            acc[key] = value;
          }
          return acc;
        }, {} as Partial<UserFormData>);

        await api.put(`/users/${editingUser.id}`, updateData);
        setSuccessMessage(`User ${editingUser.username} updated successfully`);
      } else {
        await api.post('/users', userForm);
        setSuccessMessage('New user created successfully');
      }
      
      handleCloseModal();
      fetchUsers();
    } catch (err: any) {
      setError(err.response?.data?.error || `Failed to ${editingUser ? 'update' : 'create'} user`);
    }
  };

  const handleChangeRole = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingUser) return;
    
    setError('');
    try {
      await api.put(`/users/${editingUser.id}/role`, { role: userForm.role });
      setSuccessMessage(`Role updated to ${userForm.role} for ${editingUser.username}`);
      handleCloseModal();
      fetchUsers();
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to update user role');
    }
  };

  const handleDeleteUser = async (userId: number) => {
    if (!confirm('Are you sure you want to delete this user?')) return;
    
    try {
      await api.delete(`/users/${userId}`);
      setSuccessMessage('User deleted successfully');
      fetchUsers();
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to delete user');
    }
  };

  const getRoleBadgeStyle = (role: string) => {
    switch (role) {
      case 'admin':
        return 'bg-blue-500/20 text-blue-400';
      case 'user':
        return 'bg-green-500/20 text-green-400';
      case 'viewer':
        return 'bg-gray-500/20 text-gray-400';
      default:
        return 'bg-gray-500/20 text-gray-400';
    }
  };

  if (loading) {
    return (
      <div
        className="min-h-screen p-4 md:p-6 relative overflow-hidden flex items-center justify-center"
        style={{
          background: 'linear-gradient(to-b, var(--color-bg), var(--color-bg-dark))',
        }}
      >
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10" style={{
          backgroundImage: 'radial-gradient(circle at 2px 2px, var(--color-primary) 1px, transparent 0)',
          backgroundSize: '40px 40px',
        }} />
        
        <motion.div
          className="flex flex-col items-center justify-center p-8 rounded-xl backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          style={{
            background: 'linear-gradient(to-r, var(--color-surface)80, var(--color-surface-dark)80)',
            border: '1px solid var(--color-border)',
            boxShadow: '0 10px 25px -5px var(--color-shadow), 0 8px 10px -6px var(--color-shadow-light)',
          }}
        >
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            className="w-12 h-12 border-4 rounded-full mb-4"
            style={{ 
              borderColor: 'var(--color-primary)',
              borderTopColor: 'transparent'
            }}
          />
          <p className="text-lg" style={{ color: 'var(--color-text)' }}>Loading users...</p>
        </motion.div>
      </div>
    );
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
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h2 className="text-2xl md:text-3xl font-bold" style={{ color: 'var(--color-text)' }}>User Management</h2>
              <p style={{ color: 'var(--color-text-secondary)' }} className="mt-1">
                Manage user accounts and access permissions
              </p>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleOpenModal()}
              className="flex items-center justify-center px-4 py-2 rounded-lg transition-all text-white bg-blue-600 hover:bg-blue-700"
              style={{
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}
            >
              <PlusIcon className="w-5 h-5 mr-2" />
              <span className="font-medium">Add User</span>
            </motion.button>
          </div>
        </motion.div>

      {error && (
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
            {error}
          </div>
        </motion.div>
      )}

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

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="p-0 rounded-xl overflow-hidden shadow-card border" 
        style={{ 
          backgroundColor: 'var(--color-surface)',
          borderColor: 'var(--color-border)',
          boxShadow: '0 20px 25px -5px var(--color-shadow), 0 10px 10px -5px var(--color-shadow-light)',
          transform: 'translateZ(0)',
          backfaceVisibility: 'hidden',
        }}
      >
        <div className="bg-gradient-to-r from-[var(--color-primary-dark)] to-[var(--color-primary)] px-6 py-4 mb-0">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-white flex items-center">
              <UsersIcon className="h-5 w-5 mr-2" />
              User Accounts
            </h3>
            <div className="text-sm text-white/80">
              {users.length} {users.length === 1 ? 'user' : 'users'} total
            </div>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y" style={{ borderColor: 'var(--color-border)' }}>
            <thead style={{ backgroundColor: 'var(--color-surface-dark)' }}>
              <tr>
                <th className="px-4 md:px-6 py-3 text-left text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
                  Name
                </th>
                <th className="px-4 md:px-6 py-3 text-left text-xs font-medium uppercase tracking-wider hidden sm:table-cell" style={{ color: 'var(--color-text-secondary)' }}>
                  Username
                </th>
                <th className="px-4 md:px-6 py-3 text-left text-xs font-medium uppercase tracking-wider hidden md:table-cell" style={{ color: 'var(--color-text-secondary)' }}>
                  Email
                </th>
                <th className="px-4 md:px-6 py-3 text-left text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
                  Role
                </th>
                <th className="px-4 md:px-6 py-3 text-left text-xs font-medium uppercase tracking-wider hidden lg:table-cell" style={{ color: 'var(--color-text-secondary)' }}>
                  Created At
                </th>
                <th className="px-4 md:px-6 py-3 text-right text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y" style={{ borderColor: 'var(--color-border)' }}>
              {users.map((user) => (
                <motion.tr 
                  key={user.id} 
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className="hover:bg-surface-light transition-colors"
                  style={{ backgroundColor: 'var(--color-surface)' }}
                >
                  <td className="px-4 md:px-6 py-4 whitespace-nowrap" style={{ color: 'var(--color-text)' }}>
                    <div className="flex items-center">
                      <div className="w-8 h-8 rounded-full flex items-center justify-center mr-3 text-sm font-semibold" style={{
                        backgroundColor: 'var(--color-primary-20)',
                        color: 'var(--color-primary)'
                      }}>
                        {user.name ? user.name.charAt(0).toUpperCase() : user.username.charAt(0).toUpperCase()}
                      </div>
                      <span>{user.name || '-'}</span>
                    </div>
                  </td>
                  <td className="px-4 md:px-6 py-4 whitespace-nowrap hidden sm:table-cell" style={{ color: 'var(--color-text)' }}>{user.username}</td>
                  <td className="px-4 md:px-6 py-4 whitespace-nowrap hidden md:table-cell" style={{ color: 'var(--color-text)' }}>{user.email || '-'}</td>
                  <td className="px-4 md:px-6 py-4 whitespace-nowrap">
                    <motion.span 
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="px-2 py-1 rounded-full text-xs inline-block cursor-pointer font-medium"
                      onClick={() => handleOpenRoleModal(user)}
                      title="Click to change role"
                      style={{ 
                        backgroundColor: user.role === 'admin' ? 'rgba(var(--color-primary-rgb), 0.2)' : 
                                         user.role === 'user' ? 'rgba(var(--color-success-rgb), 0.2)' : 
                                         'rgba(var(--color-text-muted-rgb), 0.2)',
                        color: user.role === 'admin' ? 'var(--color-primary)' : 
                               user.role === 'user' ? 'var(--color-success)' : 
                               'var(--color-text-muted)',
                        border: `1px solid ${
                          user.role === 'admin' ? 'rgba(var(--color-primary-rgb), 0.5)' : 
                          user.role === 'user' ? 'rgba(var(--color-success-rgb), 0.5)' : 
                          'rgba(var(--color-text-muted-rgb), 0.5)'
                        }`
                      }}
                    >
                      {user.role}
                    </motion.span>
                  </td>
                  <td className="px-4 md:px-6 py-4 whitespace-nowrap hidden lg:table-cell" style={{ color: 'var(--color-text-muted)' }}>
                    {new Date(user.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 md:px-6 py-4 whitespace-nowrap text-right space-x-2">
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => handleOpenRoleModal(user)}
                      className="inline-flex items-center px-2 py-1 rounded-lg mr-1 font-medium"
                      title="Change role"
                      style={{ 
                        backgroundColor: 'rgba(var(--color-warning-rgb), 0.1)',
                        color: 'var(--color-warning)',
                        border: '1px solid rgba(var(--color-warning-rgb), 0.3)'
                      }}
                    >
                      <UserGroupIcon className="w-4 h-4" />
                      <span className="hidden sm:inline ml-1">Role</span>
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => handleOpenModal(user)}
                      className="inline-flex items-center px-2 py-1 rounded-lg mr-1 font-medium"
                      title="Edit user"
                      style={{ 
                        backgroundColor: 'rgba(var(--color-primary-rgb), 0.1)',
                        color: 'var(--color-primary)',
                        border: '1px solid rgba(var(--color-primary-rgb), 0.3)'
                      }}
                    >
                      <PencilIcon className="w-4 h-4" />
                      <span className="hidden sm:inline ml-1">Edit</span>
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => handleDeleteUser(user.id)}
                      className="inline-flex items-center px-2 py-1 rounded-lg font-medium"
                      title="Delete user"
                      style={{ 
                        backgroundColor: 'rgba(var(--color-error-rgb), 0.1)',
                        color: 'var(--color-error)',
                        border: '1px solid rgba(var(--color-error-rgb), 0.3)'
                      }}
                    >
                      <TrashIcon className="w-4 h-4" />
                      <span className="hidden sm:inline ml-1">Delete</span>
                    </motion.button>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* User Modal (Create/Edit) */}
      {showModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50 backdrop-blur-sm">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="p-0 rounded-xl overflow-hidden shadow-card border w-full max-w-md" 
            style={{ 
              backgroundColor: 'var(--color-surface)',
              borderColor: 'var(--color-border)',
              boxShadow: '0 20px 25px -5px var(--color-shadow), 0 10px 10px -5px var(--color-shadow-light)',
              transform: 'translateZ(0)',
              backfaceVisibility: 'hidden',
            }}
          >
            <div className="bg-gradient-to-r from-[var(--color-primary-dark)] to-[var(--color-primary)] px-6 py-4">
              <div className="flex items-center">
                <div className="w-10 h-10 rounded-full flex items-center justify-center mr-3" style={{
                  backgroundColor: 'rgba(255, 255, 255, 0.2)',
                }}>
                  {editingUser ? (
                    <PencilIcon className="h-5 w-5 text-white" />
                  ) : (
                    <PlusIcon className="h-5 w-5 text-white" />
                  )}
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">
                    {editingUser ? 'Edit User' : 'Create New User'}
                  </h3>
                  {editingUser && (
                    <p className="text-sm text-white/80">
                      {editingUser.username}
                    </p>
                  )}
                </div>
              </div>
            </div>
            
            <div className="p-6">
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1" style={{ color: 'var(--color-text-secondary)' }}>Name</label>
                  <input
                    type="text"
                    className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                    style={{
                      backgroundColor: 'var(--color-surface-dark)',
                      borderColor: 'var(--color-border)',
                      color: 'var(--color-text)'
                    }}
                    value={userForm.name}
                    onChange={(e) => setUserForm({ ...userForm, name: e.target.value })}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1" style={{ color: 'var(--color-text-secondary)' }}>Username</label>
                  <input
                    type="text"
                    className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                    style={{
                      backgroundColor: 'var(--color-surface-dark)',
                      borderColor: 'var(--color-border)',
                      color: 'var(--color-text)'
                    }}
                    value={userForm.username}
                    onChange={(e) => setUserForm({ ...userForm, username: e.target.value })}
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1" style={{ color: 'var(--color-text-secondary)' }}>Email</label>
                  <input
                    type="email"
                    className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                    style={{
                      backgroundColor: 'var(--color-surface-dark)',
                      borderColor: 'var(--color-border)',
                      color: 'var(--color-text)'
                    }}
                    value={userForm.email}
                    onChange={(e) => setUserForm({ ...userForm, email: e.target.value })}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1" style={{ color: 'var(--color-text-secondary)' }}>
                    Password {editingUser && '(leave blank to keep current)'}
                  </label>
                  <input
                    type="password"
                    className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                    style={{
                      backgroundColor: 'var(--color-surface-dark)',
                      borderColor: 'var(--color-border)',
                      color: 'var(--color-text)'
                    }}
                    value={userForm.password}
                    onChange={(e) => setUserForm({ ...userForm, password: e.target.value })}
                    required={!editingUser}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1" style={{ color: 'var(--color-text-secondary)' }}>Role</label>
                  <select
                    className="w-full rounded px-3 py-2 focus:outline-none focus:ring-1"
                    style={{
                      backgroundColor: 'var(--color-surface-dark)',
                      borderColor: 'var(--color-border)',
                      color: 'var(--color-text)'
                    }}
                    value={userForm.role}
                    onChange={(e) => setUserForm({ ...userForm, role: e.target.value })}
                  >
                    {AVAILABLE_ROLES.map(role => (
                      <option key={role} value={role}>
                        {role.charAt(0).toUpperCase() + role.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
                {error && (
                  <div className="text-sm p-2 rounded" style={{ backgroundColor: 'var(--color-error)10', color: 'var(--color-error)' }}>
                    {error}
                  </div>
                )}
                <div className="flex justify-end space-x-3 pt-4">
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    type="button"
                    onClick={handleCloseModal}
                    className="px-4 py-2 rounded-lg font-medium"
                    style={{
                      backgroundColor: 'var(--color-surface-dark)',
                      color: 'var(--color-text)',
                      border: '1px solid var(--color-border)'
                    }}
                  >
                    Cancel
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    type="submit"
                    className="px-4 py-2 rounded-lg text-white font-medium bg-blue-600 hover:bg-blue-700"
                    style={{
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                      border: '1px solid rgba(255, 255, 255, 0.1)'
                    }}
                  >
                    {editingUser ? 'Update User' : 'Create User'}
                  </motion.button>
                </div>
              </form>
            </div>
          </motion.div>
        </div>
      )}

      {/* Role Change Modal */}
      {showRoleModal && editingUser && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50 backdrop-blur-sm">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="p-0 rounded-xl overflow-hidden shadow-card border w-full max-w-md" 
            style={{ 
              backgroundColor: 'var(--color-surface)',
              borderColor: 'var(--color-border)',
              boxShadow: '0 20px 25px -5px var(--color-shadow), 0 10px 10px -5px var(--color-shadow-light)',
              transform: 'translateZ(0)',
              backfaceVisibility: 'hidden',
            }}
          >
            <div className="bg-gradient-to-r from-[var(--color-primary-dark)] to-[var(--color-primary)] px-6 py-4">
              <div className="flex items-center">
                <div className="w-10 h-10 rounded-full flex items-center justify-center mr-3" style={{
                  backgroundColor: 'rgba(255, 255, 255, 0.2)',
                }}>
                  <UserGroupIcon className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">Change User Role</h3>
                  <p className="text-sm text-white/80">
                    {editingUser.username} {editingUser.name ? `(${editingUser.name})` : ''}
                  </p>
                </div>
              </div>
            </div>

            <div className="p-6">
              <form onSubmit={handleChangeRole} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>Role</label>
                  <div className="grid grid-cols-1 gap-3">
                    {AVAILABLE_ROLES.map(role => (
                      <motion.div 
                        key={role}
                        whileHover={{ y: -3 }}
                        whileTap={{ scale: 0.98 }}
                        className="p-3 rounded-lg border transition-all cursor-pointer"
                        style={{
                          borderColor: userForm.role === role ? 'var(--color-primary)' : 'var(--color-border)',
                          backgroundColor: userForm.role === role ? 'var(--color-primary)10' : 'var(--color-surface)',
                          boxShadow: userForm.role === role ? '0 4px 6px -1px var(--color-shadow), 0 2px 4px -1px var(--color-shadow-light)' : 'none'
                        }}
                        onClick={() => setUserForm({...userForm, role})}
                      >
                        <div className="flex items-center">
                          <div className="w-5 h-5 rounded-full mr-3 flex items-center justify-center" style={{
                            backgroundColor: userForm.role === role ? 'var(--color-primary)' : 'var(--color-surface-dark)',
                            border: userForm.role === role ? 'none' : '1px solid var(--color-border)'
                          }}>
                            {userForm.role === role && (
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 text-white" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                              </svg>
                            )}
                          </div>
                          <div>
                            <div className="font-medium" style={{ color: 'var(--color-text)' }}>{role.charAt(0).toUpperCase() + role.slice(1)}</div>
                            <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                              {role === 'admin' && 'Full access to all features and user management'}
                              {role === 'user' && 'Full access to all features without user management'}
                              {role === 'viewer' && 'Read-only access to dashboards and data'}
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
                
                {error && (
                  <div className="text-sm p-2 rounded" style={{ backgroundColor: 'var(--color-error)10', color: 'var(--color-error)' }}>
                    {error}
                  </div>
                )}
                
                <div className="flex justify-end space-x-3 pt-4">
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    type="button"
                    onClick={handleCloseModal}
                    className="px-4 py-2 rounded-lg font-medium"
                    style={{
                      backgroundColor: 'var(--color-surface-dark)',
                      color: 'var(--color-text)',
                      border: '1px solid var(--color-border)'
                    }}
                  >
                    Cancel
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    type="submit"
                    className={`px-4 py-2 rounded-lg font-medium ${
                      userForm.role === editingUser.role 
                        ? 'bg-gray-300 text-gray-500' 
                        : 'bg-blue-600 hover:bg-blue-700 text-white'
                    }`}
                    disabled={userForm.role === editingUser.role}
                    style={{
                      boxShadow: userForm.role === editingUser.role ? 'none' : '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                      border: userForm.role === editingUser.role ? '1px solid #e5e7eb' : '1px solid rgba(255, 255, 255, 0.1)',
                      opacity: userForm.role === editingUser.role ? 0.7 : 1
                    }}
                  >
                    Update Role
                  </motion.button>
                </div>
              </form>
            </div>
          </motion.div>
        </div>
      )}
      </div>
    </div>
  );
} 
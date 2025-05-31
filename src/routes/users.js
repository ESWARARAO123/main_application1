const express = require('express');
const bcrypt = require('bcrypt');
const { db } = require('../database');
const router = express.Router();

// Middleware to check if user is authenticated
const isAuthenticated = (req, res, next) => {
  if (req.session.userId) {
    next();
  } else {
    res.status(401).json({ error: 'Unauthorized' });
  }
};

// Middleware to check if user is admin
const isAdmin = async (req, res, next) => {
  if (!req.session.userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  try {
    const user = await db.prepare('SELECT role FROM users WHERE id = ?').get(req.session.userId);
    if (!user || user.role !== 'admin') {
      res.status(403).json({ error: 'Forbidden' });
    } else {
      next();
    }
  } catch (error) {
    console.error('Admin check error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

// Get all users (admin only)
router.get('/', isAuthenticated, isAdmin, async (req, res) => {
  try {
    const users = await db.prepare('SELECT id, username, email, role, name, created_at FROM users').all();
    res.json(users);
  } catch (error) {
    console.error('Get users error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Create a new user (admin only)
router.post('/', isAuthenticated, isAdmin, async (req, res) => {
  const { username, password, email, role, name } = req.body;

  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password are required' });
  }

  try {
    // Check if username already exists
    const existingUser = await db.prepare('SELECT id FROM users WHERE username = ?').get(username);
    if (existingUser) {
      return res.status(400).json({ error: 'Username already exists' });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Insert user - including name field
    const result = await db.prepare(
      'INSERT INTO users (username, password, email, role, name) VALUES (?, ?, ?, ?, ?)'
    ).run(username, hashedPassword, email || null, role || 'user', name || null);

    res.status(201).json({
      id: result.lastInsertRowid,
      username,
      email,
      role: role || 'user',
      name
    });
  } catch (error) {
    console.error('Create user error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get a specific user
router.get('/:id', isAuthenticated, async (req, res) => {
  const { id } = req.params;
  
  // Only admins can view other users
  if (req.session.userId !== id && req.session.role !== 'admin') {
    return res.status(403).json({ error: 'Forbidden' });
  }

  try {
    const user = await db.prepare('SELECT id, username, email, role, name, created_at FROM users WHERE id = ?').get(id);
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json(user);
  } catch (error) {
    console.error('Get user error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Update user profile (self or admin)
router.put('/profile', isAuthenticated, async (req, res) => {
  const { username, email, name } = req.body;

  if (!username) {
    return res.status(400).json({ error: 'Username is required' });
  }

  try {
    // Check if username already exists (not belonging to this user)
    const existingUser = await db.prepare('SELECT id FROM users WHERE username = ? AND id != ?').get(username, req.session.userId);
    if (existingUser) {
      return res.status(400).json({ error: 'Username already exists' });
    }

    // Check if email already exists (not belonging to this user)
    if (email) {
      const existingEmail = await db.prepare('SELECT id FROM users WHERE email = ? AND id != ?').get(email, req.session.userId);
      if (existingEmail) {
        return res.status(400).json({ error: 'Email already exists' });
      }
    }

    // Update user profile - including name field
    await db.prepare(
      'UPDATE users SET username = ?, email = ?, name = ? WHERE id = ?'
    ).run(username, email || null, name || null, req.session.userId);

    // Return updated user
    const user = await db.prepare('SELECT id, username, email, role, name FROM users WHERE id = ?').get(req.session.userId);
    res.json(user);
  } catch (error) {
    console.error('Update profile error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Update user password
router.put('/password', isAuthenticated, async (req, res) => {
  const { currentPassword, newPassword } = req.body;

  if (!currentPassword || !newPassword) {
    return res.status(400).json({ error: 'Current password and new password are required' });
  }

  try {
    // Get current user with password
    const user = await db.prepare('SELECT password FROM users WHERE id = ?').get(req.session.userId);
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    // Verify current password
    const validPassword = await bcrypt.compare(currentPassword, user.password);
    if (!validPassword) {
      return res.status(400).json({ error: 'Current password is incorrect' });
    }
    
    // Hash new password
    const hashedPassword = await bcrypt.hash(newPassword, 10);
    
    // Update password
    await db.prepare('UPDATE users SET password = ? WHERE id = ?').run(hashedPassword, req.session.userId);
    
    res.json({ message: 'Password updated successfully' });
  } catch (error) {
    console.error('Update password error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Delete a user (admin only)
router.delete('/:id', isAuthenticated, isAdmin, async (req, res) => {
  const { id } = req.params;

  try {
    // Check if user exists
    const user = await db.prepare('SELECT id FROM users WHERE id = ?').get(id);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    // Delete user
    await db.prepare('DELETE FROM users WHERE id = ?').run(id);
    
    res.json({ message: 'User deleted successfully' });
  } catch (error) {
    console.error('Delete user error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Update user role (admin only)
router.put('/:id/role', isAuthenticated, isAdmin, async (req, res) => {
  const { id } = req.params;
  const { role } = req.body;

  if (!role || !['admin', 'user', 'viewer'].includes(role)) {
    return res.status(400).json({ error: 'Valid role is required (admin, user, or viewer)' });
  }

  try {
    // Check if user exists
    const user = await db.prepare('SELECT id, username FROM users WHERE id = ?').get(id);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    // Update user role
    await db.prepare('UPDATE users SET role = ? WHERE id = ?').run(role, id);
    
    res.json({ 
      message: `Role updated to ${role} successfully`,
      username: user.username,
      role
    });
  } catch (error) {
    console.error('Update role error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router; 
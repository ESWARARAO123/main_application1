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

// Login route
router.post('/login', async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password are required' });
  }

  try {
    const user = await db.prepare('SELECT * FROM users WHERE username = ?').get(username);
    
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Store user session
    req.session.userId = user.id;
    req.session.role = user.role;
    
    console.log('Login successful for user:', username);
    console.log('User role:', user.role);
    console.log('Session data:', req.session);

    res.json({
      id: user.id,
      username: user.username,
      role: user.role,
      name: user.name
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Logout route
router.post('/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      return res.status(500).json({ error: 'Error logging out' });
    }
    res.clearCookie('connect.sid');
    res.json({ message: 'Logged out successfully' });
  });
});

// Get current user info
router.get('/me', isAuthenticated, async (req, res) => {
  try {
    const user = await db.prepare('SELECT id, username, email, role, name FROM users WHERE id = ?').get(req.session.userId);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.json(user);
  } catch (error) {
    console.error('Get user error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = {
  router,
  requireAuth: isAuthenticated,
  requireAdmin: isAdmin
}; 
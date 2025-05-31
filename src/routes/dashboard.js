const express = require('express');
const { db, pool } = require('../database');
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

// Get dashboard metrics - Admin only
router.get('/metrics', isAuthenticated, isAdmin, async (req, res) => {
  console.log('Dashboard metrics requested by user:', req.session.userId);
  
  try {
    // Check if the dashboard_metrics table exists
    let useMetricsTable = false;
    try {
      const tableCheck = await pool.query(`
        SELECT EXISTS (
          SELECT FROM information_schema.tables 
          WHERE table_schema = 'public' 
          AND table_name = 'dashboard_metrics'
        ) AS table_exists;
      `);
      useMetricsTable = tableCheck.rows[0]?.table_exists;
    } catch (tableCheckError) {
      console.error('Error checking dashboard_metrics table:', tableCheckError);
    }

    let responseData = {
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
        expirationDate: "2024-12-31",
        daysRemaining: 245
      }
    };

    if (useMetricsTable) {
      console.log('Using dashboard_metrics table for metrics');
      
      // Get metrics from dashboard_metrics table
      const metricsQuery = await pool.query(`SELECT metric_name, metric_value FROM dashboard_metrics`);
      
      // Process the results
      for (const row of metricsQuery.rows) {
        if (row.metric_name === 'user_stats') {
          responseData.userStats = row.metric_value;
        } else if (row.metric_name === 'message_stats') {
          responseData.messageStats = row.metric_value;
        } else if (row.metric_name === 'license_usage') {
          responseData.licenseUsage = row.metric_value;
        }
      }

      // Force update metrics - normally this would happen via triggers
      try {
        await pool.query('SELECT update_dashboard_metrics()');
        console.log('Dashboard metrics updated');
      } catch (updateError) {
        console.error('Error updating dashboard metrics:', updateError);
      }
    } else {
      console.log('Using direct queries for metrics (dashboard_metrics table not found)');
      
      // Get user statistics - using the exact schema the user provided
      let userStats = { 
        total_users: 0, 
        admin_users: 0, 
        regular_users: 0, 
        recent_users: 0 
      };
      
      try {
        const userStatsQuery = await pool.query(`
          SELECT 
            COUNT(*) as total_users,
            COUNT(CASE WHEN role = 'admin' THEN 1 END) as admin_users,
            COUNT(CASE WHEN role = 'user' THEN 1 END) as regular_users,
            COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END) as recent_users
          FROM users
        `);
        
        userStats = userStatsQuery.rows[0] || userStats;
        console.log('User stats query result:', userStats);
        
        responseData.userStats = {
          totalUsers: parseInt(userStats.total_users) || 0,
          adminUsers: parseInt(userStats.admin_users) || 0,
          regularUsers: parseInt(userStats.regular_users) || 0,
          recentUsers: parseInt(userStats.recent_users) || 0
        };
      } catch (userError) {
        console.error('Error getting user statistics:', userError);
        console.error('Error details:', userError.message);
      }

      // Get message statistics
      let messageStats = { 
        total_messages: 0,
        recent_messages: 0
      };
      
      try {
        const messageStatsQuery = await pool.query(`
          SELECT 
            COUNT(*) as total_messages,
            COUNT(CASE WHEN timestamp > NOW() - INTERVAL '7 days' THEN 1 END) as recent_messages
          FROM messages
        `);
        
        messageStats = messageStatsQuery.rows[0] || messageStats;
        console.log('Message stats query result:', messageStats);
        
        responseData.messageStats.totalMessages = parseInt(messageStats.total_messages) || 0;
        responseData.messageStats.recentMessages = parseInt(messageStats.recent_messages) || 0;
      } catch (messageError) {
        console.error('Error getting message statistics:', messageError);
        console.error('Error details:', messageError.message);
      }

      // Count documents
      let documentCount = 0;
      try {
        const documentCountQuery = await pool.query(`
          SELECT COUNT(*) as total_documents FROM documents
        `);
        
        documentCount = documentCountQuery.rows[0]?.total_documents || 0;
        console.log('Document count query result:', documentCount);
        
        responseData.messageStats.totalDocuments = parseInt(documentCount) || 0;
      } catch (documentError) {
        console.error('Error getting document count:', documentError);
        console.error('Error details:', documentError.message);
      }

      // License usage - this is hardcoded since license information
      // isn't stored in the database yet
      responseData.licenseUsage = {
        totalLicenses: 25,
        activeLicenses: responseData.userStats.totalUsers > 0 ? 
          Math.min(responseData.userStats.totalUsers, 12) : 0,
        expirationDate: "2024-12-31",
        daysRemaining: 245
      };
    }
    
    console.log('Sending dashboard metrics response:', responseData);
    
    // Return metrics
    res.json(responseData);
  } catch (error) {
    console.error('Error fetching dashboard metrics:', error);
    res.status(500).json({ 
      error: 'Failed to fetch dashboard metrics',
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
  }
});

module.exports = router; 
const express = require('express');
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
const isAdmin = (req, res, next) => {
  if (!req.session.userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  try {
    const user = db.prepare('SELECT role FROM users WHERE id = ?').get(req.session.userId);
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

// Get all runs (admin only)
router.get('/', isAuthenticated, isAdmin, (req, res) => {
  try {
    // In a real app, this would fetch from the database
    res.json(mockRuns);
  } catch (error) {
    console.error('Error fetching runs:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get runs for a specific user
router.get('/user/:userId', isAuthenticated, (req, res) => {
  const userId = parseInt(req.params.userId);
  
  // Check if user is requesting their own runs or is an admin
  if (req.session.userId !== userId && req.session.role !== 'admin') {
    return res.status(403).json({ error: 'Forbidden' });
  }
  
  try {
    // In a real app, this would filter from the database
    const userRuns = mockRuns.filter(run => run.userId === userId);
    res.json(userRuns);
  } catch (error) {
    console.error('Error fetching user runs:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get a specific run
router.get('/:runId', isAuthenticated, (req, res) => {
  const runId = req.params.runId;
  
  try {
    // In a real app, this would fetch from the database
    const run = mockRuns.find(r => r.id === runId);
    
    if (!run) {
      return res.status(404).json({ error: 'Run not found' });
    }
    
    // Check if user is requesting their own run or is an admin
    if (req.session.userId !== run.userId && req.session.role !== 'admin') {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    res.json(run);
  } catch (error) {
    console.error('Error fetching run:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Pause a run
router.post('/:runId/pause', isAuthenticated, (req, res) => {
  const runId = req.params.runId;
  
  try {
    // In a real app, this would update the database
    const run = mockRuns.find(r => r.id === runId);
    
    if (!run) {
      return res.status(404).json({ error: 'Run not found' });
    }
    
    // Check if user owns the run or is an admin
    if (req.session.userId !== run.userId && req.session.role !== 'admin') {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    // Check if run can be paused
    if (run.status !== 'running') {
      return res.status(400).json({ error: 'Run is not in a running state' });
    }
    
    // Update run status (in a real app, this would update the database)
    run.status = 'paused';
    
    res.json({ message: 'Run paused successfully' });
  } catch (error) {
    console.error('Error pausing run:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Resume a run
router.post('/:runId/resume', isAuthenticated, (req, res) => {
  const runId = req.params.runId;
  
  try {
    // In a real app, this would update the database
    const run = mockRuns.find(r => r.id === runId);
    
    if (!run) {
      return res.status(404).json({ error: 'Run not found' });
    }
    
    // Check if user owns the run or is an admin
    if (req.session.userId !== run.userId && req.session.role !== 'admin') {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    // Check if run can be resumed
    if (run.status !== 'paused') {
      return res.status(400).json({ error: 'Run is not in a paused state' });
    }
    
    // Update run status (in a real app, this would update the database)
    run.status = 'running';
    
    res.json({ message: 'Run resumed successfully' });
  } catch (error) {
    console.error('Error resuming run:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Stop a run
router.post('/:runId/stop', isAuthenticated, (req, res) => {
  const runId = req.params.runId;
  
  try {
    // In a real app, this would update the database
    const run = mockRuns.find(r => r.id === runId);
    
    if (!run) {
      return res.status(404).json({ error: 'Run not found' });
    }
    
    // Check if user owns the run or is an admin
    if (req.session.userId !== run.userId && req.session.role !== 'admin') {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    // Check if run can be stopped
    if (run.status !== 'running' && run.status !== 'paused') {
      return res.status(400).json({ error: 'Run is not in a running or paused state' });
    }
    
    // Update run status (in a real app, this would update the database)
    run.status = 'completed';
    run.progress = 100;
    run.endTime = new Date();
    
    res.json({ message: 'Run stopped successfully' });
  } catch (error) {
    console.error('Error stopping run:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Retry a failed run
router.post('/:runId/retry', isAuthenticated, (req, res) => {
  const runId = req.params.runId;
  
  try {
    // In a real app, this would update the database
    const run = mockRuns.find(r => r.id === runId);
    
    if (!run) {
      return res.status(404).json({ error: 'Run not found' });
    }
    
    // Check if user owns the run or is an admin
    if (req.session.userId !== run.userId && req.session.role !== 'admin') {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    // Check if run can be retried
    if (run.status !== 'failed') {
      return res.status(400).json({ error: 'Run is not in a failed state' });
    }
    
    // Update run status (in a real app, this would update the database)
    run.status = 'running';
    run.progress = 0;
    run.startTime = new Date();
    run.endTime = undefined;
    
    // Reset failed steps
    run.steps = run.steps.map(step => {
      if (step.status === 'failed') {
        return { ...step, status: 'pending', errorMessage: undefined };
      }
      return step;
    });
    
    res.json({ message: 'Run retry initiated successfully' });
  } catch (error) {
    console.error('Error retrying run:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get logs for a run
router.get('/:runId/logs', isAuthenticated, (req, res) => {
  const runId = req.params.runId;
  
  try {
    // In a real app, this would fetch from the database
    const run = mockRuns.find(r => r.id === runId);
    
    if (!run) {
      return res.status(404).json({ error: 'Run not found' });
    }
    
    // Check if user owns the run or is an admin
    if (req.session.userId !== run.userId && req.session.role !== 'admin') {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    // Mock logs
    const logs = [
      `[${new Date(run.startTime).toISOString()}] Run started: ${run.name}`,
      ...run.steps.flatMap(step => {
        const logs = [];
        if (step.startTime) {
          logs.push(`[${new Date(step.startTime).toISOString()}] Step started: ${step.name}`);
        }
        if (step.endTime) {
          logs.push(`[${new Date(step.endTime).toISOString()}] Step completed: ${step.name}`);
        }
        if (step.errorMessage) {
          logs.push(`[${new Date(step.endTime).toISOString()}] Error in step ${step.name}: ${step.errorMessage}`);
        }
        return logs;
      }),
      run.endTime ? `[${new Date(run.endTime).toISOString()}] Run completed with status: ${run.status}` : ''
    ].filter(Boolean);
    
    res.json(logs);
  } catch (error) {
    console.error('Error fetching run logs:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get logs for a specific step in a run
router.get('/:runId/steps/:stepId/logs', isAuthenticated, (req, res) => {
  const { runId, stepId } = req.params;
  
  try {
    // In a real app, this would fetch from the database
    const run = mockRuns.find(r => r.id === runId);
    
    if (!run) {
      return res.status(404).json({ error: 'Run not found' });
    }
    
    const step = run.steps.find(s => s.id === stepId);
    
    if (!step) {
      return res.status(404).json({ error: 'Step not found' });
    }
    
    // Check if user owns the run or is an admin
    if (req.session.userId !== run.userId && req.session.role !== 'admin') {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    // Mock step logs
    const logs = [
      step.startTime ? `[${new Date(step.startTime).toISOString()}] Step started: ${step.name}` : '',
      `[${new Date().toISOString()}] Processing data for step: ${step.name}`,
      step.errorMessage ? `[${new Date(step.endTime).toISOString()}] Error: ${step.errorMessage}` : '',
      step.endTime ? `[${new Date(step.endTime).toISOString()}] Step completed: ${step.name}` : ''
    ].filter(Boolean);
    
    res.json(logs);
  } catch (error) {
    console.error('Error fetching step logs:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router; 
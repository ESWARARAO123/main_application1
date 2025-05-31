import { api } from './api';

export interface RunStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'warning';
  startTime?: Date;
  endTime?: Date;
  logs?: string[];
  errorMessage?: string;
}

export interface Run {
  id: string;
  name: string;
  userId: number;
  username: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  startTime: Date;
  endTime?: Date;
  steps: RunStep[];
  description?: string;
  type: string;
}

// Helper function to convert date strings from API to Date objects
const convertDates = (run: any): Run => {
  return {
    ...run,
    startTime: new Date(run.startTime),
    endTime: run.endTime ? new Date(run.endTime) : undefined,
    steps: run.steps.map((step: any) => ({
      ...step,
      startTime: step.startTime ? new Date(step.startTime) : undefined,
      endTime: step.endTime ? new Date(step.endTime) : undefined,
    }))
  };
};

const runService = {
  // Get all runs (admin only)
  getRuns: async (): Promise<Run[]> => {
    try {
      const response = await api.get('/runs');
      return response.data.map(convertDates);
    } catch (error) {
      console.error('Error fetching runs:', error);
      // For demo purposes, return mock data if API fails
      return mockRuns;
    }
  },
  
  // Get runs for a specific user
  getUserRuns: async (userId: number): Promise<Run[]> => {
    try {
      const response = await api.get(`/runs/user/${userId}`);
      return response.data.map(convertDates);
    } catch (error) {
      console.error('Error fetching user runs:', error);
      // For demo purposes, return filtered mock data if API fails
      return mockRuns.filter(run => run.userId === userId);
    }
  },
  
  // Get a specific run by ID
  getRun: async (runId: string): Promise<Run> => {
    try {
      const response = await api.get(`/runs/${runId}`);
      return convertDates(response.data);
    } catch (error) {
      console.error(`Error fetching run ${runId}:`, error);
      // For demo purposes, return mock data if API fails
      const run = mockRuns.find(r => r.id === runId);
      if (!run) throw new Error('Run not found');
      return run;
    }
  },
  
  // Pause a running run
  pauseRun: async (runId: string): Promise<void> => {
    try {
      await api.post(`/runs/${runId}/pause`);
    } catch (error) {
      console.error(`Error pausing run ${runId}:`, error);
      throw error;
    }
  },
  
  // Resume a paused run
  resumeRun: async (runId: string): Promise<void> => {
    try {
      await api.post(`/runs/${runId}/resume`);
    } catch (error) {
      console.error(`Error resuming run ${runId}:`, error);
      throw error;
    }
  },
  
  // Stop a run
  stopRun: async (runId: string): Promise<void> => {
    try {
      await api.post(`/runs/${runId}/stop`);
    } catch (error) {
      console.error(`Error stopping run ${runId}:`, error);
      throw error;
    }
  },
  
  // Retry a failed run
  retryRun: async (runId: string): Promise<void> => {
    try {
      await api.post(`/runs/${runId}/retry`);
    } catch (error) {
      console.error(`Error retrying run ${runId}:`, error);
      throw error;
    }
  },
  
  // Get logs for a specific run
  getRunLogs: async (runId: string): Promise<string[]> => {
    try {
      const response = await api.get(`/runs/${runId}/logs`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching logs for run ${runId}:`, error);
      return ['Error fetching logs'];
    }
  },
  
  // Get logs for a specific step in a run
  getStepLogs: async (runId: string, stepId: string): Promise<string[]> => {
    try {
      const response = await api.get(`/runs/${runId}/steps/${stepId}/logs`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching logs for step ${stepId} in run ${runId}:`, error);
      return ['Error fetching step logs'];
    }
  },
  
  // Get timing report for a run
  getTimingReport: async (runId: string): Promise<any> => {
    try {
      const response = await api.get(`/runs/${runId}/reports/timing`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching timing report for run ${runId}:`, error);
      return null;
    }
  },
  
  // Get QoR report for a run
  getQorReport: async (runId: string): Promise<any> => {
    try {
      const response = await api.get(`/runs/${runId}/reports/qor`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching QoR report for run ${runId}:`, error);
      return null;
    }
  },
  
  // Get DRC report for a run
  getDrcReport: async (runId: string): Promise<any> => {
    try {
      const response = await api.get(`/runs/${runId}/reports/drc`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching DRC report for run ${runId}:`, error);
      return null;
    }
  }
};

// Mock data for development and testing - VLSI PD specific
const mockRuns: Run[] = [
  // ===== PHYSICAL DESIGN RUNS =====
  {
    id: '1',
    name: 'BigEndian CPU Core 28nm',
    userId: 1,
    username: 'admin',
    status: 'completed',
    progress: 100,
    startTime: new Date(Date.now() - 3600000 * 24 * 3),
    endTime: new Date(Date.now() - 3600000 * 24),
    type: 'PD',
    description: 'BigEndian CPU core implementation in 28nm technology',
    steps: [
      {
        id: 's1',
        name: 'RTL Design',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 3),
        endTime: new Date(Date.now() - 3600000 * 24 * 2.8),
      },
      {
        id: 's2',
        name: 'Synthesis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 2.8),
        endTime: new Date(Date.now() - 3600000 * 24 * 2.5),
      },
      {
        id: 's3',
        name: 'Floorplan',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 2.5),
        endTime: new Date(Date.now() - 3600000 * 24 * 2.3),
      },
      {
        id: 's4',
        name: 'Placement',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 2.3),
        endTime: new Date(Date.now() - 3600000 * 24 * 2),
      },
      {
        id: 's5',
        name: 'Clock Tree Synthesis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 2),
        endTime: new Date(Date.now() - 3600000 * 24 * 1.8),
      },
      {
        id: 's6',
        name: 'Routing',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 1.8),
        endTime: new Date(Date.now() - 3600000 * 24 * 1.5),
      },
      {
        id: 's7',
        name: 'Timing Analysis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 1.5),
        endTime: new Date(Date.now() - 3600000 * 24 * 1.3),
      },
      {
        id: 's8',
        name: 'Power Analysis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 1.3),
        endTime: new Date(Date.now() - 3600000 * 24 * 1.1),
      },
      {
        id: 's9',
        name: 'DRC',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 1.1),
        endTime: new Date(Date.now() - 3600000 * 24 * 1.05),
      },
      {
        id: 's10',
        name: 'LVS',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 1.05),
        endTime: new Date(Date.now() - 3600000 * 24),
      },
      {
        id: 's11',
        name: 'Signoff',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24),
        endTime: new Date(Date.now() - 3600000 * 23.9),
      }
    ]
  },
  {
    id: '2',
    name: 'Aurdaine Memory Controller 2nm',
    userId: 1,
    username: 'admin',
    status: 'running',
    progress: 65,
    startTime: new Date(Date.now() - 3600000 * 12),
    type: 'PD',
    description: 'Aurdaine memory controller implementation in 2nm technology',
    steps: [
      {
        id: 's1',
        name: 'RTL Design',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 12),
        endTime: new Date(Date.now() - 3600000 * 10),
      },
      {
        id: 's2',
        name: 'Synthesis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 10),
        endTime: new Date(Date.now() - 3600000 * 8),
      },
      {
        id: 's3',
        name: 'Floorplan',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 8),
        endTime: new Date(Date.now() - 3600000 * 6),
      },
      {
        id: 's4',
        name: 'Placement',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 6),
        endTime: new Date(Date.now() - 3600000 * 4),
      },
      {
        id: 's5',
        name: 'Clock Tree Synthesis',
        status: 'running',
        startTime: new Date(Date.now() - 3600000 * 4),
      },
      {
        id: 's6',
        name: 'Routing',
        status: 'pending',
      },
      {
        id: 's7',
        name: 'Timing Analysis',
        status: 'pending',
      },
      {
        id: 's8',
        name: 'Power Analysis',
        status: 'pending',
      },
      {
        id: 's9',
        name: 'DRC',
        status: 'pending',
      },
      {
        id: 's10',
        name: 'LVS',
        status: 'pending',
      },
      {
        id: 's11',
        name: 'Signoff',
        status: 'pending',
      }
    ]
  },
  {
    id: '3',
    name: 'BigEndian I/O Controller 28nm',
    userId: 2,
    username: 'john.doe',
    status: 'failed',
    progress: 33,
    startTime: new Date(Date.now() - 3600000 * 48),
    endTime: new Date(Date.now() - 3600000 * 36),
    type: 'PD',
    description: 'BigEndian I/O controller implementation in 28nm technology',
    steps: [
      {
        id: 's1',
        name: 'RTL Design',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 48),
        endTime: new Date(Date.now() - 3600000 * 46),
      },
      {
        id: 's2',
        name: 'Synthesis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 46),
        endTime: new Date(Date.now() - 3600000 * 44),
      },
      {
        id: 's3',
        name: 'Floorplan',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 44),
        endTime: new Date(Date.now() - 3600000 * 42),
      },
      {
        id: 's4',
        name: 'Placement',
        status: 'failed',
        startTime: new Date(Date.now() - 3600000 * 42),
        endTime: new Date(Date.now() - 3600000 * 36),
        errorMessage: 'Congestion issues in the core area, unable to meet density targets',
      },
      {
        id: 's5',
        name: 'Clock Tree Synthesis',
        status: 'pending',
      },
      {
        id: 's6',
        name: 'Routing',
        status: 'pending',
      },
      {
        id: 's7',
        name: 'Timing Analysis',
        status: 'pending',
      },
      {
        id: 's8',
        name: 'Power Analysis',
        status: 'pending',
      },
      {
        id: 's9',
        name: 'DRC',
        status: 'pending',
      },
      {
        id: 's10',
        name: 'LVS',
        status: 'pending',
      },
      {
        id: 's11',
        name: 'Signoff',
        status: 'pending',
      }
    ]
  },
  {
    id: '4',
    name: 'Aurdaine GPU Core 2nm',
    userId: 2,
    username: 'john.doe',
    status: 'paused',
    progress: 75,
    startTime: new Date(Date.now() - 3600000 * 72),
    type: 'PD',
    description: 'Aurdaine GPU core implementation in 2nm technology',
    steps: [
      {
        id: 's1',
        name: 'RTL Design',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 72),
        endTime: new Date(Date.now() - 3600000 * 70),
      },
      {
        id: 's2',
        name: 'Synthesis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 70),
        endTime: new Date(Date.now() - 3600000 * 68),
      },
      {
        id: 's3',
        name: 'Floorplan',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 68),
        endTime: new Date(Date.now() - 3600000 * 66),
      },
      {
        id: 's4',
        name: 'Placement',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 66),
        endTime: new Date(Date.now() - 3600000 * 64),
      },
      {
        id: 's5',
        name: 'Clock Tree Synthesis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 64),
        endTime: new Date(Date.now() - 3600000 * 62),
      },
      {
        id: 's6',
        name: 'Routing',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 62),
        endTime: new Date(Date.now() - 3600000 * 60),
      },
      {
        id: 's7',
        name: 'Timing Analysis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 60),
        endTime: new Date(Date.now() - 3600000 * 58),
      },
      {
        id: 's8',
        name: 'Power Analysis',
        status: 'warning',
        startTime: new Date(Date.now() - 3600000 * 58),
        errorMessage: 'License server unavailable, waiting for license',
      },
      {
        id: 's9',
        name: 'DRC',
        status: 'pending',
      },
      {
        id: 's10',
        name: 'LVS',
        status: 'pending',
      },
      {
        id: 's11',
        name: 'Signoff',
        status: 'pending',
      }
    ]
  },
  {
    id: '5',
    name: 'BigEndian SerDes 28nm',
    userId: 3,
    username: 'jane.smith',
    status: 'completed',
    progress: 100,
    startTime: new Date(Date.now() - 3600000 * 24 * 7),
    endTime: new Date(Date.now() - 3600000 * 24 * 5),
    type: 'PD',
    description: 'BigEndian SerDes implementation in 28nm technology',
    steps: [
      {
        id: 's1',
        name: 'RTL Design',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 7),
        endTime: new Date(Date.now() - 3600000 * 24 * 6.8),
      },
      {
        id: 's2',
        name: 'Synthesis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 6.8),
        endTime: new Date(Date.now() - 3600000 * 24 * 6.6),
      },
      {
        id: 's3',
        name: 'Floorplan',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 6.6),
        endTime: new Date(Date.now() - 3600000 * 24 * 6.4),
      },
      {
        id: 's4',
        name: 'Placement',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 6.4),
        endTime: new Date(Date.now() - 3600000 * 24 * 6.2),
      },
      {
        id: 's5',
        name: 'Clock Tree Synthesis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 6.2),
        endTime: new Date(Date.now() - 3600000 * 24 * 6),
      },
      {
        id: 's6',
        name: 'Routing',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 6),
        endTime: new Date(Date.now() - 3600000 * 24 * 5.8),
      },
      {
        id: 's7',
        name: 'Timing Analysis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 5.8),
        endTime: new Date(Date.now() - 3600000 * 24 * 5.6),
      },
      {
        id: 's8',
        name: 'Power Analysis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 5.6),
        endTime: new Date(Date.now() - 3600000 * 24 * 5.4),
      },
      {
        id: 's9',
        name: 'DRC',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 5.4),
        endTime: new Date(Date.now() - 3600000 * 24 * 5.2),
      },
      {
        id: 's10',
        name: 'LVS',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 5.2),
        endTime: new Date(Date.now() - 3600000 * 24 * 5.1),
      },
      {
        id: 's11',
        name: 'Signoff',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 24 * 5.1),
        endTime: new Date(Date.now() - 3600000 * 24 * 5),
      }
    ]
  },
  {
    id: '6',
    name: 'Aurdaine NPU 2nm',
    userId: 3,
    username: 'jane.smith',
    status: 'pending',
    progress: 0,
    startTime: new Date(Date.now() + 3600000 * 24), // Scheduled for future
    type: 'PD',
    description: 'Aurdaine Neural Processing Unit implementation in 2nm technology',
    steps: [
      {
        id: 's1',
        name: 'RTL Design',
        status: 'pending',
      },
      {
        id: 's2',
        name: 'Synthesis',
        status: 'pending',
      },
      {
        id: 's3',
        name: 'Floorplan',
        status: 'pending',
      },
      {
        id: 's4',
        name: 'Placement',
        status: 'pending',
      },
      {
        id: 's5',
        name: 'Clock Tree Synthesis',
        status: 'pending',
      },
      {
        id: 's6',
        name: 'Routing',
        status: 'pending',
      },
      {
        id: 's7',
        name: 'Timing Analysis',
        status: 'pending',
      },
      {
        id: 's8',
        name: 'Power Analysis',
        status: 'pending',
      },
      {
        id: 's9',
        name: 'DRC',
        status: 'pending',
      },
      {
        id: 's10',
        name: 'LVS',
        status: 'pending',
      },
      {
        id: 's11',
        name: 'Signoff',
        status: 'pending',
      }
    ]
  },
  
  // ===== REPORT GENERATION RUNS =====
  {
    id: '7',
    name: 'Timing Report - BigEndian I/O',
    userId: 2,
    username: 'john.doe',
    status: 'completed',
    progress: 100,
    startTime: new Date(Date.now() - 3600000 * 36),
    endTime: new Date(Date.now() - 3600000 * 35),
    type: 'Report',
    description: 'Generate timing reports for BigEndian I/O Controller',
    steps: [
      {
        id: 's1',
        name: 'Setup Timing Analysis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 36),
        endTime: new Date(Date.now() - 3600000 * 35.8),
      },
      {
        id: 's2',
        name: 'Run Timing Analysis',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 35.8),
        endTime: new Date(Date.now() - 3600000 * 35.5),
      },
      {
        id: 's3',
        name: 'Generate Timing Report',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 35.5),
        endTime: new Date(Date.now() - 3600000 * 35),
      }
    ]
  },
  {
    id: '8',
    name: 'QoR Report - BigEndian CPU',
    userId: 1,
    username: 'admin',
    status: 'running',
    progress: 66,
    startTime: new Date(Date.now() - 3600000 * 2),
    type: 'Report',
    description: 'Generate Quality of Results report for BigEndian CPU Core',
    steps: [
      {
        id: 's1',
        name: 'Collect Design Metrics',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 2),
        endTime: new Date(Date.now() - 3600000 * 1.5),
      },
      {
        id: 's2',
        name: 'Analyze Performance',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 1.5),
        endTime: new Date(Date.now() - 3600000 * 1),
      },
      {
        id: 's3',
        name: 'Generate QoR Report',
        status: 'running',
        startTime: new Date(Date.now() - 3600000 * 1),
      }
    ]
  },
  {
    id: '9',
    name: 'DRC Report - BigEndian SerDes',
    userId: 3,
    username: 'jane.smith',
    status: 'failed',
    progress: 33,
    startTime: new Date(Date.now() - 3600000 * 12),
    endTime: new Date(Date.now() - 3600000 * 11),
    type: 'Report',
    description: 'Generate Design Rule Check report for BigEndian SerDes',
    steps: [
      {
        id: 's1',
        name: 'Setup DRC Run',
        status: 'completed',
        startTime: new Date(Date.now() - 3600000 * 12),
        endTime: new Date(Date.now() - 3600000 * 11.8),
      },
      {
        id: 's2',
        name: 'Run DRC Check',
        status: 'failed',
        startTime: new Date(Date.now() - 3600000 * 11.8),
        endTime: new Date(Date.now() - 3600000 * 11),
        errorMessage: 'DRC tool license expired during run',
      },
      {
        id: 's3',
        name: 'Generate DRC Report',
        status: 'pending',
      }
    ]
  }
];

export default runService; 
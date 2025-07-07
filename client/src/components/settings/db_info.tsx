import React, { useState, useEffect } from 'react';
import axios from 'axios';

const DBInfoSettings: React.FC = () => {
  const [form, setForm] = useState({
    host: '',
    database: '',
    user: '',
    password: '',
    port: ''
  });
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [testPassed, setTestPassed] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [saveMessage, setSaveMessage] = useState('');

  const API_URL = '/api/chat2sql/db-config'; // or use your full URL if needed

  // Helper to get username from localStorage (assumes user info is stored as 'user')
  function getUsername() {
    try {
      const user = localStorage.getItem('user');
      if (user) {
        const parsed = JSON.parse(user);
        return parsed.username || 'default';
      }
    } catch {}
    return 'default';
  }

  // Fetch existing config on mount (prefill form, but do not test connection)
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setLoading(true);
        const res = await axios.get('http://localhost:5000/api/chat2sql/db-config', {
          headers: { 'x-username': getUsername() }
        });
        if (res.data) {
          setForm({
            host: res.data.host || '',
            database: res.data.database || '',
            user: res.data.user || '',
            password: res.data.password || '',
            port: res.data.port ? String(res.data.port) : ''
          });
        }
      } catch (e) {
        // ignore
      } finally {
        setLoading(false);
      }
    };
    fetchConfig();
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setTestPassed(false); // Reset test status on change
    setStatus('idle');
    setMessage('');
    setSaveStatus('idle');
    setSaveMessage('');
  };

  // Test connection only
  const handleTestConnection = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setStatus('idle');
    setMessage('');
    setTestPassed(false);
    try {
      const res = await axios.post(API_URL, {
        ...form,
        port: Number(form.port)
      });
      setStatus('success');
      setMessage('Connection successful!');
      setTestPassed(true);
    } catch (err: any) {
      setStatus('error');
      const errorMsg = err.response?.data?.error || 'Failed to connect to database.';
      setMessage(errorMsg);
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${errorMsg}`]);
      setTestPassed(false);
    } finally {
      setLoading(false);
    }
  };

  // Save config only if last test passed
  const handleSaveSettings = async () => {
    setSaveStatus('idle');
    setSaveMessage('');
    if (!testPassed) {
      setSaveStatus('error');
      setSaveMessage('Please test the connection and ensure it is successful before saving.');
      return;
    }
    setLoading(true);
    try {
      await axios.post(API_URL, {
        ...form,
        port: Number(form.port),
        save: true
      });
      setSaveStatus('success');
      setSaveMessage('Settings saved successfully!');
    } catch (err: any) {
      setSaveStatus('error');
      setSaveMessage(err.response?.data?.error || 'Failed to save settings.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--color-text)' }}>Database Settings</h2>
      <form className="space-y-4 max-w-md" onSubmit={handleTestConnection}>
        <div>
          <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>Host</label>
          <input type="text" name="host" value={form.host} onChange={handleChange} placeholder="localhost" className="w-full rounded px-3 py-2" style={{ backgroundColor: 'var(--color-surface-dark)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }} required />
        </div>
        <div>
          <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>Database</label>
          <input type="text" name="database" value={form.database} onChange={handleChange} placeholder="copilot" className="w-full rounded px-3 py-2" style={{ backgroundColor: 'var(--color-surface-dark)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }} required />
        </div>
        <div>
          <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>User</label>
          <input type="text" name="user" value={form.user} onChange={handleChange} placeholder="postgres" className="w-full rounded px-3 py-2" style={{ backgroundColor: 'var(--color-surface-dark)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }} required />
        </div>
        <div>
          <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>Password</label>
          <input type="password" name="password" value={form.password} onChange={handleChange} placeholder="" className="w-full rounded px-3 py-2" style={{ backgroundColor: 'var(--color-surface-dark)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }} required />
        </div>
        <div>
          <label className="block text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>Port</label>
          <input type="number" name="port" value={form.port} onChange={handleChange} placeholder="5432" className="w-full rounded px-3 py-2" style={{ backgroundColor: 'var(--color-surface-dark)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }} required />
        </div>
        <button type="submit" disabled={loading} className="w-full px-4 py-2 rounded-lg mt-4 font-semibold transition-all bg-blue-600 hover:bg-blue-700 text-white shadow-md">
          {loading ? 'Testing...' : 'Test Connection'}
        </button>
      </form>
      {status === 'success' && <div className="mt-4 text-green-600">{message}</div>}
      {status === 'error' && <div className="mt-4 text-red-600">{message}</div>}
      <button
        className="w-full px-4 py-2 rounded-lg mt-4 font-semibold transition-all bg-blue-500 hover:bg-blue-600 text-white shadow-md"
        style={{ opacity: testPassed ? 1 : 0.6, cursor: testPassed ? 'pointer' : 'not-allowed' }}
        onClick={handleSaveSettings}
        disabled={!testPassed || loading}
      >
        Save Settings
      </button>
      {saveStatus === 'success' && <div className="mt-4 text-green-600">{saveMessage}</div>}
      {saveStatus === 'error' && <div className="mt-4 text-red-600">{saveMessage}</div>}
      {logs.length > 0 && (
        <div className="mt-6 p-3 bg-gray-900 rounded text-xs text-red-300 max-w-md">
          <div className="mb-2 font-semibold text-red-400">Connection Logs:</div>
          <ul className="space-y-1">
            {logs.map((log, idx) => <li key={idx}>{log}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
};

export default DBInfoSettings;

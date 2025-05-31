import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

type ThemeType = 'dark' | 'light' | 'midnight';

interface ThemeContextType {
  currentTheme: ThemeType;
  setTheme: (theme: ThemeType) => void;
  isDark: boolean;
}

const ThemeContext = createContext<ThemeContextType>({
  currentTheme: 'dark',
  setTheme: () => {},
  isDark: true,
});

export const useTheme = () => useContext(ThemeContext);

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [currentTheme, setCurrentTheme] = useState<ThemeType>('dark');

  useEffect(() => {
    // Try to get theme from localStorage
    const savedTheme = localStorage.getItem('theme') as ThemeType;
    if (savedTheme && ['dark', 'light', 'midnight'].includes(savedTheme)) {
      setCurrentTheme(savedTheme);
    } else {
      // Use system preference as fallback
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setCurrentTheme(prefersDark ? 'dark' : 'light');
    }
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    
    // Remove all theme classes
    root.classList.remove('theme-dark', 'theme-light', 'theme-midnight');
    
    // Add current theme class
    root.classList.add(`theme-${currentTheme}`);
    
    // Save to localStorage
    localStorage.setItem('theme', currentTheme);

    // Update CSS variables based on theme
    if (currentTheme === 'light') {
      // Light theme
      root.style.setProperty('--color-bg', '#f8fafc');
      root.style.setProperty('--color-surface', '#ffffff');
      root.style.setProperty('--color-surface-light', '#f1f5f9');
      root.style.setProperty('--color-surface-dark', '#e2e8f0');
      
      root.style.setProperty('--color-primary', '#3b82f6');
      root.style.setProperty('--color-primary-light', '#60a5fa');
      root.style.setProperty('--color-primary-dark', '#2563eb');
      
      root.style.setProperty('--color-secondary', '#8b5cf6');
      root.style.setProperty('--color-secondary-light', '#a78bfa');
      root.style.setProperty('--color-secondary-dark', '#7c3aed');
      
      root.style.setProperty('--color-text', '#1e293b');
      root.style.setProperty('--color-text-secondary', '#475569');
      root.style.setProperty('--color-text-muted', '#94a3b8');
      
      root.style.setProperty('--color-border', '#e2e8f0');
      root.style.setProperty('--color-border-light', '#cbd5e1');
      
      // Status colors
      root.style.setProperty('--color-success', '#10b981');
      root.style.setProperty('--color-warning', '#f59e0b');
      root.style.setProperty('--color-error', '#ef4444');
      root.style.setProperty('--color-info', '#3b82f6');
    } 
    else if (currentTheme === 'midnight') {
      // Midnight theme
      root.style.setProperty('--color-bg', '#000000');
      root.style.setProperty('--color-surface', '#111827');
      root.style.setProperty('--color-surface-light', '#1f2937');
      root.style.setProperty('--color-surface-dark', '#0f172a');
      
      root.style.setProperty('--color-primary', '#8b5cf6');
      root.style.setProperty('--color-primary-light', '#a78bfa');
      root.style.setProperty('--color-primary-dark', '#7c3aed');
      
      root.style.setProperty('--color-secondary', '#ec4899');
      root.style.setProperty('--color-secondary-light', '#f472b6');
      root.style.setProperty('--color-secondary-dark', '#db2777');
      
      root.style.setProperty('--color-text', '#f9fafb');
      root.style.setProperty('--color-text-secondary', '#e5e7eb');
      root.style.setProperty('--color-text-muted', '#9ca3af');
      
      root.style.setProperty('--color-border', '#374151');
      root.style.setProperty('--color-border-light', '#4b5563');
      
      // Status colors
      root.style.setProperty('--color-success', '#10b981');
      root.style.setProperty('--color-warning', '#f59e0b');
      root.style.setProperty('--color-error', '#ef4444');
      root.style.setProperty('--color-info', '#6366f1');
    }
    else {
      // Default dark theme
      root.style.setProperty('--color-bg', '#0f1117');
      root.style.setProperty('--color-surface', '#1a1f2d');
      root.style.setProperty('--color-surface-light', '#232a3d');
      root.style.setProperty('--color-surface-dark', '#151c2c');
      
      root.style.setProperty('--color-primary', '#3b82f6');
      root.style.setProperty('--color-primary-light', '#60a5fa');
      root.style.setProperty('--color-primary-dark', '#2563eb');
      
      root.style.setProperty('--color-secondary', '#8b5cf6');
      root.style.setProperty('--color-secondary-light', '#a78bfa');
      root.style.setProperty('--color-secondary-dark', '#7c3aed');
      
      root.style.setProperty('--color-text', '#f3f4f6');
      root.style.setProperty('--color-text-secondary', '#9ca3af');
      root.style.setProperty('--color-text-muted', '#6b7280');
      
      root.style.setProperty('--color-border', '#2a3349');
      root.style.setProperty('--color-border-light', '#384466');
      
      // Status colors
      root.style.setProperty('--color-success', '#10b981');
      root.style.setProperty('--color-warning', '#f59e0b');
      root.style.setProperty('--color-error', '#ef4444');
      root.style.setProperty('--color-info', '#3b82f6');
    }
    
    // Chart colors (consistent across themes)
    root.style.setProperty('--color-chart-1', '#3b82f6');
    root.style.setProperty('--color-chart-2', '#8b5cf6');
    root.style.setProperty('--color-chart-3', '#ec4899');
    root.style.setProperty('--color-chart-4', '#10b981');
    root.style.setProperty('--color-chart-5', '#f59e0b');
    
  }, [currentTheme]);

  const setTheme = (theme: ThemeType) => {
    setCurrentTheme(theme);
  };

  return (
    <ThemeContext.Provider 
      value={{ 
        currentTheme, 
        setTheme,
        isDark: currentTheme !== 'light'
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};

export default ThemeProvider; 
import React from 'react';

interface ContextReadingIndicatorProps {
  isReading: boolean;
}

const ContextReadingIndicator: React.FC<ContextReadingIndicatorProps> = ({ isReading }) => {
  if (!isReading) return null;

  return (
    <div className="flex items-center p-2 mb-2 rounded-md" style={{ 
      backgroundColor: 'var(--color-surface-accent)',
      border: '1px solid var(--color-border-accent)'
    }}>
      <div className="inline-block animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 mr-2" style={{ 
        borderColor: 'var(--color-primary)' 
      }}></div>
      <span style={{ color: 'var(--color-text)' }}>
        Reading your context preferences...
      </span>
    </div>
  );
};

export default ContextReadingIndicator;

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

interface MetricDetail {
  label: string;
  value: number | string;
  highlight?: boolean;
}

interface MetricCardProps {
  title: string;
  primaryMetric: number | string;
  primaryLabel: string;
  details: MetricDetail[];
  link?: {
    to: string;
    label: string;
  };
  progress?: number;
  badge?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  primaryMetric,
  primaryLabel,
  details,
  link,
  progress,
  badge
}) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ 
        y: -8,
        transition: { duration: 0.2 }
      }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="p-6 rounded-xl backdrop-blur-sm relative overflow-hidden h-full"
      style={{
        background: 'var(--color-surface)',
        boxShadow: isHovered 
          ? '0 20px 25px -5px var(--color-shadow-dark), 0 10px 10px -5px var(--color-shadow)'
          : '0 10px 15px -3px var(--color-shadow), 0 4px 6px -2px var(--color-shadow-light)',
        border: '1px solid var(--color-border)',
        transition: 'all 0.3s ease',
        display: 'flex',
        flexDirection: 'column',
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Decorative top border */}
      <div className="absolute top-0 left-0 w-full h-1" style={{
        background: 'linear-gradient(to right, var(--color-primary), var(--color-primary-dark))',
      }}></div>
      
      {/* Background glow effect on hover */}
      {isHovered && (
        <div className="absolute -top-10 -right-10 w-40 h-40 rounded-full" style={{
          background: 'radial-gradient(circle, var(--color-primary-glow) 0%, transparent 70%)',
          zIndex: 0,
        }}></div>
      )}

      <div className="flex justify-between items-start mb-4 relative z-10">
        <h3 className="text-lg font-semibold" style={{ color: 'var(--color-text)' }}>{title}</h3>
        {badge && (
          <span className="px-2 py-1 text-xs rounded-full font-semibold" style={{
            backgroundColor: 'var(--color-primary)20',
            color: 'var(--color-primary)',
          }}>
            {badge}
          </span>
        )}
      </div>

      <div className="mb-6 relative z-10">
        <div className="text-3xl font-bold mb-1" style={{ color: 'var(--color-primary)' }}>
          {primaryMetric}
        </div>
        <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          {primaryLabel}
        </div>
      </div>

      {progress !== undefined && (
        <div className="mb-4 relative z-10">
          <div className="h-2 w-full rounded-full overflow-hidden" style={{ backgroundColor: 'var(--color-surface-light)' }}>
            <div
              className="h-full rounded-full"
              style={{
                width: `${progress}%`,
                background: 'linear-gradient(to right, var(--color-primary), var(--color-primary-dark))'
              }}
            />
          </div>
          <div className="text-xs mt-1 text-right" style={{ color: 'var(--color-text-secondary)' }}>
            {progress}% used
          </div>
        </div>
      )}

      <div className="space-y-3 mb-6 relative z-10 flex-grow">
        {details.map((detail, index) => (
          <div 
            key={index} 
            className="flex justify-between items-center p-2 rounded-lg transition-colors"
            style={{ 
              backgroundColor: 'var(--color-surface)'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--color-surface-light)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--color-surface)';
            }}
          >
            <span className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
              {detail.label}
            </span>
            <span
              className="font-medium px-2 py-1 rounded-md"
              style={{
                backgroundColor: detail.highlight ? 'var(--color-primary)20' : 'transparent',
                color: detail.highlight ? 'var(--color-primary)' : 'var(--color-text)'
              }}
            >
              {detail.value}
            </span>
          </div>
        ))}
      </div>

      {link && (
        <div className="relative z-10 mt-auto pt-2">
          <Link
            to={link.to}
            className="text-sm flex items-center justify-center w-full py-3 rounded-lg transition-all"
            style={{
              background: isHovered 
                ? 'linear-gradient(135deg, var(--color-primary), var(--color-primary-dark))' 
                : 'var(--color-surface-light)',
              color: isHovered ? 'white' : 'var(--color-primary)',
              border: '1px solid',
              borderColor: isHovered ? 'transparent' : 'var(--color-border)',
              boxShadow: isHovered ? '0 4px 6px -1px var(--color-shadow)' : 'none',
              fontWeight: '600',
              height: '42px', // Fixed height for alignment
              lineHeight: '1',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {link.label}
            {isHovered && (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            )}
          </Link>
        </div>
      )}
    </motion.div>
  );
};

export default MetricCard;
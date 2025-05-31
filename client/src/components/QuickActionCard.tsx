import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

interface QuickAction {
  to: string;
  label: string;
  description: string;
  color: string;
}

interface QuickActionCardProps {
  actions: QuickAction[];
}

const QuickActionCard: React.FC<QuickActionCardProps> = ({ actions }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="p-6 rounded-xl backdrop-blur-sm"
      style={{
        background: 'linear-gradient(to-r, var(--color-surface), var(--color-surface-dark))',
        border: '1px solid var(--color-border)',
      }}
    >
      <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--color-text)' }}>Quick Actions</h3>
      <div className="space-y-3">
        {actions.map((action, index) => (
          <Link
            key={index}
            to={action.to}
            className="p-3 rounded-lg flex items-center hover:bg-[var(--color-primary)]10 transition-all duration-300"
            style={{ color: 'var(--color-text)' }}
          >
            <div className="p-2 rounded-lg mr-3" style={{ 
              backgroundColor: `${action.color}20`,
              color: action.color,
            }}>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
            <div>
              <div className="font-medium">{action.label}</div>
              <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>{action.description}</div>
            </div>
          </Link>
        ))}
      </div>
    </motion.div>
  );
};

export default QuickActionCard;
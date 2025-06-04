import React from 'react';

interface UserIconProps {
  size?: number;
  className?: string;
  variant?: 'default' | 'chat2sql' | 'small';
}

const UserIcon: React.FC<UserIconProps> = ({ size = 32, className = '', variant = 'default' }) => {
  const getStyles = () => {
    switch (variant) {
      case 'chat2sql':
        return {
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          boxShadow: '0 4px 15px rgba(102, 126, 234, 0.3), 0 2px 8px rgba(0, 0, 0, 0.1)',
          border: '2px solid rgba(255, 255, 255, 0.2)',
        };
      case 'small':
        return {
          background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
          boxShadow: '0 2px 8px rgba(79, 172, 254, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.15)',
        };
      default:
        return {
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          boxShadow: '0 6px 20px rgba(102, 126, 234, 0.4), 0 4px 12px rgba(0, 0, 0, 0.15)',
          border: '3px solid rgba(255, 255, 255, 0.25)',
        };
    }
  };

  return (
    <div 
      className={`flex items-center justify-center rounded-full transition-all duration-300 hover:scale-105 hover:shadow-lg ${className}`}
      style={{ 
        width: size, 
        height: size,
        minWidth: size,
        minHeight: size,
        ...getStyles()
      }}
    >
      <svg
        width={size * 0.55}
        height={size * 0.55}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="drop-shadow-sm"
      >
        <path
          d="M12 12C14.7614 12 17 9.76142 17 7C17 4.23858 14.7614 2 12 2C9.23858 2 7 4.23858 7 7C7 9.76142 9.23858 12 12 12Z"
          fill="white"
          fillOpacity="0.95"
        />
        <path
          d="M12 14C7.58172 14 4 17.5817 4 22H20C20 17.5817 16.4183 14 12 14Z"
          fill="white"
          fillOpacity="0.95"
        />
      </svg>
    </div>
  );
};

export default UserIcon;
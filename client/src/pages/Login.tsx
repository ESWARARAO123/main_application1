import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import LoadingScreen from '../components/LoadingScreen';
import { motion, AnimatePresence } from 'framer-motion';

// Define animations
const glowAnimation = {
  glow: {
    boxShadow: [
      '0 0 5px rgba(99, 102, 241, 0.5)',
      '0 0 20px rgba(99, 102, 241, 0.8)',
      '0 0 5px rgba(99, 102, 241, 0.5)',
    ],
    transition: {
      duration: 0.8,
      repeat: 1,
    },
  },
};

const floatingAnimation = {
  initial: { y: 0 },
  animate: {
    y: [0, -10, 0],
    transition: {
      duration: 4,
      repeat: Infinity,
      ease: "easeInOut"
    }
  }
};

// SVG Background Elements (BlobSVG)
const BlobSVG = () => (
  <div
    style={{
      position: 'absolute',
      width: '100%',
      height: '100%',
      overflow: 'hidden',
      zIndex: 0,
      opacity: 0.8,
      pointerEvents: 'none',
    }}
  >
    <svg width="100%" height="100%" viewBox="0 0 1000 1000" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="blob1" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{ stopColor: '#4f46e5', stopOpacity: 0.3 }} />
          <stop offset="100%" style={{ stopColor: '#7c3aed', stopOpacity: 0.2 }} />
        </linearGradient>
        <linearGradient id="blob2" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{ stopColor: '#3b82f6', stopOpacity: 0.2 }} />
          <stop offset="100%" style={{ stopColor: '#06b6d4', stopOpacity: 0.1 }} />
        </linearGradient>
      </defs>
      <motion.path
        initial={{ scale: 0.8, x: -100 }}
        animate={{
          scale: [0.8, 1.1, 0.8],
          x: [-100, 50, -100],
          y: [0, 50, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          repeatType: "reverse",
        }}
        d="M833.5,445.5Q805,641,609.5,737Q414,833,277.5,666.5Q141,500,252.5,307Q364,114,582,182Q800,250,831,375Q862,500,833.5,445.5Z"
        fill="url(#blob1)"
      />
      <motion.path
        initial={{ scale: 0.8, x: 100 }}
        animate={{
          scale: [0.8, 1.2, 0.8],
          x: [100, -50, 100],
          y: [0, -50, 0],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          repeatType: "reverse",
        }}
        d="M421.5,355Q365,460,260,421.5Q155,383,113.5,266.5Q72,150,188.5,89.5Q305,29,413,89.5Q521,150,477,275Q433,400,421.5,355Z"
        fill="url(#blob2)"
      />
    </svg>
  </div>
);

// Decorative Circle Elements
interface CircleDecorationProps {
  size: string;
  top?: string;
  left?: string;
  right?: string;
  bottom?: string;
  color: string;
  delay?: number;
}

const CircleDecoration = ({ size, top, left, right, bottom, color, delay = 0 }: CircleDecorationProps) => (
  <motion.div
    initial={{ opacity: 0, scale: 0 }}
    animate={{ opacity: 1, scale: 1 }}
    transition={{ duration: 0.8, delay }}
    style={{
      position: 'absolute',
      width: size,
      height: size,
      borderRadius: '50%',
      background: color,
      top,
      left,
      right,
      bottom,
      filter: 'blur(20px)',
      zIndex: 0,
    }}
  />
);

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { login, user, loading } = useAuth();
  const { currentTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // Inject custom CSS for animated gradient borders and input focus effects
    const style = document.createElement('style');
    style.innerHTML = `
      @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
      }

      .animated-input:focus {
        border-color: transparent !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.6) !important;
        transition: all 0.3s ease;
      }

      .animated-gradient-border {
        position: relative;
        border-radius: 0.5rem;
      }

      .animated-gradient-border::before {
        content: "";
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #4f46e5, #7c3aed, #3b82f6, #06b6d4);
        background-size: 400% 400%;
        animation: gradientAnimation 6s ease infinite;
        border-radius: 0.6rem;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
      }

      .animated-gradient-border:focus-within::before {
        opacity: 1;
      }
    `;
    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, []);

  useEffect(() => {
    if (!loading && user) {
      const from = location.state?.from || '/dashboard';
      navigate(from, { replace: true });
    }
  }, [user, loading, navigate, location]);

  if (loading) {
    return <LoadingScreen />;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      await login(username, password);
    } catch (err: any) {
      console.error('Login error:', err);
      setError(err.response?.data?.error || 'Failed to login. Please try again.');
      setIsLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden"
      style={{
        background: 'linear-gradient(to-br, #f8fafc, #f1f5f9, #e2e8f0)',
      }}
    >
      {/* Animated Background Elements */}
      <BlobSVG />

      {/* Decorative Circles */}
      <CircleDecoration size="300px" top="-50px" left="-100px" color="rgba(99, 102, 241, 0.1)" delay={0.2} />
      <CircleDecoration size="200px" bottom="-30px" right="-50px" color="rgba(59, 130, 246, 0.1)" delay={0.4} />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className="w-full max-w-md relative z-10"
        style={{ maxWidth: '500px' }}
      >
        {/* Header Section with Logo and Title */}
        <div className="text-center mb-6" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <motion.div
            variants={floatingAnimation}
            initial="initial"
            animate="animate"
            className="mb-4"
          >
            <div
              className="mx-auto"
              style={{
                width: '90px',
                height: '90px',
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #e11d48, #1e3a8a)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 10px 30px -10px rgba(225, 29, 72, 0.4), inset 0 2px 10px rgba(255, 255, 255, 0.3)',
                position: 'relative',
                border: '3px solid rgba(255, 255, 255, 0.8)',
              }}
            >
              <span
                style={{
                  fontSize: '3.5rem',
                  fontWeight: 'bold',
                  color: 'white',
                  textShadow: '0 2px 5px rgba(0,0,0,0.3)',
                }}
              >
                P
              </span>
              
              {/* Shine effect */}
              <div style={{
                position: 'absolute',
                top: '5px',
                left: '10px',
                width: '15px',
                height: '15px',
                borderRadius: '50%',
                background: 'rgba(255, 255, 255, 0.7)',
                filter: 'blur(2px)',
              }}></div>
            </div>
          </motion.div>

          {/* Pinnacleflow Title */}
          <motion.h1
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="text-3xl md:text-4xl font-extrabold"
            style={{ letterSpacing: 'tight' }}
          >
            <span style={{ color: '#e11d48' }}>Pinnacle</span>
            {/* <span style={{ color: '#1e3a8a' }}>flow</span> */}
            <span style={{ 
              display: 'inline-block',
              marginLeft: '0.2rem',
              fontWeight: '700',
              color: '#e11d48'
            }}>AI</span>
          </motion.h1>
        </div>

        <div
          className="rounded-xl shadow-2xl transition-all duration-300 hover:shadow-3xl relative overflow-hidden"
          style={{
            background: 'white',
            border: '1px solid #e2e8f0',
            padding: '2.5rem',
            transition: 'all 0.3s ease',
          }}
        >
          {/* Top Gradient Border */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '6px',
              background: 'linear-gradient(to-r, #4f46e5, #7c3aed, #3b82f6, #06b6d4)',
            }}
          />

          {/* Form Content */}
          <div className="relative z-1" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            {/* Subtitle Section */}
            <div className="text-center" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <motion.h2
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="text-2xl font-bold"
                style={{ color: '#1f2937' }}
              >
                Welcome Back
              </motion.h2>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                className="mt-3"
                style={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  gap: '0.2rem'
                }}
              >
                <span style={{ 
                  fontSize: '1.8rem', 
                  fontWeight: '700',
                  color: '#e11d48',
                  letterSpacing: '-0.01em',
                  lineHeight: '1.2'
                }}>
                  Sierra
                </span>
                <span style={{ 
                  fontSize: '1.8rem', 
                  fontWeight: '700',
                  color: '#1e3a8a',
                  letterSpacing: '-0.01em',
                  lineHeight: '1.2'
                }}>
                  Edge
                </span>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  marginLeft: '0.3rem',
                  position: 'relative'
                }}>
                  <span style={{ 
                    fontSize: '1.8rem', 
                    fontWeight: '700',
                    color: '#e11d48'
                  }}>
                    A
                  </span>
                  <span style={{ 
                    fontSize: '1.8rem', 
                    fontWeight: '700',
                    color: '#1e3a8a'
                  }}>
                    I
                  </span>
                </div>
              </motion.div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="flex items-center gap-2 p-4 rounded-lg border"
                    style={{
                      background: 'rgba(254, 242, 242, 1)',
                      borderLeft: '4px solid #ef4444',
                      color: '#ef4444',
                      fontSize: '0.875rem',
                      fontWeight: 'medium',
                      textAlign: 'center',
                    }}
                  >
                    <span>{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Username Input */}
              <div className="relative animated-gradient-border">
                <input
                  id="username"
                  type="text"
                  required
                  autoComplete="username"
                  className="peer w-full px-4 py-4 rounded-lg transition-all duration-300 animated-input"
                  style={{
                    backgroundColor: 'white',
                    color: '#1f2937',
                    border: '2px solid #e5e7eb',
                    fontSize: '1rem',
                  }}
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  disabled={isLoading}
                />
                <label
                  htmlFor="username"
                  className={`absolute left-4 px-1 text-sm transition-all duration-300 ${
                    username ? '-top-2.5 text-xs' : 'top-4 text-base'
                  } peer-focus:-top-2.5 peer-focus:text-xs`}
                  style={{
                    backgroundColor: 'white',
                    color: '#6b7280',
                    fontWeight: 'medium',
                  }}
                >
                  Username
                </label>
              </div>

              {/* Password Input */}
              <div className="relative animated-gradient-border">
                <input
                  id="password"
                  type="password"
                  required
                  autoComplete="current-password"
                  className="peer w-full px-4 py-4 rounded-lg transition-all duration-300 animated-input"
                  style={{
                    backgroundColor: 'white',
                    color: '#1f2937',
                    border: '2px solid #e5e7eb',
                    fontSize: '1rem',
                  }}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isLoading}
                />
                <label
                  htmlFor="password"
                  className={`absolute left-4 px-1 text-sm transition-all duration-300 ${
                    password ? '-top-2.5 text-xs' : 'top-4 text-base'
                  } peer-focus:-top-2.5 peer-focus:text-xs`}
                  style={{
                    backgroundColor: 'white',
                    color: '#6b7280',
                    fontWeight: 'medium',
                  }}
                >
                  Password
                </label>
              </div>

              {/* Submit Button */}
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                style={{ width: '100%' }}
              >
                <motion.button
                  type="submit"
                  className="w-full py-4 rounded-lg font-medium relative overflow-hidden"
                  style={{
                    background: 'linear-gradient(135deg, #e11d48, #1e3a8a)',
                    color: '#ffffff',
                    boxShadow: '0 4px 15px rgba(225, 29, 72, 0.3)',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                  }}
                  disabled={isLoading}
                  whileTap={{
                    scale: 0.98,
                    ...glowAnimation.glow,
                  }}
                >
                  {isLoading ? (
                    <span className="flex items-center justify-center">
                      <motion.span
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                        className="inline-block w-5 h-5 border-2 border-white border-t-transparent rounded-full mr-2"
                        style={{ borderColor: 'white', borderTopColor: 'transparent' }}
                      />
                      Signing in...
                    </span>
                  ) : (
                    <span className="flex items-center justify-center">
                      <span style={{ fontSize: '1.1rem', letterSpacing: '0.03em', fontWeight: 'bold' }}>Sign in</span>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </span>
                  )}
                </motion.button>
              </motion.div>



              {/* Divider with Text */}
              <div className="relative py-4">
                <div
                  style={{
                    borderTop: '1px solid #e5e7eb',
                  }}
                />
                <span
                  style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    background: 'white',
                    padding: '0 0.75rem',
                    fontSize: '0.75rem',
                    color: '#6b7280',
                  }}
                >
                  Powered by advanced AI
                </span>
              </div>

              {/* Feature Highlights */}
              <div className="flex justify-center gap-2 flex-wrap">
                {["Smart Analytics", "AI Insights", "Real-time Data"].map((feature, index) => (
                  <motion.div
                    key={feature}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + (index * 0.1) }}
                  >
                    <div
                      style={{
                        padding: '0.25rem 0.75rem',
                        background: '#f9fafb',
                        color: '#4b5563',
                        fontSize: '0.75rem',
                        fontWeight: 'medium',
                        borderRadius: '9999px',
                        border: '1px solid #e5e7eb',
                      }}
                    >
                      {feature}
                    </div>
                  </motion.div>
                ))}
              </div>
            </form>
          </div>
        </div>

        {/* Footer Text */}
        <div
          className="mt-6 text-center"
          style={{
            fontSize: '0.75rem',
            color: '#6b7280',
          }}
        >
          © {new Date().getFullYear()} SierraEdge AI. All rights reserved.
        </div>
      </motion.div>
    </div>
  );
}
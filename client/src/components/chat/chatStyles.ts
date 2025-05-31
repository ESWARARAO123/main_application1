/**
 * Chat UI Styles
 * This file contains styling properties for the chat components
 */

// Message bubble styles
export const messageBubbleStyles = {
  // AI message styles
  ai: {
    container: {
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'flex-start',
      marginBottom: '1rem',
    },
    avatar: {
      width: '2rem',
      height: '2rem',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'white',
      marginRight: '0.75rem',
      flexShrink: 0,
      background: 'linear-gradient(to right, var(--color-primary), var(--color-secondary))',
      fontSize: '0.75rem',
      fontWeight: 600,
    },
    header: {
      display: 'flex',
      alignItems: 'center',
      marginBottom: '0.5rem',
    },
    content: {
      color: 'var(--color-text)',
      backgroundColor: 'var(--color-surface-light)',
      borderRadius: '0.75rem 0.75rem 0.75rem 0',
      padding: '0.75rem 1rem',
      maxWidth: '85%',
      marginLeft: '2.75rem',
      boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
      overflowWrap: 'break-word' as const,
      wordWrap: 'break-word' as const,
      wordBreak: 'break-word' as const,
    },
    timestamp: {
      fontSize: '0.75rem',
      color: 'var(--color-text-muted)',
    },
    codeBlock: {
      borderRadius: '0.5rem',
      marginTop: '0.5rem',
      marginBottom: '0.5rem',
      overflow: 'hidden',
    },
    inlineCode: {
      backgroundColor: 'var(--color-surface-dark)',
      padding: '0.1rem 0.3rem',
      borderRadius: '0.25rem',
      fontFamily: 'monospace',
      fontSize: '0.9em',
    },
  },

  // User message styles
  user: {
    container: {
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'flex-end',
      marginBottom: '1rem',
    },
    avatar: {
      width: '2rem',
      height: '2rem',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      marginRight: '0.75rem',
      flexShrink: 0,
      backgroundColor: 'var(--color-surface-dark)',
      color: 'var(--color-text)',
    },
    header: {
      display: 'flex',
      alignItems: 'center',
      marginBottom: '0.5rem',
      justifyContent: 'flex-end',
    },
    content: {
      color: 'white',
      backgroundColor: 'var(--color-primary)',
      borderRadius: '0.75rem 0.75rem 0 0.75rem',
      padding: '0.75rem 1rem',
      maxWidth: '85%',
      marginRight: '0',
      boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
      overflowWrap: 'break-word' as const,
      wordWrap: 'break-word' as const,
      wordBreak: 'break-word' as const,
    },
    timestamp: {
      fontSize: '0.75rem',
      color: 'var(--color-text-muted)',
    },
  },
};

// Message list styles
export const messageListStyles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    flex: 1,
    overflowY: 'auto' as const,
    padding: '1rem',
    scrollBehavior: 'smooth' as const,
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    padding: '0 1rem',
  },
  emptyIcon: {
    width: '4rem',
    height: '4rem',
    marginBottom: '1rem',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
  },
  loadMoreButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '0.5rem 1rem',
    borderRadius: '0.5rem',
    backgroundColor: 'var(--color-surface-dark)',
    color: 'var(--color-text-muted)',
    fontSize: '0.875rem',
    cursor: 'pointer',
    border: 'none',
    margin: '1rem auto',
    transition: 'all 0.2s ease',
  },
  loadingIndicator: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '1rem',
  },
  typingIndicator: {
    display: 'flex',
    alignItems: 'center',
    marginLeft: '2.75rem',
    marginTop: '0.5rem',
  },
  typingDot: {
    width: '0.5rem',
    height: '0.5rem',
    borderRadius: '50%',
    backgroundColor: 'var(--color-text-muted)',
    margin: '0 0.125rem',
    animation: 'bounce 1.4s infinite ease-in-out both',
  },
};

// Chat input styles
export const chatInputStyles = {
  container: {
    position: 'relative' as const,
    display: 'flex',
    flexDirection: 'column' as const,
    backgroundColor: 'var(--color-surface)',
    borderRadius: '0.75rem',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
    padding: '0.5rem',
    margin: '0 auto',
    maxWidth: '100%',
    transition: 'all 0.3s ease',
  },
  inputRow: {
    display: 'flex',
    alignItems: 'center',
    width: '100%',
  },
  input: {
    flex: 1,
    border: 'none',
    outline: 'none',
    backgroundColor: 'transparent',
    color: 'var(--color-text)',
    padding: '0.75rem 1rem',
    fontSize: '0.95rem',
    lineHeight: '1.5',
    resize: 'none' as const,
    maxHeight: '150px',
    overflowY: 'auto' as const,
  },
  sendButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '2.5rem',
    height: '2.5rem',
    borderRadius: '50%',
    backgroundColor: 'var(--color-primary)',
    color: 'white',
    border: 'none',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  },
  disabledSendButton: {
    backgroundColor: 'var(--color-surface-dark)',
    color: 'var(--color-text-muted)',
    cursor: 'not-allowed',
  },
  placeholder: {
    color: 'var(--color-text-muted)',
  },
  iconButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '2rem',
    height: '2rem',
    borderRadius: '50%',
    backgroundColor: 'transparent',
    color: 'var(--color-text)',
    border: 'none',
    padding: 0,
    marginRight: '0.5rem',
  },
  filePreviewContainer: {
    marginBottom: '0.5rem',
    width: '100%',
  },
  buttonsContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  buttonsRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    marginTop: '0.75rem',
    paddingLeft: '0.25rem',
    overflowX: 'auto' as const,
    flexWrap: 'nowrap' as const,
    scrollbarWidth: 'none' as any,
    msOverflowStyle: 'none' as any,
    justifyContent: 'flex-start',
  },
  ragToggleButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '1.5rem',
    padding: '0.4rem 0.8rem',
    fontSize: '0.8rem',
    fontWeight: 500,
    transition: 'all 0.2s ease',
    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.08)',
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    backdropFilter: 'blur(5px)',
    WebkitBackdropFilter: 'blur(5px)',
    height: '2.25rem',
    minWidth: '4.5rem',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  ragToggleEnabled: {
    backgroundColor: 'var(--color-primary-translucent)',
    color: 'var(--color-primary)',
    border: '1px solid var(--color-primary-translucent)',
  },
  ragToggleDisabled: {
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    color: 'var(--color-text-muted)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  mcpToggleButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '1.5rem',
    padding: '0.4rem 0.8rem',
    fontSize: '0.8rem',
    fontWeight: 500,
    transition: 'all 0.2s ease',
    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.08)',
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    backdropFilter: 'blur(5px)',
    WebkitBackdropFilter: 'blur(5px)',
    height: '2.25rem',
    minWidth: '4.5rem',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  mcpToggleEnabled: {
    backgroundColor: 'var(--color-success-translucent)',
    color: 'var(--color-success)',
    border: '1px solid var(--color-success-translucent)',
  },
  mcpToggleDisabled: {
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    color: 'var(--color-text-muted)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
};

// Markdown content styles
export const markdownStyles = {
  container: {
    fontSize: '0.95rem',
    lineHeight: '1.6',
  },
  paragraph: {
    marginTop: '0.5rem',
    marginBottom: '0.5rem',
  },
  heading: {
    fontWeight: 600,
    marginTop: '1.5rem',
    marginBottom: '0.75rem',
  },
  h1: {
    fontSize: '1.5rem',
  },
  h2: {
    fontSize: '1.3rem',
  },
  h3: {
    fontSize: '1.1rem',
  },
  list: {
    paddingLeft: '1.5rem',
    marginTop: '0.5rem',
    marginBottom: '0.5rem',
  },
  listItem: {
    marginTop: '0.25rem',
    marginBottom: '0.25rem',
  },
  link: {
    color: 'var(--color-primary)',
    textDecoration: 'underline',
  },
  blockquote: {
    borderLeft: '4px solid var(--color-border)',
    paddingLeft: '1rem',
    fontStyle: 'italic',
    margin: '1rem 0',
  },
  table: {
    borderCollapse: 'collapse' as const,
    width: '100%',
    marginTop: '1rem',
    marginBottom: '1rem',
  },
  tableCell: {
    border: '1px solid var(--color-border)',
    padding: '0.5rem',
  },
  tableHeader: {
    backgroundColor: 'var(--color-surface-dark)',
    fontWeight: 600,
  },
};

// Animation keyframes
export const animations = {
  bounce: `
    @keyframes bounce {
      0%, 80%, 100% { transform: translateY(0); }
      40% { transform: translateY(-6px); }
    }
  `,
  fadeIn: `
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
  `,
  slideIn: `
    @keyframes slideIn {
      from { transform: translateY(10px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
  `,
};

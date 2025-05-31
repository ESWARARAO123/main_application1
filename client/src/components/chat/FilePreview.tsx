import React from 'react';
import { DocumentIcon, DocumentTextIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface FilePreviewProps {
  file: File;
  onRemove: () => void;
  uploadProgress?: number;
}

const FilePreview: React.FC<FilePreviewProps> = ({
  file,
  onRemove,
  uploadProgress
}) => {
  // Format file size to human-readable format
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Get file icon based on type
  const getFileIcon = () => {
    const fileType = file.type;
    if (fileType === 'application/pdf') {
      return <DocumentTextIcon className="h-6 w-6 text-red-500" />;
    } else if (fileType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
      return <DocumentTextIcon className="h-6 w-6 text-blue-500" />;
    } else if (fileType === 'text/plain') {
      return <DocumentTextIcon className="h-6 w-6 text-gray-500" />;
    } else {
      return <DocumentIcon className="h-6 w-6 text-gray-500" />;
    }
  };

  return (
    <div 
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '0.5rem',
        backgroundColor: 'var(--color-surface-light)',
        borderRadius: '0.5rem',
        marginBottom: '0.5rem',
        position: 'relative',
        maxWidth: '100%',
        overflow: 'hidden'
      }}
    >
      <div style={{ marginRight: '0.5rem' }}>
        {getFileIcon()}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ 
          fontWeight: 500, 
          whiteSpace: 'nowrap', 
          overflow: 'hidden', 
          textOverflow: 'ellipsis' 
        }}>
          {file.name}
        </div>
        <div style={{ 
          fontSize: '0.75rem', 
          color: 'var(--color-text-muted)' 
        }}>
          {formatFileSize(file.size)}
        </div>
        
        {/* Progress bar */}
        {uploadProgress !== undefined && uploadProgress > 0 && (
          <div style={{ 
            height: '4px', 
            backgroundColor: 'var(--color-surface-dark)', 
            borderRadius: '2px',
            marginTop: '0.25rem',
            overflow: 'hidden'
          }}>
            <div style={{ 
              height: '100%', 
              width: `${uploadProgress}%`, 
              backgroundColor: 'var(--color-primary)',
              transition: 'width 0.3s ease'
            }} />
          </div>
        )}
      </div>
      
      {/* Remove button */}
      {uploadProgress === undefined || uploadProgress === 0 ? (
        <button
          type="button"
          onClick={onRemove}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: '0.25rem',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--color-text-muted)',
            transition: 'all 0.2s ease',
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--color-surface-dark)';
            e.currentTarget.style.color = 'var(--color-text)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = 'var(--color-text-muted)';
          }}
          aria-label="Remove file"
          title="Remove file"
        >
          <XMarkIcon className="h-4 w-4" />
        </button>
      ) : null}
    </div>
  );
};

export default FilePreview;

import React, { useRef, useState } from 'react';
import { PaperClipIcon } from '@heroicons/react/24/outline';
import { chatInputStyles } from './chatStyles';

interface FileUploadButtonProps {
  onFileSelect: (file: File) => void;
  onAutoUpload?: (file: File) => void; // New prop for automatic upload
  isLoading: boolean;
  acceptedFileTypes?: string; // e.g., ".pdf,.docx,.txt"
  disabled?: boolean;
  autoUpload?: boolean; // New prop to control auto-upload behavior
}

const FileUploadButton: React.FC<FileUploadButtonProps> = ({
  onFileSelect,
  onAutoUpload,
  isLoading,
  acceptedFileTypes = ".pdf,.docx,.txt",
  disabled = false,
  autoUpload = true // Default to auto-upload
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isHovered, setIsHovered] = useState(false);

  const handleClick = () => {
    if (isLoading || disabled) return;
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      // Always notify about the selection
      onFileSelect(files[0]);

      // If auto-upload is enabled and handler is provided, call it
      if (autoUpload && onAutoUpload) {
        onAutoUpload(files[0]);
      }
    }

    // Reset the input value so the same file can be selected again if needed
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <>
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={handleFileChange}
        accept={acceptedFileTypes}
        aria-label="Upload document"
      />
      <button
        type="button"
        onClick={handleClick}
        disabled={isLoading || disabled}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        style={{
          ...chatInputStyles.ragToggleButton,
          opacity: isLoading || disabled ? 0.5 : isHovered ? 0.9 : 0.7,
          cursor: isLoading || disabled ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease',
          backgroundColor: 'rgba(255, 255, 255, 0.08)',
          color: 'var(--color-text-muted)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          width: '2.5rem',
          height: '2.5rem',
          padding: '0.5rem',
          minWidth: 'auto',
        }}
        className="hover:bg-opacity-90 transition-all"
        aria-label="Upload document"
        title="Upload document (PDF, DOCX, TXT)"
      >
        <PaperClipIcon className="h-5 w-5" />
      </button>
    </>
  );
};

export default FileUploadButton;

import React from 'react';
import {
  FolderIcon,
  DocumentIcon,
  ArrowUpTrayIcon
} from '@heroicons/react/24/outline';

export interface FileItem {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size?: number;
  permissions?: string;
  owner?: string;
  group?: string;
  lastModified?: string;
}

interface FileSystemItemProps {
  item: FileItem;
  onSelect: (item: FileItem) => void;
  onNavigate: (item: FileItem) => void;
  isSelected: boolean;
}

const FileSystemItem: React.FC<FileSystemItemProps> = ({
  item,
  onSelect,
  onNavigate,
  isSelected
}) => {
  const isDirectory = item.type === 'directory';
  
  const handleClick = () => {
    if (isDirectory) {
      onSelect(item);
    }
  };
  
  const handleDoubleClick = () => {
    if (isDirectory) {
      onNavigate(item);
    }
  };
  
  // Format file size
  const formatSize = (size?: number): string => {
    if (size === undefined) return '-';
    
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  return (
    <div
      className={`p-2 rounded-md flex items-center cursor-pointer transition-colors ${isSelected ? 'bg-opacity-20' : 'hover:bg-opacity-10'}`}
      style={{ 
        backgroundColor: isSelected ? 'var(--color-primary-translucent)' : 'transparent',
        borderColor: isSelected ? 'var(--color-primary)' : 'transparent',
      }}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
    >
      <div className="mr-3">
        {isDirectory ? (
          <FolderIcon className="h-6 w-6" style={{ color: 'var(--color-primary)' }} />
        ) : (
          <DocumentIcon className="h-6 w-6" style={{ color: 'var(--color-text-muted)' }} />
        )}
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="font-medium truncate" style={{ color: 'var(--color-text)' }}>
          {item.name}
        </div>
        
        {item.permissions && (
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            {item.permissions} {item.owner && `${item.owner}:${item.group}`}
          </div>
        )}
      </div>
      
      <div className="ml-4 text-sm text-right" style={{ color: 'var(--color-text-secondary)' }}>
        {isDirectory ? (
          <div className="flex items-center">
            <span>Directory</span>
            <ArrowUpTrayIcon className="h-4 w-4 ml-1 transform rotate-90" />
          </div>
        ) : (
          formatSize(item.size)
        )}
        
        {item.lastModified && (
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            {new Date(item.lastModified).toLocaleDateString()}
          </div>
        )}
      </div>
    </div>
  );
};

export default FileSystemItem;

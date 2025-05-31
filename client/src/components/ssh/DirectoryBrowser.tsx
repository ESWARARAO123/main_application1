import React, { useState, useEffect } from 'react';
import {
  FolderIcon,
  ArrowUpIcon,
  HomeIcon,
  ArrowPathIcon,
  ExclamationCircleIcon
} from '@heroicons/react/24/outline';
import FileSystemItem, { FileItem } from './FileSystemItem';

interface DirectoryBrowserProps {
  sshConfigId: string;
  currentPath: string;
  onPathChange: (path: string) => void;
  onSelectDirectory: (path: string) => void;
  fetchDirectoryContents: (path: string) => Promise<FileItem[]>;
}

const DirectoryBrowser: React.FC<DirectoryBrowserProps> = ({
  sshConfigId,
  currentPath,
  onPathChange,
  onSelectDirectory,
  fetchDirectoryContents
}) => {
  const [items, setItems] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<FileItem | null>(null);
  
  // Parse the current path into breadcrumb segments
  const pathSegments = currentPath.split('/').filter(segment => segment.length > 0);
  
  useEffect(() => {
    loadDirectoryContents(currentPath);
  }, [currentPath, sshConfigId]);
  
  const loadDirectoryContents = async (path: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const contents = await fetchDirectoryContents(path);
      setItems(contents);
      setSelectedItem(null);
    } catch (err: any) {
      setError(err.message || 'Failed to load directory contents');
      console.error('Error loading directory:', err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleNavigate = (item: FileItem) => {
    if (item.type === 'directory') {
      onPathChange(item.path);
    }
  };
  
  const handleSelect = (item: FileItem) => {
    if (item.type === 'directory') {
      setSelectedItem(item);
      onSelectDirectory(item.path);
    }
  };
  
  const navigateUp = () => {
    const parentPath = currentPath.split('/').slice(0, -1).join('/') || '/';
    onPathChange(parentPath);
  };
  
  const navigateHome = () => {
    onPathChange('/home');
  };
  
  const navigateToBreadcrumb = (index: number) => {
    const path = '/' + pathSegments.slice(0, index + 1).join('/');
    onPathChange(path);
  };
  
  const refreshDirectory = () => {
    loadDirectoryContents(currentPath);
  };

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center p-2 border-b" style={{ borderColor: 'var(--color-border)' }}>
        <button
          onClick={navigateUp}
          className="p-1.5 rounded-md mr-1 transition-colors"
          style={{ 
            backgroundColor: 'var(--color-surface-dark)', 
            color: 'var(--color-text-muted)'
          }}
          title="Go up one directory"
        >
          <ArrowUpIcon className="h-5 w-5" />
        </button>
        
        <button
          onClick={navigateHome}
          className="p-1.5 rounded-md mr-1 transition-colors"
          style={{ 
            backgroundColor: 'var(--color-surface-dark)', 
            color: 'var(--color-text-muted)'
          }}
          title="Go to home directory"
        >
          <HomeIcon className="h-5 w-5" />
        </button>
        
        <button
          onClick={refreshDirectory}
          className="p-1.5 rounded-md mr-3 transition-colors"
          style={{ 
            backgroundColor: 'var(--color-surface-dark)', 
            color: 'var(--color-text-muted)'
          }}
          title="Refresh directory"
        >
          <ArrowPathIcon className={`h-5 w-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
        
        {/* Breadcrumb navigation */}
        <div className="flex items-center overflow-x-auto flex-1 hide-scrollbar">
          <button
            onClick={() => onPathChange('/')}
            className="flex items-center p-1 rounded hover:bg-opacity-10"
            style={{ 
              backgroundColor: 'transparent', 
              color: 'var(--color-text-muted)'
            }}
          >
            <FolderIcon className="h-4 w-4 mr-1" />
            <span>/</span>
          </button>
          
          {pathSegments.map((segment, index) => (
            <React.Fragment key={index}>
              <span style={{ color: 'var(--color-text-muted)' }}>/</span>
              <button
                onClick={() => navigateToBreadcrumb(index)}
                className="flex items-center p-1 rounded hover:bg-opacity-10 max-w-xs"
                style={{ 
                  backgroundColor: 'transparent', 
                  color: 'var(--color-text)'
                }}
              >
                <span className="truncate">{segment}</span>
              </button>
            </React.Fragment>
          ))}
        </div>
      </div>
      
      {/* Directory contents */}
      <div className="flex-1 overflow-y-auto p-2">
        {loading && items.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <ArrowPathIcon className="h-8 w-8 animate-spin" style={{ color: 'var(--color-text-muted)' }} />
            <span className="ml-2" style={{ color: 'var(--color-text-muted)' }}>Loading directory contents...</span>
          </div>
        )}
        
        {error && (
          <div className="p-4 rounded-md flex items-start" style={{ backgroundColor: 'rgba(var(--color-error-rgb), 0.1)' }}>
            <ExclamationCircleIcon className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" style={{ color: 'var(--color-error)' }} />
            <div>
              <div className="font-medium" style={{ color: 'var(--color-error)' }}>Error loading directory</div>
              <div style={{ color: 'var(--color-text-secondary)' }}>{error}</div>
            </div>
          </div>
        )}
        
        {!loading && !error && items.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full">
            <FolderIcon className="h-12 w-12 mb-2" style={{ color: 'var(--color-text-muted)' }} />
            <span style={{ color: 'var(--color-text-muted)' }}>This directory is empty</span>
          </div>
        )}
        
        <div className="space-y-1">
          {items.map((item) => (
            <FileSystemItem
              key={item.path}
              item={item}
              onSelect={handleSelect}
              onNavigate={handleNavigate}
              isSelected={selectedItem?.path === item.path}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default DirectoryBrowser;

import React, { useState, useEffect } from 'react';
import { XMarkIcon, FolderIcon, CheckIcon } from '@heroicons/react/24/outline';
import DirectoryBrowser from './DirectoryBrowser';
import { FileItem } from './FileSystemItem';
import axios from 'axios';

interface RemoteFilesystemExplorerProps {
  sshConfigId: string;
  sshPassword: string;
  initialPath?: string;
  onPathSelect: (path: string) => void;
  onCancel: () => void;
}

const RemoteFilesystemExplorer: React.FC<RemoteFilesystemExplorerProps> = ({
  sshConfigId,
  sshPassword,
  initialPath = '/home',
  onPathSelect,
  onCancel
}) => {
  const [currentPath, setCurrentPath] = useState(initialPath);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    // Initialize with the initial path
    setCurrentPath(initialPath);
  }, [initialPath]);
  
  const fetchDirectoryContents = async (path: string): Promise<FileItem[]> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`/api/mcp/ssh/fs/list`, {
        params: { 
          sshConfigId,
          path
        },
        headers: {
          'X-SSH-Password': sshPassword // Note: In a production app, use a more secure method
        }
      });
      
      if (response.data && response.data.items) {
        return response.data.items;
      }
      
      throw new Error('Invalid response format');
    } catch (err: any) {
      const errorMessage = err.response?.data?.error || 'Failed to load directory contents';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  };
  
  const handlePathChange = (path: string) => {
    setCurrentPath(path);
  };
  
  const handleSelectDirectory = (path: string) => {
    setSelectedPath(path);
  };
  
  const handleConfirm = () => {
    if (selectedPath) {
      onPathSelect(selectedPath);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div 
        className="bg-opacity-95 rounded-lg shadow-xl flex flex-col w-full max-w-4xl h-[80vh]"
        style={{ backgroundColor: 'var(--color-surface-light)' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b" style={{ borderColor: 'var(--color-border)' }}>
          <div className="flex items-center">
            <FolderIcon className="h-6 w-6 mr-2" style={{ color: 'var(--color-primary)' }} />
            <h2 className="text-xl font-medium" style={{ color: 'var(--color-text)' }}>
              Select Installation Directory
            </h2>
          </div>
          
          <button
            onClick={onCancel}
            className="p-1 rounded-full hover:bg-opacity-10"
            style={{ 
              backgroundColor: 'transparent', 
              color: 'var(--color-text-muted)'
            }}
          >
            <XMarkIcon className="h-6 w-6" />
          </button>
        </div>
        
        {/* Directory Browser */}
        <div className="flex-1 overflow-hidden">
          <DirectoryBrowser
            sshConfigId={sshConfigId}
            currentPath={currentPath}
            onPathChange={handlePathChange}
            onSelectDirectory={handleSelectDirectory}
            fetchDirectoryContents={fetchDirectoryContents}
          />
        </div>
        
        {/* Footer */}
        <div className="p-4 border-t flex items-center justify-between" style={{ borderColor: 'var(--color-border)' }}>
          <div style={{ color: 'var(--color-text-secondary)' }}>
            {selectedPath ? (
              <div className="flex items-center">
                <span className="mr-1">Selected:</span>
                <span className="font-mono bg-opacity-30 px-2 py-0.5 rounded" style={{ backgroundColor: 'var(--color-surface-dark)' }}>
                  {selectedPath}
                </span>
              </div>
            ) : (
              <span>Select a directory to install MCP</span>
            )}
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={onCancel}
              className="px-4 py-2 rounded"
              style={{
                backgroundColor: 'var(--color-surface-dark)',
                color: 'var(--color-text)'
              }}
            >
              Cancel
            </button>
            
            <button
              onClick={handleConfirm}
              disabled={!selectedPath}
              className="px-4 py-2 rounded flex items-center"
              style={{
                backgroundColor: selectedPath ? 'var(--color-primary)' : 'var(--color-primary-muted)',
                color: 'white',
                cursor: selectedPath ? 'pointer' : 'not-allowed'
              }}
            >
              <CheckIcon className="h-5 w-5 mr-1" />
              Select Directory
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RemoteFilesystemExplorer;

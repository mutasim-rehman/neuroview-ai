import React, { useCallback, useState } from 'react';
import { UploadCloud, FileType, Play } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect }) => {
  const [isLoadingDemo, setIsLoadingDemo] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Validate file type
  const isValidFileType = (file: File): boolean => {
    const validExtensions = ['.nii', '.nii.gz'];
    const fileName = file.name.toLowerCase();
    return validExtensions.some(ext => fileName.endsWith(ext));
  };

  // Handle file selection with validation
  const handleFile = useCallback((file: File) => {
    setError(null);
    
    if (!isValidFileType(file)) {
      setError('Invalid file type. Please upload a .nii or .nii.gz file.');
      return;
    }

    if (file.size === 0) {
      setError('File is empty. Please select a valid NIfTI file.');
      return;
    }

    // Check file size (warn if very large, but don't block)
    const maxSize = 500 * 1024 * 1024; // 500MB
    if (file.size > maxSize) {
      console.warn(`Large file detected: ${(file.size / 1024 / 1024).toFixed(2)}MB. Processing may take longer.`);
    }

    try {
      onFileSelect(file);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to process file';
      setError(errorMessage);
      console.error('File processing error:', err);
    }
  }, [onFileSelect]);

  const handleLoadDemo = async () => {
    setIsLoadingDemo(true);
    setError(null);
    try {
      const response = await fetch('/demo.nii');
      if (!response.ok) {
        throw new Error(`Failed to load demo file: ${response.status} ${response.statusText}`);
      }
      const blob = await response.blob();
      if (blob.size === 0) {
        throw new Error('Demo file is empty');
      }
      const file = new File([blob], 'demo.nii', { type: 'application/octet-stream' });
      handleFile(file);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load demo file';
      console.error('Failed to load demo file:', error);
      setError(errorMessage);
    } finally {
      setIsLoadingDemo(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
      // Reset input so same file can be selected again
      e.target.value = '';
    }
  };

  return (
    <div 
      className={`w-full h-96 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center p-10 transition-all group cursor-pointer ${
        isDragging 
          ? 'border-emerald-500 bg-emerald-900/20' 
          : 'border-zinc-700 hover:border-emerald-500/50 bg-zinc-900/50 hover:bg-zinc-900'
      }`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
        <div className={`w-20 h-20 rounded-full flex items-center justify-center mb-6 transition-all ${
          isDragging 
            ? 'bg-emerald-600 scale-110' 
            : 'bg-zinc-800 group-hover:scale-110'
        }`}>
            <UploadCloud size={40} className={`transition-colors ${
              isDragging 
                ? 'text-white' 
                : 'text-zinc-400 group-hover:text-emerald-400'
            }`} />
        </div>
        <h3 className={`text-xl font-semibold mb-2 transition-colors ${
          isDragging ? 'text-emerald-300' : 'text-zinc-200'
        }`}>
          {isDragging ? 'Drop file here' : 'Upload Scan File'}
        </h3>
        <p className="text-zinc-500 text-center max-w-sm mb-6">
            Drag and drop your <code className="bg-zinc-800 px-1 rounded text-emerald-400">.nii</code> or <code className="bg-zinc-800 px-1 rounded text-emerald-400">.nii.gz</code> file here to visualize it instantly in the browser.
        </p>
        
        <div className="flex gap-4">
            <label className="relative cursor-pointer">
                <input type="file" className="hidden" accept=".nii,.nii.gz" onChange={handleChange} />
                <span className="px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium transition shadow-lg shadow-emerald-900/20 inline-block">
                    Browse Files
                </span>
            </label>
            
            <button
                onClick={handleLoadDemo}
                disabled={isLoadingDemo}
                className="px-6 py-3 bg-zinc-700 hover:bg-zinc-600 text-white rounded-lg font-medium transition shadow-lg shadow-zinc-900/20 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                <Play size={16} className={isLoadingDemo ? 'animate-pulse' : ''} />
                {isLoadingDemo ? 'Loading...' : 'Load Demo'}
            </button>
        </div>
        {error && (
            <div className="mt-4 px-4 py-2 bg-red-900/20 border border-red-700/50 rounded-lg text-red-400 text-sm">
                {error}
            </div>
        )}
        
        <div className="mt-8 flex items-center gap-6 text-zinc-600 text-sm">
            <span className="flex items-center gap-2"><FileType size={14}/> Secure Client-side Parsing</span>
            <span className="flex items-center gap-2"><FileType size={14}/> No Upload to Server</span>
        </div>
    </div>
  );
};

export default FileUpload;

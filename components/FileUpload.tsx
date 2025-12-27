import React, { useCallback, useState } from 'react';
import { UploadCloud, FileType, Play } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect }) => {
  const [isLoadingDemo, setIsLoadingDemo] = useState(false);

  const handleLoadDemo = async () => {
    setIsLoadingDemo(true);
    try {
      const response = await fetch('/demo.nii');
      const blob = await response.blob();
      const file = new File([blob], 'demo.nii', { type: 'application/octet-stream' });
      onFileSelect(file);
    } catch (error) {
      console.error('Failed to load demo file:', error);
    } finally {
      setIsLoadingDemo(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div 
      className="w-full h-96 border-2 border-dashed border-zinc-700 hover:border-emerald-500/50 rounded-2xl flex flex-col items-center justify-center p-10 transition-all bg-zinc-900/50 hover:bg-zinc-900 group cursor-pointer"
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
    >
        <div className="w-20 h-20 bg-zinc-800 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
            <UploadCloud size={40} className="text-zinc-400 group-hover:text-emerald-400 transition-colors" />
        </div>
        <h3 className="text-xl font-semibold text-zinc-200 mb-2">Upload Scan File</h3>
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
        
        <div className="mt-8 flex items-center gap-6 text-zinc-600 text-sm">
            <span className="flex items-center gap-2"><FileType size={14}/> Secure Client-side Parsing</span>
            <span className="flex items-center gap-2"><FileType size={14}/> No Upload to Server</span>
        </div>
    </div>
  );
};

export default FileUpload;

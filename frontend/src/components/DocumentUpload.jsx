import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle2, AlertCircle, Loader2, FileText } from 'lucide-react';

export default function DocumentUpload({ onUploadSuccess }){
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState(null); // { type: 'success' | 'error', message: string }
  const fileInputRef = useRef(null);

  const handleUpload = async (file) => {
    if (!file) return;

    // Validate extension
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'txt', 'md'].includes(ext)) {
      setStatus({ type: 'error', message: 'Unsupported format. Use .pdf, .txt, or .md' });
      return;
    }

    setIsUploading(true);
    setStatus(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      setStatus({
        type: 'success',
        message: `Indexed "${data.filename}" (${data.chunks_created} chunks, ${data.pages_parsed} pages)`
      });

      if (onUploadSuccess) onUploadSuccess(data);
    } catch (err) {
      setStatus({ type: 'error', message: err.message || 'Error uploading file' });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
      <h2 className="text-lg font-semibold text-slate-100 mb-2 flex items-center gap-2">
        <FileText className="w-5 h-5 text-indigo-400" />
        Ingest Knowledge Base
      </h2>
      <p className="text-xs text-slate-400 mb-4">
        Upload PDF, Markdown, or Plain Text files for real-time parsing, chunking, and vector indexing.
      </p>

      {/* Drag & Drop Box */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-indigo-500 bg-indigo-500/10'
            : 'border-slate-700 hover:border-slate-600 bg-slate-950/50'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.md"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
        />

        {isUploading ? (
          <div className="flex flex-col items-center gap-2 text-indigo-400">
            <Loader2 className="w-8 h-8 animate-spin" />
            <span className="text-sm font-medium">Processing, chunking, and embedding vectors...</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <UploadCloud className="w-9 h-9 text-slate-400 hover:text-indigo-400 transition-colors" />
            <div className="text-sm font-medium text-slate-200">
              Click to browse or drag & drop documents here
            </div>
            <div className="text-xs text-slate-500">Supported: PDF, TXT, MD</div>
          </div>
        )}
      </div>

      {/* Status Indicators */}
      {status && (
        <div
          className={`mt-4 p-3 rounded-lg flex items-center gap-2 text-xs ${
            status.type === 'success'
              ? 'bg-emerald-950/60 border border-emerald-800 text-emerald-300'
              : 'bg-rose-950/60 border border-rose-800 text-rose-300'
          }`}
        >
          {status.type === 'success' ? (
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
          ) : (
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          )}
          <span>{status.message}</span>
        </div>
      )}
    </div>
  );
}
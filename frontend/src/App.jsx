import React, { useState, useEffect } from 'react';
import DocumentUpload from './components/DocumentUpload';
import ChatInterface from './components/ChatInterface';
import { Database, ShieldCheck, Cpu } from 'lucide-react';

export default function App(){
  const [healthStatus, setHealthStatus] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then((res) => res.json())
      .then((data) => setHealthStatus(data))
      .catch(() => setHealthStatus({ status: 'offline' }));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 text-white p-2 rounded-lg shadow-lg shadow-indigo-600/30">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-base font-bold text-slate-100 leading-tight">Enterprise RAG Engine</h1>
              <p className="text-xs text-slate-400">Multimodal Hybrid Search & Citations</p>
            </div>
          </div>

          {/* System Badge */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-xs bg-slate-800/80 border border-slate-700 px-3 py-1.5 rounded-lg">
              <Database className="w-3.5 h-3.5 text-indigo-400" />
              <span>Chunks: <strong className="text-indigo-300">{healthStatus?.indexed_chunks ?? 0}</strong></span>
            </div>
            <div className="flex items-center gap-1.5 text-xs bg-emerald-950/80 border border-emerald-800 text-emerald-400 px-3 py-1.5 rounded-lg">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Backend Ready</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload & Ingestion (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          <DocumentUpload onUploadSuccess={() => {
            fetch('http://localhost:8000/health')
              .then((res) => res.json())
              .then((data) => setHealthStatus(data));
          }} />
        </div>

        {/* Right Column: Chat Interface (8 cols) */}
        <div className="lg:col-span-8">
          <ChatInterface />
        </div>
      </main>
    </div>
  );
}
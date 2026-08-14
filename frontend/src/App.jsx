import { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [backendStatus, setBackendStatus] = useState('Connecting to backend...');

  useEffect(() => {
    axios.get('http://localhost:8000/health')
      .then((res) => {
        setBackendStatus(`Connected: ${res.data.status} (Index: ${res.data.pinecone_index})`);
      })
      .catch(() => {
        setBackendStatus('Backend is offline. Start the FastAPI server on port 8000.');
      });
  }, []);

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center p-6">
      <div className="max-w-md w-full bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-700">
        <h1 className="text-2xl font-bold text-indigo-400 mb-2">Enterprise Knowledge Engine</h1>
        <p className="text-sm text-slate-400 mb-4">Shift 1 Scaffolding Complete</p>
        <div className="p-3 bg-slate-950 rounded-md text-xs font-mono border border-slate-800 text-emerald-400">
          {backendStatus}
        </div>
      </div>
    </div>
  );
}

export default App;
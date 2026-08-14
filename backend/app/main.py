import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env
load_dotenv()

app = FastAPI(title="Enterprise Multimodal Knowledge Engine")

# Enable CORS for React Frontend (default Vite dev server on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Enterprise Knowledge Engine API is running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pinecone_index": os.getenv("PINECONE_INDEX_NAME", "Not Set"),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here")
    }
"""
FastAPI RAG Application
Complete API wrapper for RAG system with chat and knowledge base management
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os
import shutil
from pathlib import Path
from datetime import datetime
import json

from rag_pipeline_ollama import DocumentLoader, TextChunker, VectorStore, OllamaRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
KNOWLEDGE_BASE_DIR = "./knowledge_base"
INDEX_PATH = "./rag_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:4b"  # Change as needed

# Ensure directories exist
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="RAG Knowledge Base API",
    description="API for RAG system with chat and document management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
vector_store: Optional[VectorStore] = None
rag_system: Optional[OllamaRAG] = None
is_initialized = False

# ==================== Pydantic Models ====================

class ChatRequest(BaseModel):
    question: str = Field(..., description="Question to ask the RAG system")
    k: int = Field(3, ge=1, le=10, description="Number of context chunks to retrieve")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(512, ge=50, le=2048, description="Maximum tokens to generate")

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    chunks_retrieved: int
    model: str
    timestamp: str

class DocumentInfo(BaseModel):
    filename: str
    filepath: str
    size_bytes: int
    uploaded_at: str
    file_type: str

class SystemStatus(BaseModel):
    status: str
    is_initialized: bool
    model: str
    documents_count: int
    chunks_count: int
    embedding_model: str
    knowledge_base_dir: str

class RebuildResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    message: str

# ==================== Helper Functions ====================

def initialize_rag_system():
    """Initialize or reinitialize the RAG system"""
    global vector_store, rag_system, is_initialized
    
    try:
        logger.info("Initializing RAG system...")
        
        # Check if index exists
        if not os.path.exists(f"{INDEX_PATH}_index.faiss"):
            logger.warning("No index found. Please rebuild the knowledge base.")
            is_initialized = False
            return False
        
        # Load vector store
        vector_store = VectorStore(embedding_model_name=EMBEDDING_MODEL)
        vector_store.load(INDEX_PATH)
        logger.info(f"✓ Loaded vector store with {len(vector_store.chunks)} chunks")
        
        # Initialize RAG
        rag_system = OllamaRAG(
            model_name=OLLAMA_MODEL,
            vector_store=vector_store
        )
        logger.info(f"✓ RAG system initialized with model: {OLLAMA_MODEL}")
        
        is_initialized = True
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        is_initialized = False
        return False

def rebuild_index_sync():
    """Rebuild the vector index from knowledge base"""
    global vector_store, rag_system
    
    try:
        logger.info("Starting index rebuild...")
        
        # Load documents
        loader = DocumentLoader()
        documents = loader.load_documents(KNOWLEDGE_BASE_DIR)
        
        if not documents:
            logger.warning("No documents found in knowledge base")
            return {"documents_processed": 0, "chunks_created": 0}
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Chunk documents
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Build vector store
        vector_store = VectorStore(embedding_model_name=EMBEDDING_MODEL)
        vector_store.build_index(chunks)
        vector_store.save(INDEX_PATH)
        logger.info("✓ Index saved")
        
        # Reinitialize RAG system
        initialize_rag_system()
        
        return {
            "documents_processed": len(documents),
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        raise

def get_documents_list() -> List[DocumentInfo]:
    """Get list of all documents in knowledge base"""
    documents = []
    kb_path = Path(KNOWLEDGE_BASE_DIR)
    
    for file_path in kb_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt']:
            stat = file_path.stat()
            documents.append(DocumentInfo(
                filename=file_path.name,
                filepath=str(file_path),
                size_bytes=stat.st_size,
                uploaded_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                file_type=file_path.suffix.lower()[1:]  # Remove the dot
            ))
    
    return documents

# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    logger.info("=" * 50)
    logger.info("Starting RAG API Server...")
    logger.info("=" * 50)
    
    # Try to initialize if index exists
    initialize_rag_system()
    
    if not is_initialized:
        logger.warning("⚠️  RAG system not initialized. Upload documents and rebuild index.")
    else:
        logger.info("✓ RAG API Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG API Server...")

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Knowledge Base API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "/chat",
            "upload": "/documents/upload",
            "list_documents": "/documents",
            "delete_document": "/documents/{filename}",
            "rebuild_index": "/index/rebuild",
            "status": "/status"
        }
    }

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status and statistics"""
    documents = get_documents_list()
    
    return SystemStatus(
        status="ready" if is_initialized else "not_initialized",
        is_initialized=is_initialized,
        model=OLLAMA_MODEL,
        documents_count=len(documents),
        chunks_count=len(vector_store.chunks) if vector_store else 0,
        embedding_model=EMBEDDING_MODEL,
        knowledge_base_dir=KNOWLEDGE_BASE_DIR
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the RAG system
    
    Send a question and get an answer based on the knowledge base
    """
    if not is_initialized:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please upload documents and rebuild the index."
        )
    
    try:
        logger.info(f"Chat request: {request.question[:100]}...")
        
        # Query the RAG system
        result = rag_system.query(
            question=request.question,
            k=request.k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        logger.info("✓ Chat response generated")
        
        return ChatResponse(
            question=result['question'],
            answer=result['answer'],
            sources=result['sources'],
            chunks_retrieved=len(result['retrieved_chunks']),
            model=OLLAMA_MODEL,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """
    List all documents in the knowledge base
    
    Returns metadata for all PDF and TXT files
    """
    try:
        documents = get_documents_list()
        logger.info(f"Listed {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    auto_rebuild: bool = False
):
    """
    Upload a new document to the knowledge base
    
    - **file**: PDF or TXT file to upload
    - **auto_rebuild**: Automatically rebuild index after upload (default: False)
    
    Returns upload status and optionally triggers index rebuild
    """
    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported"
        )
    
    try:
        # Save file to knowledge base
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, file.filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' already exists. Delete it first or rename your file."
            )
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"✓ Uploaded: {file.filename} ({file_size} bytes)")
        
        response = {
            "status": "success",
            "message": f"File '{file.filename}' uploaded successfully",
            "filename": file.filename,
            "size_bytes": file_size,
            "filepath": file_path
        }
        
        # Auto rebuild if requested
        if auto_rebuild:
            logger.info("Auto-rebuild requested, rebuilding index...")
            result = rebuild_index_sync()
            response["index_rebuilt"] = True
            response["rebuild_stats"] = result
        else:
            response["index_rebuilt"] = False
            response["message"] += ". Remember to rebuild the index to make it searchable."
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    auto_rebuild: bool = False
):
    """
    Upload multiple documents at once
    
    - **files**: List of PDF or TXT files to upload
    - **auto_rebuild**: Automatically rebuild index after upload (default: False)
    """
    uploaded_files = []
    failed_files = []
    
    for file in files:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            failed_files.append({
                "filename": file.filename,
                "error": "Invalid file type. Only PDF and TXT supported."
            })
            continue
        
        try:
            file_path = os.path.join(KNOWLEDGE_BASE_DIR, file.filename)
            
            # Check if file already exists
            if os.path.exists(file_path):
                failed_files.append({
                    "filename": file.filename,
                    "error": "File already exists"
                })
                continue
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_size = os.path.getsize(file_path)
            uploaded_files.append({
                "filename": file.filename,
                "size_bytes": file_size,
                "filepath": file_path
            })
            logger.info(f"✓ Uploaded: {file.filename}")
            
        except Exception as e:
            failed_files.append({
                "filename": file.filename,
                "error": str(e)
            })
            logger.error(f"Failed to upload {file.filename}: {e}")
    
    response = {
        "status": "completed",
        "uploaded_count": len(uploaded_files),
        "failed_count": len(failed_files),
        "uploaded_files": uploaded_files,
        "failed_files": failed_files
    }
    
    # Auto rebuild if requested and files were uploaded
    if auto_rebuild and uploaded_files:
        logger.info("Auto-rebuild requested, rebuilding index...")
        result = rebuild_index_sync()
        response["index_rebuilt"] = True
        response["rebuild_stats"] = result
    else:
        response["index_rebuilt"] = False
    
    return response

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a document from the knowledge base
    
    Note: You need to rebuild the index after deletion to update the search
    """
    file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    
    # Security check: ensure file is in knowledge base directory
    real_path = os.path.realpath(file_path)
    real_kb_path = os.path.realpath(KNOWLEDGE_BASE_DIR)
    
    if not real_path.startswith(real_kb_path):
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        os.remove(file_path)
        logger.info(f"✓ Deleted: {filename}")
        
        return {
            "status": "success",
            "message": f"File '{filename}' deleted successfully",
            "note": "Remember to rebuild the index to update the search"
        }
        
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/rebuild", response_model=RebuildResponse)
async def rebuild_index(background_tasks: BackgroundTasks, background: bool = False):
    """
    Rebuild the vector index from all documents in the knowledge base
    
    - **background**: Run rebuild in background (default: False)
    
    This should be called after uploading or deleting documents
    """
    try:
        if background:
            # Run in background
            background_tasks.add_task(rebuild_index_sync)
            logger.info("Index rebuild started in background")
            
            return RebuildResponse(
                status="started",
                documents_processed=0,
                chunks_created=0,
                message="Index rebuild started in background. Check status endpoint for completion."
            )
        else:
            # Run synchronously
            result = rebuild_index_sync()
            
            return RebuildResponse(
                status="completed",
                documents_processed=result["documents_processed"],
                chunks_created=result["chunks_created"],
                message="Index rebuilt successfully"
            )
            
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "initialized": is_initialized
    }

# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting RAG Knowledge Base API Server")
    print("=" * 60)
    print(f"📂 Knowledge Base: {KNOWLEDGE_BASE_DIR}")
    print(f"🤖 Model: {OLLAMA_MODEL}")
    print(f"📊 Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
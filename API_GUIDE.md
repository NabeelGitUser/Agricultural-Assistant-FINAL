# FastAPI RAG Application - Complete Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_api.txt
```

### 2. Start the Server
```bash
python app.py
```

The API will be available at:
- **API Base:** http://localhost:8000
- **Interactive Docs:** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc

## 📋 API Endpoints

### 1. **Chat with RAG System**
Ask questions about your documents

**Endpoint:** `POST /chat`

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main topics in the documents?",
    "k": 3,
    "temperature": 0.1,
    "max_tokens": 512
  }'
```

**Python Example:**
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "question": "What is the main topic?",
    "k": 3,
    "temperature": 0.1,
    "max_tokens": 512
})

result = response.json()
print(result['answer'])
print("Sources:", result['sources'])
```

**Response:**
```json
{
  "question": "What are the main topics?",
  "answer": "Based on the documents...",
  "sources": ["document1.pdf", "notes.txt"],
  "chunks_retrieved": 3,
  "model": "gemma3:4b",
  "timestamp": "2024-01-29T10:30:00"
}
```

---

### 2. **Upload a Document**
Add a new PDF or TXT file to the knowledge base

**Endpoint:** `POST /documents/upload`

```bash
# Upload without rebuilding index
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@/path/to/document.pdf"

# Upload and auto-rebuild index
curl -X POST "http://localhost:8000/documents/upload?auto_rebuild=true" \
  -F "file=@/path/to/document.pdf"
```

**Python Example:**
```python
import requests

# Upload file
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/documents/upload",
        files=files,
        params={"auto_rebuild": True}
    )

print(response.json())
```

**Response:**
```json
{
  "status": "success",
  "message": "File 'document.pdf' uploaded successfully",
  "filename": "document.pdf",
  "size_bytes": 245678,
  "filepath": "./knowledge_base/document.pdf",
  "index_rebuilt": true,
  "rebuild_stats": {
    "documents_processed": 5,
    "chunks_created": 123
  }
}
```

---

### 3. **Upload Multiple Documents**
Upload several files at once

**Endpoint:** `POST /documents/upload-multiple`

```bash
curl -X POST "http://localhost:8000/documents/upload-multiple?auto_rebuild=true" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt" \
  -F "files=@document3.pdf"
```

**Python Example:**
```python
import requests

files = [
    ("files", open("doc1.pdf", "rb")),
    ("files", open("doc2.txt", "rb")),
    ("files", open("doc3.pdf", "rb"))
]

response = requests.post(
    "http://localhost:8000/documents/upload-multiple",
    files=files,
    params={"auto_rebuild": True}
)

print(response.json())
```

---

### 4. **List All Documents**
Get metadata for all documents in the knowledge base

**Endpoint:** `GET /documents`

```bash
curl "http://localhost:8000/documents"
```

**Python Example:**
```python
import requests

response = requests.get("http://localhost:8000/documents")
documents = response.json()

for doc in documents:
    print(f"{doc['filename']} - {doc['size_bytes']} bytes")
```

**Response:**
```json
[
  {
    "filename": "document.pdf",
    "filepath": "./knowledge_base/document.pdf",
    "size_bytes": 245678,
    "uploaded_at": "2024-01-29T10:00:00",
    "file_type": "pdf"
  }
]
```

---

### 5. **Delete a Document**
Remove a document from the knowledge base

**Endpoint:** `DELETE /documents/{filename}`

```bash
curl -X DELETE "http://localhost:8000/documents/document.pdf"
```

**Python Example:**
```python
import requests

response = requests.delete("http://localhost:8000/documents/document.pdf")
print(response.json())

# Remember to rebuild the index
requests.post("http://localhost:8000/index/rebuild")
```

---

### 6. **Rebuild Index**
Rebuild the vector index after adding/deleting documents

**Endpoint:** `POST /index/rebuild`

```bash
# Rebuild synchronously (wait for completion)
curl -X POST "http://localhost:8000/index/rebuild"

# Rebuild in background
curl -X POST "http://localhost:8000/index/rebuild?background=true"
```

**Python Example:**
```python
import requests

# Synchronous rebuild
response = requests.post("http://localhost:8000/index/rebuild")
result = response.json()
print(f"Processed {result['documents_processed']} documents")
print(f"Created {result['chunks_created']} chunks")
```

**Response:**
```json
{
  "status": "completed",
  "documents_processed": 5,
  "chunks_created": 123,
  "message": "Index rebuilt successfully"
}
```

---

### 7. **Get System Status**
Check system status and statistics

**Endpoint:** `GET /status`

```bash
curl "http://localhost:8000/status"
```

**Response:**
```json
{
  "status": "ready",
  "is_initialized": true,
  "model": "gemma3:4b",
  "documents_count": 5,
  "chunks_count": 123,
  "embedding_model": "all-MiniLM-L6-v2",
  "knowledge_base_dir": "./knowledge_base"
}
```

---

### 8. **Health Check**
Simple health check endpoint

**Endpoint:** `GET /health`

```bash
curl "http://localhost:8000/health"
```

---

## 🔄 Complete Workflow Example

### Initial Setup
```bash
# 1. Start the server
python app.py

# 2. Upload your first documents
curl -X POST "http://localhost:8000/documents/upload?auto_rebuild=true" \
  -F "file=@document1.pdf"

curl -X POST "http://localhost:8000/documents/upload?auto_rebuild=true" \
  -F "file=@document2.txt"

# 3. Start chatting!
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics?"}'
```

### Adding More Documents Later
```bash
# Upload new document (without auto-rebuild for speed)
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@new_document.pdf"

# Manually rebuild when done uploading
curl -X POST "http://localhost:8000/index/rebuild"
```

### Updating Documents
```bash
# Delete old version
curl -X DELETE "http://localhost:8000/documents/old_version.pdf"

# Upload new version
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@new_version.pdf"

# Rebuild index
curl -X POST "http://localhost:8000/index/rebuild"
```

---

## 🐍 Python Client Example

Complete Python script to interact with the API:

```python
import requests
from typing import List

class RAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def chat(self, question: str, k: int = 3) -> dict:
        """Ask a question"""
        response = requests.post(
            f"{self.base_url}/chat",
            json={"question": question, "k": k}
        )
        return response.json()
    
    def upload_document(self, file_path: str, auto_rebuild: bool = False) -> dict:
        """Upload a document"""
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{self.base_url}/documents/upload",
                files=files,
                params={"auto_rebuild": auto_rebuild}
            )
        return response.json()
    
    def list_documents(self) -> List[dict]:
        """List all documents"""
        response = requests.get(f"{self.base_url}/documents")
        return response.json()
    
    def delete_document(self, filename: str) -> dict:
        """Delete a document"""
        response = requests.delete(f"{self.base_url}/documents/{filename}")
        return response.json()
    
    def rebuild_index(self) -> dict:
        """Rebuild the index"""
        response = requests.post(f"{self.base_url}/index/rebuild")
        return response.json()
    
    def get_status(self) -> dict:
        """Get system status"""
        response = requests.get(f"{self.base_url}/status")
        return response.json()

# Usage
if __name__ == "__main__":
    client = RAGClient()
    
    # Upload a document
    result = client.upload_document("document.pdf", auto_rebuild=True)
    print(f"Uploaded: {result['filename']}")
    
    # Ask a question
    answer = client.chat("What is the main topic?")
    print(f"\nQuestion: {answer['question']}")
    print(f"Answer: {answer['answer']}")
    print(f"Sources: {answer['sources']}")
    
    # List all documents
    docs = client.list_documents()
    print(f"\nTotal documents: {len(docs)}")
    for doc in docs:
        print(f"  - {doc['filename']}")
```

---

## ⚙️ Configuration

Edit these variables in `app.py`:

```python
KNOWLEDGE_BASE_DIR = "./knowledge_base"  # Where documents are stored
INDEX_PATH = "./rag_index"               # Where index is saved
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # Embedding model
OLLAMA_MODEL = "gemma3:4b"               # Ollama model to use
```

---

## 🔒 Security Considerations

### For Production:

1. **Add Authentication:**
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/chat")
async def chat(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token
    ...
```

2. **Add Rate Limiting:**
```bash
pip install slowapi
```

3. **File Upload Limits:**
```python
from fastapi import UploadFile, File

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(..., max_length=10_000_000)  # 10MB limit
):
    ...
```

4. **CORS Configuration:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Use different port
uvicorn app:app --port 8001
```

### Ollama connection issues
```bash
# Make sure Ollama is running
ollama list

# Check Ollama service
curl http://localhost:11434/api/tags
```

### Upload fails
- Check file size limits
- Verify file extensions (.pdf, .txt only)
- Ensure knowledge_base directory exists and is writable

### Index rebuild takes too long
- Run rebuild in background: `?background=true`
- Reduce document size or chunk size
- Use smaller embedding model

---

## 📊 Monitoring

View logs:
```bash
tail -f rag_api.log
```

Check system status:
```bash
curl http://localhost:8000/status
```

---

## 🚀 Deployment

### Using Docker:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_api.txt .
RUN pip install -r requirements_api.txt

COPY . .

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t rag-api .
docker run -p 8000:8000 -v $(pwd)/knowledge_base:/app/knowledge_base rag-api
```

### Using systemd (Linux):
Create `/etc/systemd/system/rag-api.service`:
```ini
[Unit]
Description=RAG API Service
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/rag
ExecStart=/usr/bin/python3 /path/to/rag/app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable rag-api
sudo systemctl start rag-api
```

---

## 📝 Notes

- The API automatically initializes on startup if an index exists
- Upload files first, then rebuild the index for best performance
- Use `auto_rebuild=true` only when uploading single files
- For multiple uploads, upload all files first, then rebuild once
- The index persists across server restarts
- Documents are stored in `./knowledge_base` directory

Enjoy your RAG API! 🎉
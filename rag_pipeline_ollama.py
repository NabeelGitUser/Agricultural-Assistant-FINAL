"""
RAG Pipeline for Ollama Models (Gemma3, Llama, etc.)
Supports PDF and TXT files as knowledge base
"""

import os
import pickle
from typing import List, Dict, Any
from pathlib import Path
import requests
import json
import torch

# Document processing
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class DocumentLoader:
    """Load and process documents from various file types"""
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def load_txt(file_path: str) -> str:
        """Load text from TXT file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_documents(self, directory: str) -> List[Dict[str, Any]]:
        """Load all PDF and TXT files from directory"""
        documents = []
        directory_path = Path(directory)
        
        # First, list all files found
        all_files = list(directory_path.rglob('*'))
        print(f"📁 Found {len([f for f in all_files if f.is_file()])} total files in directory")
        
        for file_path in all_files:
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                try:
                    if file_ext == '.pdf':
                        print(f"  📄 Loading PDF: {file_path.name}...")
                        text = self.load_pdf(str(file_path))
                        
                        if not text or len(text.strip()) < 10:
                            print(f"    ⚠️  Warning: PDF appears empty or has very little text")
                        else:
                            print(f"    ✓ Extracted {len(text)} characters")
                        
                        documents.append({
                            'content': text,
                            'source': str(file_path),
                            'type': 'pdf'
                        })
                        
                    elif file_ext == '.txt':
                        print(f"  📝 Loading TXT: {file_path.name}...")
                        text = self.load_txt(str(file_path))
                        
                        if not text or len(text.strip()) < 10:
                            print(f"    ⚠️  Warning: TXT file appears empty or has very little text")
                        else:
                            print(f"    ✓ Loaded {len(text)} characters")
                        
                        documents.append({
                            'content': text,
                            'source': str(file_path),
                            'type': 'txt'
                        })
                    elif file_ext in ['.doc', '.docx', '.rtf']:
                        print(f"  ⚠️  Skipping unsupported format: {file_path.name} ({file_ext})")
                        
                except Exception as e:
                    print(f"  ❌ Error loading {file_path.name}: {e}")
        
        return documents


class TextChunker:
    """Split documents into smaller chunks for better retrieval"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'content': chunk_text,
                'source': metadata['source'],
                'type': metadata['type'],
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk all documents"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc['content'], doc)
            all_chunks.extend(chunks)
        return all_chunks


class VectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Embedding model using: {device}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.index = None
        self.chunks = []
        self.dimension = None
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index from document chunks"""
        if not chunks:
            raise ValueError("Cannot build index: no chunks provided")
        
        print(f"Building index for {len(chunks)} chunks...")
        
        # Store chunks
        self.chunks = chunks
        
        # Generate embeddings
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Ensure embeddings is 2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Create FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for most relevant chunks"""
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'chunk': self.chunks[idx],
                'distance': float(distance)
            })
        
        return results
    
    def save(self, path: str):
        """Save index and chunks to disk"""
        faiss.write_index(self.index, f"{path}_index.faiss")
        with open(f"{path}_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"Index saved to {path}")
    
    def load(self, path: str):
        """Load index and chunks from disk"""
        self.index = faiss.read_index(f"{path}_index.faiss")
        with open(f"{path}_chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"Index loaded from {path}")


class OllamaRAG:
    """RAG system with Ollama models"""
    
    def __init__(
        self,
        model_name: str,
        vector_store: VectorStore,
        ollama_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.vector_store = vector_store
        self.ollama_url = ollama_url
        
        # Test connection to Ollama
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name not in model_names:
                    print(f"⚠️  Warning: Model '{self.model_name}' not found in Ollama")
                    print(f"   Available models: {', '.join(model_names)}")
                    print(f"   Attempting to use anyway...")
                else:
                    print(f"✓ Connected to Ollama - using model: {self.model_name}")
            else:
                print(f"⚠️  Warning: Could not connect to Ollama at {self.ollama_url}")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print(f"   Make sure Ollama is running!")
    
    def generate_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Create prompt with retrieved context"""
        context = "\n\n".join([
            f"[Source: {chunk['chunk']['source']}]\n{chunk['chunk']['content']}"
            for chunk in context_chunks
        ])
        
        # Detect language
        tamil_chars = set('அஆஇஈஉஊஎஏஐஒஓஔகசடதநபமயரலவழளறனஜஷஸஹ')
        is_tamil = any(char in tamil_chars for char in query)
        
        if is_tamil:
            prompt = f"""நீங்கள் ஒரு விவசாய உதவியாளர். கீழே உள்ள தகவல்களை பயன்படுத்தி தமிழில் மட்டும் பதில் சொல்லுங்கள்.

        தகவல்:
        {context}

        கேள்வி: {query}

        தமிழில் பதில்:"""
        else:
            # Strip Tamil from context for English questions
            import re
            # Remove Tamil characters from context
            clean_context = re.sub(r'[\u0B80-\u0BFF]+', '[Tamil content - translate to English]', context)
    
            prompt = f"""You are an agricultural assistant. Answer in English only.

        Context:
        {clean_context}

        Question: {query}

        English Answer:"""
        
        return prompt
    
    def query(
        self,
        question: str,
        k: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Query the RAG system"""
        # Retrieve relevant chunks
        print(f"Retrieving top {k} relevant chunks...")
        retrieved_chunks = self.vector_store.search(question, k=k)
        
        # Generate prompt
        prompt = self.generate_prompt(question, retrieved_chunks)
        
        # Generate response using Ollama API
        print(f"Generating response with {self.model_name}...")
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
            else:
                answer = f"Error: Ollama returned status code {response.status_code}"
                
        except Exception as e:
            answer = f"Error generating response: {e}"
        
        return {
            'question': question,
            'answer': answer,
            'sources': [chunk['chunk']['source'] for chunk in retrieved_chunks],
            'retrieved_chunks': retrieved_chunks
        }


def main():
    """Example usage"""
    
    # Configuration
    DOCUMENTS_DIR = "./knowledge_base"  # Directory containing PDFs and TXTs
    MODEL_NAME = "gemma3:1b"  # Change to your preferred Ollama model
    INDEX_PATH = "./rag_index"
    
    # Step 1: Load documents
    print("Step 1: Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_documents(DOCUMENTS_DIR)
    print(f"Loaded {len(documents)} documents")
    
    # Step 2: Chunk documents
    print("\nStep 2: Chunking documents...")
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Step 3: Build vector store
    print("\nStep 3: Building vector store...")
    vector_store = VectorStore(embedding_model_name='all-MiniLM-L6-v2')
    vector_store.build_index(chunks)
    
    # Optionally save the index
    vector_store.save(INDEX_PATH)
    
    # Step 4: Initialize RAG system
    print("\nStep 4: Initializing RAG system...")
    rag_system = OllamaRAG(
        model_name=MODEL_NAME,
        vector_store=vector_store
    )
    
    # Step 5: Query the system
    print("\nStep 5: Ready to answer questions!")
    print("-" * 50)
    
    # Example queries
    questions = [
        "What are the main topics covered in the documents?",
        # Add your questions here
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag_system.query(question, k=3)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources: {', '.join(set(result['sources']))}")
        print("-" * 50)


if __name__ == "__main__":
    main()

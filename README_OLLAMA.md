# RAG Pipeline for Ollama Models

A RAG pipeline specifically designed for **Ollama** models (Gemma3, Llama, Tamil models, etc.) with support for PDF and TXT files.

## Features

- 🤖 **Ollama Integration**: Works with any Ollama model
- 📄 **Multi-format support**: PDF and TXT files
- 🔍 **Efficient retrieval**: FAISS-based vector search
- 💾 **Persistent storage**: Save and load indexes
- 🎯 **Smart chunking**: Overlapping text chunks for better context
- 🔄 **Interactive CLI**: Easy-to-use command-line interface

## Prerequisites

1. **Ollama installed and running**
   ```bash
   # Check if Ollama is running
   ollama list
   ```

2. **Python dependencies**
   ```bash
   pip install -r requirements_ollama.txt
   ```

## Quick Start

### 1. Prepare Your Knowledge Base

Create a directory with your PDF and TXT files:
```
knowledge_base/
├── document1.pdf
├── document2.txt
└── research_paper.pdf
```

### 2. Build the Index

```bash
python rag_cli_ollama.py --mode build --docs-dir ./knowledge_base
```

### 3. Query Your Documents

**With Gemma3 1B (fastest):**
```bash
python rag_cli_ollama.py --mode query --model gemma3:1b
```

**With Gemma3 4B (better quality):**
```bash
python rag_cli_ollama.py --mode query --model gemma3:4b
```

**With Llama 3.2:**
```bash
python rag_cli_ollama.py --mode query --model llama3.2:latest
```

**With Tamil Gemma:**
```bash
python rag_cli_ollama.py --mode query --model hf.co/abhinand/gemma-2b-it-tamil-v0.1-alpha-GGUF:latest
```

**With Tamil Llama:**
```bash
python rag_cli_ollama.py --mode query --model hf.co/mradermacher/tamil-llama-7b-instruct-v0.1-GGUF:latest
```

## Your Available Models

Based on your Ollama installation:
- ✅ `gemma3:4b` - Good balance of speed and quality
- ✅ `llama3.2:latest` - **Recommended** for general use
- ✅ `gemma3:1b` - Fastest option
- ✅ `hf.co/abhinand/gemma-2b-it-tamil-v0.1-alpha-GGUF:latest` - Tamil language support
- ✅ `hf.co/unsloth/gemma-3-1b-it-GGUF:latest` - Alternative Gemma
- ✅ `hf.co/mradermacher/tamil-llama-7b-instruct-v0.1-GGUF:latest` - Best for Tamil

## Advanced Usage

### Custom Settings

```bash
python rag_cli_ollama.py --mode build \
  --docs-dir ./my_docs \
  --index-path ./my_index \
  --embedding-model sentence-transformers/all-mpnet-base-v2
```

### Different Model for Querying

```bash
python rag_cli_ollama.py --mode query \
  --model llama3.2:latest \
  --index-path ./my_index
```

## Programmatic Usage

```python
from rag_pipeline_ollama import DocumentLoader, TextChunker, VectorStore, OllamaRAG

# Load and chunk documents
loader = DocumentLoader()
documents = loader.load_documents("./knowledge_base")

chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_documents(documents)

# Build vector store
vector_store = VectorStore()
vector_store.build_index(chunks)

# Initialize RAG with your Ollama model
rag = OllamaRAG(
    model_name="gemma3:1b",
    vector_store=vector_store
)

# Query
result = rag.query("What is the main topic?", k=3)
print(result['answer'])
print(f"Sources: {result['sources']}")
```

## Configuration

### Chunking Parameters
- `chunk_size`: Words per chunk (default: 500)
- `chunk_overlap`: Overlapping words (default: 50)

### Query Parameters
- `k`: Number of chunks to retrieve (default: 3)
- `temperature`: Sampling temperature (default: 0.1)
- `max_tokens`: Maximum tokens in response (default: 512)

## Performance Recommendations

### Model Selection
- **Fast response needed**: `gemma3:1b`
- **Best quality**: `llama3.2:latest` or `gemma3:4b`
- **Tamil documents**: `tamil-llama-7b` or `gemma-2b-it-tamil`

### Retrieval Settings
- Start with `k=3` chunks
- Increase to `k=5` for complex questions
- Reduce to `k=1-2` for simple lookups

### Memory Optimization
- Smaller models (1b-4b) for limited RAM
- Reduce chunk_size if running out of memory
- Use `faiss-cpu` instead of `faiss-gpu` to save VRAM

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# Or restart it
ollama stop
ollama serve
```

### "Model not found"
```bash
# Pull the model first
ollama pull gemma3:1b

# Or use a model you already have
ollama list
```

### Empty PDF Results
- Check if PDF has selectable text (not scanned images)
- Try opening the PDF and copy-pasting text to verify
- Consider using OCR for scanned PDFs

### Slow Performance
- Use smaller model (gemma3:1b)
- Reduce `k` parameter
- Reduce `max_tokens`
- Use GPU acceleration if available

## Architecture

```
Documents (PDF/TXT)
    ↓
Text Extraction
    ↓
Chunking (with overlap)
    ↓
Embedding Generation
    ↓
FAISS Vector Store
    ↓
Query → Retrieval → Context + Query → Ollama Model → Answer
```

## Differences from Transformers Version

This Ollama version:
- ✅ Uses Ollama API (no need for HuggingFace downloads)
- ✅ Works with locally installed Ollama models
- ✅ Lighter weight (no PyTorch needed for inference)
- ✅ Easier model switching
- ✅ Better for production deployment

## File Structure

```
.
├── rag_pipeline_ollama.py     # Core RAG implementation
├── rag_cli_ollama.py          # Interactive CLI
├── requirements_ollama.txt    # Dependencies
└── README_OLLAMA.md          # This file
```

## Tips

1. **Test your model first**: Run `ollama run gemma3:1b` to verify it works
2. **Start small**: Use `gemma3:1b` for initial testing
3. **Optimize chunks**: Adjust based on your document structure
4. **Monitor memory**: Large models + many chunks = high RAM usage

Enjoy your RAG pipeline! 🚀

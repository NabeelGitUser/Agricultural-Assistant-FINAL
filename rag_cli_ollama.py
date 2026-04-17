"""
Interactive RAG Query Interface for Ollama Models
"""

from rag_pipeline_ollama import DocumentLoader, TextChunker, VectorStore, OllamaRAG
import argparse

    
def build_knowledge_base(docs_dir: str, index_path: str, embedding_model: str):
    """Build and save the knowledge base"""
    print("Building knowledge base...")
    print(f"📂 Looking for documents in: {docs_dir}")
    
    # Check if directory exists
    import os
    if not os.path.exists(docs_dir):
        print(f"❌ Error: Directory '{docs_dir}' does not exist!")
        print(f"   Please create it and add your PDF/TXT files.")
        return None
    
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_documents(docs_dir)
    
    if len(documents) == 0:
        print(f"❌ Error: No PDF or TXT files found in '{docs_dir}'")
        print(f"   Please add at least one .pdf or .txt file to the directory.")
        return None
    
    print(f"✓ Loaded {len(documents)} documents")
    
    # Chunk documents
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)
    
    if len(chunks) == 0:
        print(f"❌ Error: No text content found in documents")
        print(f"   Please check if your PDF files contain extractable text.")
        return None
    
    print(f"✓ Created {len(chunks)} chunks")
    
    # Build vector store
    vector_store = VectorStore(embedding_model_name=embedding_model)
    vector_store.build_index(chunks)
    vector_store.save(index_path)
    print(f"✓ Index saved to {index_path}")
    
    return vector_store


def interactive_query(model_name: str, index_path: str, embedding_model: str):
    """Interactive query interface"""
    print("\nLoading vector store...")
    vector_store = VectorStore(embedding_model_name=embedding_model)
    vector_store.load(index_path)
    
    print("Initializing RAG system...")
    rag_system = OllamaRAG(model_name=model_name, vector_store=vector_store)
    
    print("\n" + "="*60)
    print("RAG System Ready! Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("\n📝 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            # Query the system
            result = rag_system.query(question, k=3)
            
            print("\n" + "-"*60)
            print(f"💡 Answer:\n{result['answer']}")
            print("\n📚 Sources:")
            for source in set(result['sources']):
                print(f"  - {source}")
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline for Ollama Models")
    parser.add_argument(
        '--mode',
        choices=['build', 'query'],
        required=True,
        help='build: Create knowledge base | query: Interactive Q&A'
    )
    parser.add_argument(
        '--docs-dir',
        default='./knowledge_base',
        help='Directory containing PDF and TXT files'
    )
    parser.add_argument(
        '--model',
        default='gemma3:1b',
        help='Ollama model name (e.g., gemma3:1b, llama3.2:latest, gemma3:4b)'
    )
    parser.add_argument(
        '--index-path',
        default='./rag_index',
        help='Path to save/load the vector index'
    )
    parser.add_argument(
        '--embedding-model',
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model for embeddings'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'build':
        result = build_knowledge_base(args.docs_dir, args.index_path, args.embedding_model)
        if result:
            print("\n✅ Knowledge base built successfully!")
            print(f"Run with --mode query --model {args.model} to start asking questions")
        else:
            print("\n❌ Failed to build knowledge base. Please check the errors above.")
        
    elif args.mode == 'query':
        interactive_query(args.model, args.index_path, args.embedding_model)


if __name__ == "__main__":
    main()

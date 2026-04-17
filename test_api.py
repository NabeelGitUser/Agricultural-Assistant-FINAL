"""
Test script for RAG API
Run this after starting the server to verify everything works
"""

import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_status():
    """Test status endpoint"""
    print_section("Testing System Status")
    response = requests.get(f"{BASE_URL}/status")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Initialized: {data['is_initialized']}")
    print(f"Model: {data['model']}")
    print(f"Documents: {data['documents_count']}")
    print(f"Chunks: {data['chunks_count']}")
    return response.status_code == 200

def test_list_documents():
    """Test listing documents"""
    print_section("Testing Document List")
    response = requests.get(f"{BASE_URL}/documents")
    print(f"Status Code: {response.status_code}")
    documents = response.json()
    print(f"Total Documents: {len(documents)}")
    for doc in documents:
        print(f"  - {doc['filename']} ({doc['size_bytes']} bytes)")
    return response.status_code == 200

def test_upload_sample():
    """Test uploading a sample document"""
    print_section("Testing Document Upload")
    
    # Create a sample text file
    sample_content = """
    This is a sample document for testing the RAG API.
    
    The RAG (Retrieval-Augmented Generation) system combines:
    1. Document storage and retrieval
    2. Semantic search using embeddings
    3. Language model generation
    
    This allows for answering questions based on a knowledge base.
    """
    
    sample_file = "test_document.txt"
    with open(sample_file, "w") as f:
        f.write(sample_content)
    
    # Upload the file
    try:
        with open(sample_file, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{BASE_URL}/documents/upload",
                files=files,
                params={"auto_rebuild": True}
            )
        
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Message: {data['message']}")
        print(f"Filename: {data['filename']}")
        print(f"Size: {data['size_bytes']} bytes")
        if data.get('index_rebuilt'):
            print(f"Index Rebuilt: Yes")
            print(f"  - Documents Processed: {data['rebuild_stats']['documents_processed']}")
            print(f"  - Chunks Created: {data['rebuild_stats']['chunks_created']}")
        
        return response.status_code == 200
    finally:
        # Cleanup
        Path(sample_file).unlink(missing_ok=True)

def test_chat():
    """Test chat endpoint"""
    print_section("Testing Chat")
    
    questions = [
        "What is RAG?",
        "What are the main components mentioned?",
        "How does the system work?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "question": question,
                "k": 3,
                "temperature": 0.1,
                "max_tokens": 256
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Answer: {data['answer'][:200]}...")
            print(f"Sources: {data['sources']}")
            print(f"Chunks Retrieved: {data['chunks_retrieved']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
        
        time.sleep(1)  # Brief pause between requests
    
    return True

def test_rebuild():
    """Test index rebuild"""
    print_section("Testing Index Rebuild")
    
    response = requests.post(f"{BASE_URL}/index/rebuild")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Documents Processed: {data['documents_processed']}")
        print(f"Chunks Created: {data['chunks_created']}")
        print(f"Message: {data['message']}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪" * 30)
    print("  RAG API Test Suite")
    print("🧪" * 30)
    
    tests = [
        ("Health Check", test_health),
        ("System Status", test_status),
        ("List Documents", test_list_documents),
        ("Upload Document", test_upload_sample),
        ("System Status (After Upload)", test_status),
        ("Chat", test_chat),
        ("Index Rebuild", test_rebuild)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Error in {test_name}: {e}")
            results.append((test_name, False))
        
        time.sleep(0.5)
    
    # Summary
    print_section("Test Results Summary")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    print("Make sure the API server is running on http://localhost:8000")
    print("Start it with: python app.py")
    input("\nPress Enter when ready to run tests...")
    
    run_all_tests()

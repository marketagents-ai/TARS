import asyncio
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from agent.hybrid_memory import UserMemoryIndex
from agent.api_client import initialize_api_client
from types import SimpleNamespace

async def simulate_memory_system():
    # Initialize API client first (use Ollama for embeddings)
    args = SimpleNamespace(api='ollama', model='nomic-embed-text')
    initialize_api_client(args)
    
    # Initialize the memory system
    memory_index = UserMemoryIndex("test_bot/memory_index")
    
    # Test User IDs
    user_1 = "user123"
    user_2 = "user456"
    
    # Test Case 1: Adding a simple memory
    print("\n=== Test Case 1: Simple Memory Addition ===")
    await memory_index.add_memory(user_1, "The sky is blue today and the weather is perfect for a walk.")
    results = await memory_index.search("weather", user_id=user_1)
    print("Search for 'weather':")
    for memory, score in results:
        print(f"Score: {score:.2f} | Memory: {memory}")

    # Test Case 2: Long Memory Chunking
    print("\n=== Test Case 2: Long Memory Chunking ===")
    long_text = """
    The artificial intelligence revolution has transformed various sectors of the economy. 
    From healthcare to finance, AI systems are being deployed to solve complex problems.
    In healthcare, AI is being used for diagnosis and treatment planning.
    Machine learning models can analyze medical images with high accuracy.
    Natural language processing helps in analyzing patient records.
    In the financial sector, AI is used for fraud detection and risk assessment.
    Algorithmic trading systems use AI to make split-second decisions.
    The technology is also revolutionizing customer service through chatbots.
    AI-powered recommendation systems help in personalizing user experiences.
    However, there are concerns about AI ethics and privacy implications.
    """
    await memory_index.add_memory(user_1, long_text, max_tokens=100)  # Smaller chunk size for testing
    
    print("\nSearch for 'AI technology':")
    results = await memory_index.search("AI technology", user_id=user_1, k=3)
    for memory, score in results:
        print(f"Score: {score:.2f} | Memory: {memory}")

    # Test Case 3: Cross-user isolation
    print("\n=== Test Case 3: User Memory Isolation ===")
    await memory_index.add_memory(user_2, "AI is transforming education through personalized learning.")
    
    print("\nUser 1 searching for 'AI':")
    results = await memory_index.search("AI", user_id=user_1, k=2)
    for memory, score in results:
        print(f"Score: {score:.2f} | Memory: {memory}")
        
    print("\nUser 2 searching for 'AI':")
    results = await memory_index.search("AI", user_id=user_2, k=2)
    for memory, score in results:
        print(f"Score: {score:.2f} | Memory: {memory}")

    # Test Case 4: Deduplication
    print("\n=== Test Case 4: Deduplication ===")
    similar_texts = [
        "The cat sat on the mat watching birds.",
        "A cat was sitting on the mat observing birds.",
        "Dogs are loyal companions and great pets.",
    ]
    for text in similar_texts:
        await memory_index.add_memory(user_1, text)
    
    print("\nSearch for 'cat' with deduplication:")
    results = await memory_index.search("cat", user_id=user_1, similarity_threshold=0.7)
    for memory, score in results:
        print(f"Score: {score:.2f} | Memory: {memory}")

    # Test Case 5: Memory clearing
    print("\n=== Test Case 5: Memory Clearing ===")
    memory_index.clear_user_memories(user_1)
    results = await memory_index.search("AI", user_id=user_1)
    print("\nUser 1 searching for 'AI' after clearing:")
    print(f"Results found: {len(results)}")

async def main():
    await simulate_memory_system()

if __name__ == "__main__":
    asyncio.run(main())
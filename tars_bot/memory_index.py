import tiktoken
import pickle
import os
from collections import defaultdict, Counter
import logging
import re
import string
from cache_manager import CacheManager

class UserMemoryIndex:
    def __init__(self, cache_type, max_tokens, context_chunks):
        self.cache_manager = CacheManager('user_memory_index')
        self.cache_dir = self.cache_manager.get_cache_dir(cache_type)
        self.max_tokens = max_tokens
        self.context_chunks = context_chunks
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.inverted_index = defaultdict(list)
        self.memories = []
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        self.user_memories = defaultdict(list)  # Store memories per user
        self.load_cache()

    def clean_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        return ' '.join(words)

    def add_memory(self, user_id, memory_text):
        memory_id = len(self.memories)
        self.memories.append(memory_text)
        self.user_memories[user_id].append(memory_id)
        
        cleaned_text = self.clean_text(memory_text)
        words = cleaned_text.split()
        for word in words:
            self.inverted_index[word].append(memory_id)
        
        logging.info(f"Added new memory for user {user_id}: {memory_text[:100]}...")
        self.save_cache()

    def clear_user_memories(self, user_id):
        if user_id in self.user_memories:
            memory_ids_to_remove = self.user_memories[user_id]
            for memory_id in memory_ids_to_remove:
                self.memories[memory_id] = None  # Mark as removed
            
            # Update inverted index
            for word, ids in self.inverted_index.items():
                self.inverted_index[word] = [id for id in ids if id not in memory_ids_to_remove]
            
            del self.user_memories[user_id]
            logging.info(f"Cleared memories for user {user_id}")
            self.save_cache()

    def search(self, query, k=5):
        cleaned_query = self.clean_text(query)
        query_words = cleaned_query.split()
        memory_scores = Counter()

        for word in query_words:
            for memory_id in self.inverted_index.get(word, []):
                memory_scores[memory_id] += 1
        
        for memory_id, score in memory_scores.items():
            memory_scores[memory_id] = score / len(self.clean_text(self.memories[memory_id]).split())
        
        sorted_memories = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        total_tokens = 0
        
        for memory_id, score in sorted_memories[:k]:
            memory = self.memories[memory_id]
            memory_tokens = self.count_tokens(memory)

            if total_tokens + memory_tokens > self.max_tokens:
                break

            results.append((memory, score))
            total_tokens += memory_tokens

        logging.info(f"Found {len(results)} relevant memories for query: {query[:100]}...")
        return results
    
    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))
    
    def save_cache(self):
        cache_data = {
            'inverted_index': self.inverted_index,
            'memories': self.memories,
            'user_memories': self.user_memories
        }
        with open(os.path.join(self.cache_dir, 'memory_cache.pkl'), 'wb') as f:
            pickle.dump(cache_data, f)
        logging.info("Memory cache saved successfully.")

    def load_cache(self):
        cache_file = os.path.join(self.cache_dir, 'memory_cache.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.inverted_index = cache_data.get('inverted_index', defaultdict(list))
            self.memories = cache_data.get('memories', [])
            self.user_memories = cache_data.get('user_memories', defaultdict(list))
            logging.info("Memory cache loaded successfully.")
            return True
        return False

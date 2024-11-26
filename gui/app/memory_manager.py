import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import uuid
from collections import defaultdict
import os

from collections import Counter
import tiktoken

from .models import MemoryUpdate

class MemoryManager:
    """Manages operations on the memory pickle file"""
    
    def __init__(self, pickle_path: str):
        # Convert relative path to absolute path if needed
        if not os.path.isabs(pickle_path):
            # Store in a 'cache' directory within the project
            pickle_path = os.path.join(os.getcwd(), 'cache', pickle_path)
        
        self.pickle_path = Path(pickle_path)
        self.inverted_index = defaultdict(list)  # Maps words to memory IDs
        self.memories = []  # List of all memory texts
        self.user_memories = defaultdict(list)  # Maps user IDs to their memory IDs
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
                            'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Create cache directory if it doesn't exist
        os.makedirs(os.path.dirname(self.pickle_path), exist_ok=True)
        self.load_memories()
    
    def load_memories(self) -> None:
        """Load memories from pickle file"""
        try:
            if self.pickle_path.exists():
                logging.info(f"Found pickle file at: {self.pickle_path}")
                with open(self.pickle_path, 'rb') as f:
                    try:
                        cache_data = pickle.load(f)
                        # Load the three main components from the cache
                        self.inverted_index = cache_data.get('inverted_index', defaultdict(list))
                        self.memories = cache_data.get('memories', [])
                        self.user_memories = cache_data.get('user_memories', defaultdict(list))
                        
                        # Log stats
                        logging.info(f"Loaded {len(self.memories)} memories")
                        logging.info(f"Loaded {len(self.user_memories)} users")
                        logging.info(f"Loaded {sum(len(v) for v in self.inverted_index.values())} index entries")
                    except Exception as e:
                        logging.error(f"Error during pickle load: {str(e)}")
                        self.inverted_index = defaultdict(list)
                        self.memories = []
                        self.user_memories = defaultdict(list)
            else:
                logging.warning(f"No pickle file found at: {self.pickle_path}")
                self.inverted_index = defaultdict(list)
                self.memories = []
                self.user_memories = defaultdict(list)
        except Exception as e:
            logging.error(f"Error loading memories: {str(e)}")
            self.inverted_index = defaultdict(list)
            self.memories = []
            self.user_memories = defaultdict(list)
    
    def save_memories(self) -> None:
        """Save memories to pickle file"""
        try:
            self.pickle_path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                'inverted_index': self.inverted_index,
                'memories': self.memories,
                'user_memories': self.user_memories
            }
            with open(self.pickle_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logging.info(f"Saved {len(self.memories)} memories to {self.pickle_path}")
        except Exception as e:
            logging.error(f"Error saving memories: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for indexing/searching"""
        import string
        import re
        
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        return ' '.join(words)
    
    def get_memory(self, memory_id: int) -> Optional[str]:
        """Retrieve a single memory by ID"""
        try:
            return self.memories[memory_id]
        except IndexError:
            return None
    
    def get_user_memories(self, user_id: str) -> List[tuple[str, int]]:
        """Get all memories for a specific user"""
        memory_ids = self.user_memories.get(user_id, [])
        return [(self.memories[mid], mid) for mid in memory_ids if mid < len(self.memories)]
    
    def add_memory(self, user_id: str, memory_text: str) -> int:
        """Add a new memory"""
        memory_id = len(self.memories)
        self.memories.append(memory_text)
        self.user_memories[user_id].append(memory_id)
        
        # Index the memory
        cleaned_text = self.clean_text(memory_text)
        words = cleaned_text.split()
        for word in words:
            self.inverted_index[word].append(memory_id)
        
        self.save_memories()
        return memory_id
    
    def update_memory(self, memory_id: int, update_data: MemoryUpdate) -> Optional[str]:
        """Update an existing memory"""
        try:
            if update_data.content is not None:
                # Remove old indexing
                old_text = self.memories[memory_id]
                old_words = self.clean_text(old_text).split()
                for word in old_words:
                    if memory_id in self.inverted_index[word]:
                        self.inverted_index[word].remove(memory_id)
                
                # Update memory and add new indexing
                self.memories[memory_id] = update_data.content
                new_words = self.clean_text(update_data.content).split()
                for word in new_words:
                    self.inverted_index[word].append(memory_id)
                
                self.save_memories()
                return self.memories[memory_id]
        except IndexError:
            return None
        return None
    
    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID"""
        try:
            # First check if memory exists and is not None
            if memory_id < 0 or memory_id >= len(self.memories) or self.memories[memory_id] is None:
                return False
                
            # Get memory text before deletion
            memory_text = self.memories[memory_id]
            
            # Remove from inverted index
            words = self.clean_text(memory_text).split()
            for word in words:
                if memory_id in self.inverted_index[word]:
                    self.inverted_index[word].remove(memory_id)
            
            # Remove from user memories
            for user_memories in self.user_memories.values():
                if memory_id in user_memories:
                    user_memories.remove(memory_id)
            
            # Mark as removed in memories list
            self.memories[memory_id] = None
            
            self.save_memories()
            return True
        except Exception as e:
            logging.error(f"Error deleting memory {memory_id}: {str(e)}")
            return False
    
    def search_memories(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[tuple[str, int, float]]:
        """Search for relevant memories matching a query, removing duplicates.

        Args:
            query (str): Search query text
            user_id (Optional[str]): If provided, only search this user's memories
            limit (int): Maximum number of results to return

        Returns:
            List[tuple[str, int, float]]: List of tuples containing (memory_text, memory_id, relevance_score)
        """


        # Initialize tokenizer
        tokenizer = tiktoken.get_encoding("cl100k_base")
        max_tokens = 2000  # Match the original implementation's token limit
        
        cleaned_query = self.clean_text(query)
        query_words = cleaned_query.split()
        memory_scores = Counter()

        # If user_id is provided, only search that user's memories
        if user_id:
            relevant_memory_ids = self.user_memories.get(user_id, [])
        else:
            relevant_memory_ids = range(len(self.memories))

        # Score memories based on word matches
        for word in query_words:
            for memory_id in self.inverted_index.get(word, []):
                if memory_id in relevant_memory_ids and self.memories[memory_id] is not None:
                    memory_scores[memory_id] += 1

        # Normalize scores by memory length
        sorted_memories = []
        for memory_id, score in memory_scores.items():
            memory = self.memories[memory_id]
            if memory:  # Skip None values (deleted memories)
                normalized_score = score / len(self.clean_text(memory).split())
                sorted_memories.append((memory_id, memory, normalized_score))
        
        sorted_memories.sort(key=lambda x: x[2], reverse=True)
        
        # Deduplication process
        results = []
        total_tokens = 0
        seen_content = set()  # Track unique content fingerprints
        
        for memory_id, memory, score in sorted_memories:
            memory_tokens = len(tokenizer.encode(memory))
            
            if total_tokens + memory_tokens > max_tokens:
                break
                
            # Create a content fingerprint
            cleaned_memory = self.clean_text(memory)
            
            # Check for similar content using n-gram comparison
            is_duplicate = False
            for seen in seen_content:
                similarity = self._calculate_similarity(cleaned_memory, seen)
                if similarity > 0.85:  # Using the same threshold as original
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                results.append((memory, memory_id, score))
                seen_content.add(cleaned_memory)
                total_tokens += memory_tokens
                
                if len(results) >= limit:
                    break
        
        return results

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using character n-grams.
        
        Args:
            text1 (str): First text to compare
            text2 (str): Second text to compare
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Use 3-character n-grams for comparison
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text)-n+1))
        
        # Get n-grams for both texts
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        # Calculate Jaccard similarity
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0

    def get_all_users(self) -> List[str]:
        """Get list of all user IDs that have memories"""
        return list(self.user_memories.keys())

    def validate_db(self) -> dict:
        """Validate database integrity and return stats
        
        Returns:
            dict: Database statistics and validation results
        """
        stats = {
            'total_memories': len(self.memories),
            'active_memories': len([m for m in self.memories if m is not None]),
            'total_users': len(self.user_memories),
            'index_size': sum(len(v) for v in self.inverted_index.values()),
            'user_stats': {},
            'validation': {
                'orphaned_memories': [],
                'invalid_user_refs': [],
                'missing_index_entries': []
            }
        }
        
        # Check each user's memories
        for user_id, memory_ids in self.user_memories.items():
            user_stats = {
                'total_memories': len(memory_ids),
                'active_memories': 0,
                'invalid_refs': 0
            }
            
            for mid in memory_ids:
                if mid >= len(self.memories):
                    stats['validation']['invalid_user_refs'].append((user_id, mid))
                    user_stats['invalid_refs'] += 1
                elif self.memories[mid] is not None:
                    user_stats['active_memories'] += 1
                    
            stats['user_stats'][user_id] = user_stats
        
        # Check for orphaned memories
        for mid, memory in enumerate(self.memories):
            if memory is not None:
                found = False
                for user_memories in self.user_memories.values():
                    if mid in user_memories:
                        found = True
                        break
                if not found:
                    stats['validation']['orphaned_memories'].append(mid)
        
        return stats
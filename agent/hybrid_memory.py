# NOT YET IMPLEMENTED. THIS IS A HYBRID REPLACEMENT FOR THIS EXISTING INVERTED INDEX. USING 
# ILLUSTRATIVE PURPOSES ONLY AS WE MOVE TO A SQL BASED SYSTEM.

import os
import json
import pickle
import logging
from datetime import datetime
from collections import defaultdict, Counter
import tiktoken
from typing import List, Tuple, Optional, Dict, Any
import string
import re
import numpy as np
from numpy.linalg import norm

from agent.bot_config import *
from agent.cache_manager import CacheManager
from agent.api_client import get_embeddings

from nltk.tokenize import sent_tokenize

# You might need to download the punkt tokenizer data first time:
# import nltk
# nltk.download('punkt')

# Memory

class UserMemoryIndex:
    """A class for indexing and searching user memories with efficient caching.
    
    This class provides functionality to store, index, and search through user memories
    using an inverted index approach. It handles caching of memories to disk and 
    supports per-user memory isolation.

    Attributes:
        cache_manager (CacheManager): Manager for handling cache operations
        cache_dir (str): Directory path for storing cache files
        max_tokens (int): Maximum number of tokens allowed in search results
        context_chunks (int): Number of context chunks to maintain
        tokenizer: Tokenizer for counting tokens in text
        inverted_index (defaultdict): Inverted index mapping words to memory IDs
        memories (list): List of all memory texts
        stopwords (set): Set of common words to ignore during indexing
        user_memories (defaultdict): Mapping of user IDs to their memory IDs
        embeddings (dict): Memory ID to embedding mapping
    """

    def __init__(self, cache_type, max_tokens=MAX_TOKENS, context_chunks=CONTEXT_CHUNKS):
        """Initialize the UserMemoryIndex.

        Args:
            cache_type (str): Type of cache to use (e.g., 'user_memory_index')
            max_tokens (int, optional): Maximum tokens in search results. Defaults to MAX_TOKENS.
            context_chunks (int, optional): Number of context chunks. Defaults to CONTEXT_CHUNKS.
        """
        # Split the cache_type to get the bot name and cache type
        parts = cache_type.split('/')
        if len(parts) >= 2:
            bot_name = parts[0]
            cache_subtype = parts[-1]  # Use the last part as the cache subtype
        else:
            bot_name = 'default'
            cache_subtype = cache_type

        self.cache_manager = CacheManager(bot_name)
        self.cache_dir = self.cache_manager.get_cache_dir(cache_subtype)
        self.max_tokens = max_tokens
        self.context_chunks = context_chunks
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.inverted_index = defaultdict(list)
        self.memories = []
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        self.user_memories = defaultdict(list)
        self.embeddings = {}  # Memory ID to embedding mapping
        self.load_cache()
        self.load_embeddings()

    def load_embeddings(self):
        """Load embeddings from separate cache file if it exists."""
        embeddings_file = os.path.join(self.cache_dir, 'memory_embeddings.pkl')
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            logging.info("Memory embeddings loaded successfully.")

    def save_embeddings(self):
        """Save embeddings to separate cache file."""
        with open(os.path.join(self.cache_dir, 'memory_embeddings.pkl'), 'wb') as f:
            pickle.dump(self.embeddings, f)
        logging.info("Memory embeddings saved successfully.")

    def clean_text(self, text):
        """Clean and normalize text for indexing/searching.

        Args:
            text (str): Text to clean

        Returns:
            str: Cleaned text with punctuation removed, numbers removed, and stopwords filtered
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        return ' '.join(words)

    async def add_memory(self, user_id: str, memory_text: str, max_tokens: int = 512):
        """Add a new memory with embedding for a user, chunking if necessary.
        
        Args:
            user_id (str): ID of the user this memory belongs to
            memory_text (str): Text content of the memory
            max_tokens (int, optional): Maximum tokens per chunk. Defaults to 512.
        """
        # Create chunks if text is too long
        text_chunks = self._create_chunks(memory_text, max_tokens)
        
        for chunk in text_chunks:
            memory_id = len(self.memories)
            self.memories.append(chunk)
            self.user_memories[user_id].append(memory_id)
            
            # Traditional inverted index
            cleaned_text = self.clean_text(chunk)
            words = cleaned_text.split()
            for word in words:
                self.inverted_index[word].append(memory_id)
            
            # Generate and store embedding
            try:
                embedding = await get_embeddings(chunk)
                self.embeddings[memory_id] = embedding
            except Exception as e:
                logging.warning(f"Failed to generate embedding for memory {memory_id}: {e}")
            
            logging.info(f"Added new memory chunk for user {user_id}: {chunk[:100]}...")
        
        self.save_cache()
        self.save_embeddings()

    def clear_user_memories(self, user_id):
        """Clear all memories for a specific user.

        Args:
            user_id (str): ID of user whose memories should be cleared
        """
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

    async def search(self, query, k=5, user_id=None, similarity_threshold=0.85, hybrid_weight=0.5):
        """Hybrid search combining BM25-style and embedding similarity.
        
        Args:
            query (str): Search query text
            k (int): Maximum number of results
            user_id (str, optional): If provided, only search this user's memories
            similarity_threshold (float): Threshold for deduplication
            hybrid_weight (float): Weight between 0 and 1 for combining scores
                                 0 = only BM25, 1 = only embeddings
        """
        # Get BM25-style scores
        cleaned_query = self.clean_text(query)
        query_words = cleaned_query.split()
        bm25_scores = Counter()

        relevant_memory_ids = self.user_memories.get(user_id, []) if user_id else range(len(self.memories))

        for word in query_words:
            for memory_id in self.inverted_index.get(word, []):
                if memory_id in relevant_memory_ids:
                    bm25_scores[memory_id] += 1

        # Normalize BM25 scores
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
        normalized_bm25 = {mid: score/max_bm25 for mid, score in bm25_scores.items()}

        # Get embedding similarity scores
        embedding_scores = {}
        try:
            query_embedding = await get_embeddings(query)
            
            for memory_id in relevant_memory_ids:
                if memory_id in self.embeddings:
                    similarity = self._cosine_similarity(query_embedding, self.embeddings[memory_id])
                    embedding_scores[memory_id] = similarity
        except Exception as e:
            logging.warning(f"Failed to compute embedding similarities: {e}")
            # Fall back to BM25-only if embeddings fail
            hybrid_weight = 0

        # Combine scores
        combined_scores = {}
        for memory_id in relevant_memory_ids:
            bm25_score = normalized_bm25.get(memory_id, 0)
            embedding_score = embedding_scores.get(memory_id, 0)
            combined_scores[memory_id] = (
                (1 - hybrid_weight) * bm25_score + 
                hybrid_weight * embedding_score
            )

        # Sort and deduplicate results
        sorted_memories = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Deduplication process
        results = []
        total_tokens = 0
        seen_content = set()  # Track unique content fingerprints
        
        for memory_id, score in sorted_memories:
            memory = self.memories[memory_id]
            memory_tokens = self.count_tokens(memory)

            if total_tokens + memory_tokens > self.max_tokens:
                break

            # Create a content fingerprint by cleaning and normalizing the text
            cleaned_memory = self.clean_text(memory)
            
            # Check for similar content using n-gram comparison
            is_duplicate = False
            for seen in seen_content:
                similarity = self._calculate_similarity(cleaned_memory, seen)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                results.append((memory, score))
                seen_content.add(cleaned_memory)
                total_tokens += memory_tokens
                
                if len(results) >= k:
                    break

        logging.info(f"Found {len(results)} unique memories for query: {query[:100]}...")
        return results

    def _calculate_similarity(self, text1, text2):
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
    def count_tokens(self, text):
        """Count the number of tokens in text.

        Args:
            text (str): Text to count tokens in

        Returns:
            int: Number of tokens
        """
        return len(self.tokenizer.encode(text))
    

    # Chunking Mechanism
    def _create_chunks(self, text: str, max_tokens: int = 512) -> List[str]:
        """Split text into chunks of approximately max_tokens size while preserving sentence boundaries.
        
        Args:
            text (str): Text to split into chunks
            max_tokens (int, optional): Maximum tokens per chunk. Defaults to 512.
            
        Returns:
            List[str]: List of text chunks
        """
        # First split into paragraphs to preserve major text boundaries
        paragraphs = text.split('\n\n')
        
        # Initialize variables
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            # Split paragraph into sentences
            sentences = sent_tokenize(paragraph)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Count tokens in this sentence
                sentence_tokens = self.count_tokens(sentence)
                
                # Handle sentences that are themselves longer than max_tokens
                if sentence_tokens > max_tokens:
                    # If we have content in current_chunk, add it to chunks
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    # Split long sentence into smaller pieces
                    sentence_chunks = self._split_long_sentence(sentence, max_tokens)
                    chunks.extend(sentence_chunks)
                    continue
                
                # Check if adding this sentence would exceed the limit
                if current_length + sentence_tokens > max_tokens:
                    # Save current chunk and start a new one
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_tokens
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_length += sentence_tokens
        
        # Add any remaining content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Post-process chunks to ensure clean formatting
        chunks = [self._clean_chunk(chunk) for chunk in chunks]
        
        return chunks

    def _split_long_sentence(self, sentence: str, max_tokens: int) -> List[str]:
        """Split a long sentence into smaller chunks while trying to preserve meaning.
        
        Args:
            sentence (str): Long sentence to split
            max_tokens (int): Maximum tokens per chunk
            
        Returns:
            List[str]: List of sentence fragments
        """
        # First try to split on punctuation
        splits = []
        current_piece = []
        current_length = 0
        
        # Define punctuation to split on (ordered by priority)
        punctuation = [';', ':', ',', ')', '}', ']', '—', '-']
        
        # First split on punctuation
        words = []
        current_word = ""
        
        for char in sentence:
            if char in punctuation:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
            elif char.isspace():
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
            else:
                current_word += char
        
        if current_word:
            words.append(current_word)
        
        for word in words:
            word_tokens = self.count_tokens(word)
            
            # If single word is too long, split it
            if word_tokens > max_tokens:
                if current_piece:
                    splits.append(' '.join(current_piece))
                    current_piece = []
                    current_length = 0
                
                # Split word into pieces of max_tokens
                while word:
                    chunk = word[:max_tokens]
                    splits.append(chunk)
                    word = word[max_tokens:]
                continue
            
            # Check if adding this word would exceed the limit
            if current_length + word_tokens > max_tokens:
                splits.append(' '.join(current_piece))
                current_piece = [word]
                current_length = word_tokens
            else:
                current_piece.append(word)
                current_length += word_tokens
        
        # Add any remaining content
        if current_piece:
            splits.append(' '.join(current_piece))
        
        return splits

    def _clean_chunk(self, chunk: str) -> str:
        """Clean up a chunk by removing extra whitespace and fixing punctuation spacing.
        
        Args:
            chunk (str): Text chunk to clean
            
        Returns:
            str: Cleaned text chunk
        """
        # Remove extra whitespace
        chunk = ' '.join(chunk.split())
        
        # Fix spacing around punctuation
        chunk = re.sub(r'\s+([,.!?:;])', r'\1', chunk)
        
        # Fix spacing after punctuation
        chunk = re.sub(r'([,.!?:;])(\S)', r'\1 \2', chunk)
        
        # Remove spaces before closing brackets/parentheses
        chunk = re.sub(r'\s+([\]\}\)])', r'\1', chunk)
        
        # Remove spaces after opening brackets/parentheses
        chunk = re.sub(r'([\[\{\(])\s+', r'\1', chunk)
        
        return chunk.strip()



    def save_cache(self):
        """Save the current state to cache file."""
        cache_data = {
            'inverted_index': self.inverted_index,
            'memories': self.memories,
            'user_memories': self.user_memories
        }
        with open(os.path.join(self.cache_dir, 'memory_cache.pkl'), 'wb') as f:
            pickle.dump(cache_data, f)
        logging.info("Memory cache saved successfully.")

    def load_cache(self):
        """Load the state from cache file if it exists.

        Returns:
            bool: True if cache was loaded successfully, False otherwise
        """
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

    def _cosine_similarity(self, v1, v2):
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm1 = norm(v1)
        norm2 = norm(v2)
        return dot_product / (norm1 * norm2)
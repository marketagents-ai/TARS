import discord
from discord import TextChannel
from discord.ext import commands

import logging
import asyncio
import os
import mimetypes
import tiktoken
import json
from datetime import datetime

import re
import pickle
from collections import defaultdict, Counter
import yaml
import argparse
from github import Github
from github import GithubException, UnknownObjectException  

import base64
import string
import threading
import nltk
# nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Configuration imports
from config import *
from api_client import initialize_api_client, call_api
from cache_manager import CacheManager

# image handling
from PIL import Image
import io
import traceback


# Set up logging
log_level = os.getenv('LOGLEVEL', 'INFO')
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

# JSONL logging setup
def log_to_jsonl(data):
    with open('bot_log.jsonl', 'a') as f:
        json.dump(data, f)
        f.write('\n')

script_dir = os.path.dirname(os.path.abspath(__file__))

# Persona intensity handling
DEFAULT_PERSONA_INTENSITY = 70
TEMPERATURE = DEFAULT_PERSONA_INTENSITY / 100.0

def update_temperature(intensity):
    """Updates the global temperature parameter for LLM sampling based on persona intensity.
    
    The persona intensity (0-100) is mapped linearly to temperature (0.0-1.0).
    Higher temperatures result in more random/creative outputs, while lower 
    temperatures make the LLM responses more focused and deterministic.
    
    Args:
        intensity (int): Persona intensity value between 0-100
            0 = Most focused/deterministic (temp = 0.0)
            100 = Most random/creative (temp = 1.0)
    """
    global TEMPERATURE
    TEMPERATURE = intensity / 100.0

# Channel summarization
class ChannelSummarizer:
    """A class for summarizing Discord channel messages and threads.

    This class provides functionality to analyze and summarize messages from Discord channels,
    including both main channel messages and thread messages. It tracks participant activity,
    shared file types, and generates content summaries using AI.

    Attributes:
        bot: The Discord bot instance
        cache_manager (CacheManager): Manager for handling cache operations
        max_entries (int): Maximum number of messages to analyze
        prompt_formats (dict): Dictionary of prompt templates
        system_prompts (dict): Dictionary of system prompt templates
    """

    def __init__(self, bot, cache_manager: CacheManager, prompt_formats, system_prompts, max_entries=100):
        """Initialize the ChannelSummarizer.

        Args:
            bot: The Discord bot instance
            cache_manager (CacheManager): Manager for handling cache operations
            prompt_formats (dict): Dictionary of prompt templates
            system_prompts (dict): Dictionary of system prompt templates
            max_entries (int, optional): Maximum messages to analyze. Defaults to 100.
        """
        self.bot = bot
        self.cache_manager = cache_manager
        self.max_entries = max_entries
        self.prompt_formats = prompt_formats
        self.system_prompts = system_prompts

    async def summarize_channel(self, channel_id):
        """Summarize messages from a Discord channel and its threads.

        Analyzes messages from both the main channel and any threads, tracking participant
        activity and generating summaries of the content.

        Args:
            channel_id: ID of the Discord channel to summarize

        Returns:
            str: A formatted summary of the channel activity and content
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return "Channel not found."

        main_messages = []
        threads = defaultdict(list)

        async for message in channel.history(limit=self.max_entries):
            if message.thread:
                threads[message.thread.id].append(message)
            else:
                main_messages.append(message)

        summary = f"Summary of #{channel.name}:\n\n"
        summary += await self._summarize_messages(main_messages, "Main Channel")

        for thread_id, thread_messages in threads.items():
            thread = channel.get_thread(thread_id)
            if thread:
                thread_summary = await self._summarize_messages(thread_messages, f"Thread: {thread.name}")
                summary += f"\n{thread_summary}"

        self.cache_manager.append_to_conversation(str(channel_id), {"summary": summary})

        return summary

    async def _summarize_messages(self, messages, context):
        """Generate a summary of a set of Discord messages.

        Analyzes messages to track participant activity, shared file types,
        and generate a content summary using AI.

        Args:
            messages (list): List of Discord message objects to analyze
            context (str): Context string describing the message source

        Returns:
            str: A formatted summary of the messages
        """
        user_message_counts = defaultdict(int)
        file_types = defaultdict(int)
        content_chunks = []
        
        for message in messages:
            user_message_counts[message.author.name] += 1
            for attachment in message.attachments:
                file_type = attachment.filename.split('.')[-1].lower()
                file_types[file_type] += 1
            
            content_chunks.append(f"{message.author.name}: {message.content}")

        summary = f"{context}\n"
        summary += "Participants:\n"
        for user, count in user_message_counts.items():
            summary += f"- {user}: {count} messages\n"

        if file_types:
            summary += "\nShared Files:\n"
            for file_type, count in file_types.items():
                summary += f"- {file_type}: {count} files\n"

        content_summary = await self._process_chunks(content_chunks, context)
        summary += f"\nContent Summary:\n{content_summary}\n"

        return summary

    async def _process_chunks(self, chunks, context):
        """Process message chunks through the AI to generate a summary.

        Args:
            chunks (list): List of message content chunks to summarize
            context (str): Context string describing the message source

        Returns:
            str: AI-generated summary of the message content

        Raises:
            Exception: If there is an error calling the AI API
        """
        prompt = self.prompt_formats['summarize_channel'].format(
            context=context,
            content="\n".join(reversed(chunks)) #reversed order of the entries from the channel
        )
        
        system_prompt = self.system_prompts['channel_summarization'].replace('{persona_intensity}', str(bot.persona_intensity))
        
        try:
            return await call_api(prompt, context="", system_prompt=system_prompt)
        except Exception as e:
            return f"Error in generating summary: {str(e)}"

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
    """

    def __init__(self, cache_type, max_tokens=MAX_TOKENS, context_chunks=CONTEXT_CHUNKS):
        """Initialize the UserMemoryIndex.

        Args:
            cache_type (str): Type of cache to use
            max_tokens (int, optional): Maximum tokens in search results. Defaults to MAX_TOKENS.
            context_chunks (int, optional): Number of context chunks. Defaults to CONTEXT_CHUNKS.
        """
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

    def add_memory(self, user_id, memory_text):
        """Add a new memory for a user.

        Args:
            user_id (str): ID of the user this memory belongs to
            memory_text (str): Text content of the memory
        """
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

    def search(self, query, k=5, user_id=None, similarity_threshold=0.85):
        """Search for relevant memories matching a query, removing duplicates.

        Args:
            query (str): Search query text
            k (int, optional): Maximum number of results to return. Defaults to 5.
            user_id (str, optional): If provided, only search this user's memories.
            similarity_threshold (float, optional): Threshold for considering memories as duplicates. 
                Higher values mean more strict duplicate detection. Defaults to 0.85.

        Returns:
            list: List of tuples containing (memory_text, relevance_score)
        """
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
                if memory_id in relevant_memory_ids:
                    memory_scores[memory_id] += 1
        
        # Normalize scores by memory length
        for memory_id, score in memory_scores.items():
            memory_scores[memory_id] = score / len(self.clean_text(self.memories[memory_id]).split())
        
        sorted_memories = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        
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

# GitHub Integration
"""GitHub repository integration and indexing functionality.

This module provides classes for interacting with GitHub repositories, including fetching
file contents, indexing repository files, and searching through indexed content.

The main components are:

- GitHubRepo: Handles direct interaction with GitHub repositories via the GitHub API
- RepoIndex: Provides indexing and search functionality for repository contents

The integration supports:
- Fetching and caching repository file contents
- Building searchable indexes of repository content
- Directory structure traversal
- File content search and retrieval

Key features:
- Efficient caching of repository contents
- Configurable depth for directory traversal 
- Text cleaning and normalization for search
- Relevance-based file search
"""

class GitHubRepo:
    def __init__(self, token, repo_name):
        self.g = Github(token)
        self.repo = self.g.get_repo(repo_name)

    def get_file_content(self, file_path):
        try:
            file_content = self.repo.get_contents(file_path)
            if file_content.size > 1000000:  # 1MB limit
                return "File is too large to fetch content directly."
            content = base64.b64decode(file_content.content).decode('utf-8')
            return content
        except Exception as e:
            return f"Error fetching file: {str(e)}"

    def get_directory_structure(self, path="", prefix="", max_depth=2, current_depth=0):
        if current_depth > max_depth:
            return []

        contents = self.repo.get_contents(path)
        structure = []
        for content in contents:
            if content.type == "dir":
                structure.append(f"{prefix}{content.name}/")
                if current_depth < max_depth:
                    structure.extend(self.get_directory_structure(
                        content.path, 
                        prefix + "  ", 
                        max_depth, 
                        current_depth + 1
                    ))
            else:
                structure.append(f"{prefix}{content.name}")
        return structure

class RepoIndex:
    def __init__(self, cache_type, max_depth=3):
        self.cache_manager = CacheManager('repo_index')
        self.cache_dir = self.cache_manager.get_cache_dir(cache_type)
        self.max_depth = max_depth
        self.repo_index = defaultdict(list)
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        self.load_cache()

    def index_repo_file(self, file_path, content):
        cleaned_content = self.clean_text(content)
        words = cleaned_content.split()
        for word in words:
            if file_path not in self.repo_index[word]:
                self.repo_index[word].append(file_path)

    def search_repo(self, query, k=5):
        cleaned_query = self.clean_text(query)
        query_words = cleaned_query.split()
        file_scores = Counter()

        for word in query_words:
            for file_path in self.repo_index.get(word, []):
                file_scores[file_path] += 1

        return file_scores.most_common(k)

    def clean_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        return ' '.join(words)

    def clear_cache(self):
        self.repo_index.clear()
        cache_file = os.path.join(self.cache_dir, 'repo_index.pkl')
        if os.path.exists(cache_file):
            os.remove(cache_file)
        logging.info("Repository index cache cleared")

    def save_cache(self):
        with open(os.path.join(self.cache_dir, 'repo_index.pkl'), 'wb') as f:
            pickle.dump(dict(self.repo_index), f)
        logging.info("Repository index cache saved successfully.")

    def load_cache(self):
        repo_index_path = os.path.join(self.cache_dir, 'repo_index.pkl')
        if os.path.exists(repo_index_path):
            with open(repo_index_path, 'rb') as f:
                self.repo_index = defaultdict(list, pickle.load(f))
            logging.info("Repository index cache loaded successfully.")
            return True
        return False

# Create an event for signaling when repository processing is complete
repo_processing_event = asyncio.Event()

async def fetch_and_chunk_repo_contents(repo, memory_index, max_depth=None):
    contents = repo.get_contents("")
    if contents is None:
        logging.error("Failed to fetch repository contents.")
        return

    logging.info(f"Starting to fetch contents for repo: {repo.full_name}")

    async def process_contents(contents, current_depth=0):
        tasks = []
        for content in contents:
            if content.type == "dir":
                if max_depth is None or current_depth < max_depth:
                    dir_contents = repo.get_contents(content.path)
                    await process_contents(dir_contents, current_depth + 1)
            elif content.type == "file":
                tasks.append(asyncio.create_task(process_repofile(content)))

            if len(tasks) >= 10:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)

    async def process_repofile(file_content):
        try:
            _, file_extension = os.path.splitext(file_content.path)
            if file_extension.lower() in ALLOWED_EXTENSIONS:
                logging.debug(f"Processing file: {file_content.path}")
                try:
                    file_data = file_content.decoded_content.decode('utf-8')
                except UnicodeDecodeError:
                    logging.warning(f"UTF-8 decoding failed for {file_content.path}, trying latin-1")
                    try:
                        file_data = file_content.decoded_content.decode('latin-1')
                    except Exception as e:
                        logging.error(f"Failed to decode {file_content.path}: {str(e)}")
                        return
                memory_index.index_repo_file(file_content.path, file_data)
                logging.info(f"Successfully processed file: {file_content.path}")
            else:
                logging.debug(f"Skipping file with unsupported extension: {file_content.path}")
        except Exception as e:
            logging.error(f"Unexpected error processing {file_content.path}: {str(e)}")

    await process_contents(contents)

    memory_index.save_cache()
    logging.info(f"Finished processing repo: {repo.full_name}")

async def start_background_processing(repo, memory_index, max_depth=None, branch='main'):
    global repo_processing_event
    repo_processing_event.clear()
    
    try:
        await process_repo_contents(repo, '', memory_index, max_depth, branch)
        memory_index.save_cache()  # Save the cache after indexing
    except Exception as e:
        logging.error(f"Error in background processing for branch '{branch}': {str(e)}")
    finally:
        repo_processing_event.set()

async def process_repo_contents(repo, path, memory_index, max_depth=None, branch='main', current_depth=0):
    if max_depth is not None and current_depth > max_depth:
        return

    try:
        contents = repo.get_contents(path, ref=branch)
        for content in contents:
            if content.type == 'dir':
                await process_repo_contents(repo, content.path, memory_index, max_depth, branch, current_depth + 1)
            elif content.type == 'file':
                try:
                    file_content = content.decoded_content.decode('utf-8')
                    memory_index.index_repo_file(content.path, file_content)
                    logging.info(f"Indexed file: {content.path} (Branch: {branch})")
                except UnicodeDecodeError:
                    logging.warning(f"Skipped binary file: {content.path} (Branch: {branch})")
                except Exception as file_error:
                    logging.error(f"Error processing file {content.path} on branch '{branch}': {str(file_error)}")
    except Exception as e:
        logging.error(f"Error processing directory {path} on branch '{branch}': {str(e)}")

def start_background_processing_thread(repo, memory_index, max_depth=None, branch='main'):
    thread = threading.Thread(target=run_background_processing, args=(repo, memory_index, max_depth, branch))
    thread.start()
    logging.info(f"Started background processing of repository contents in a separate thread (Branch: {branch}, Max Depth: {max_depth if max_depth is not None else 'Unlimited'})")

# Update the run_background_processing function to include the branch parameter
def run_background_processing(repo, memory_index, max_depth=None, branch='main'):
    global repo_processing_event
    repo_processing_event.clear()
    try:
        asyncio.run(process_repo_contents(repo, '', memory_index, max_depth, branch))
        memory_index.save_cache()  # Save the cache after indexing
    except Exception as e:
        logging.error(f"Error in background processing for branch '{branch}': {str(e)}")
    finally:
        repo_processing_event.set()

# Message processing

async def process_message(message, memory_index, prompt_formats, system_prompts, cache_manager, github_repo, is_command=False):
    """
    Process an incoming Discord message and generate an appropriate response.

    Args:
        message (discord.Message): The Discord message to process
        memory_index (UserMemoryIndex): Index for storing and retrieving user interaction memories
        prompt_formats (dict): Dictionary of prompt templates for different scenarios
        system_prompts (dict): Dictionary of system prompts for different scenarios 
        cache_manager (CacheManager): Manager for caching conversation history
        github_repo (GitHubRepo): GitHub repository interface
        is_command (bool, optional): Whether message is a command. Defaults to False.

    The function:
    1. Extracts message content and user info
    2. Retrieves relevant conversation history and memories
    3. Builds context from channel history and memories
    4. Generates response using LLM API
    5. Saves interaction to memory and conversation history
    6. Logs the interaction
    7. Sends response back to Discord channel

    Raises:
        Exception: For errors in message processing, API calls, or Discord operations
    """
    user_id = str(message.author.id)
    user_name = message.author.name
    
    if is_command:
        content = message.content.split(maxsplit=1)[1]
    else:
        if message.guild and message.guild.me:
            content = message.content.replace(f'<@!{message.guild.me.id}>', '').strip()
        else:
            content = message.content.strip()
    
    logging.info(f"Received message from {user_name} (ID: {user_id}): {content}")

    try:
        async with message.channel.typing():
            is_dm = isinstance(message.channel, discord.DMChannel)
            
            relevant_memories = memory_index.search(
                content, 
                user_id=(user_id if is_dm else None)
            )
            conversation_history = cache_manager.get_conversation_history(user_id)
            
            context = f"Current channel: {message.channel.name if hasattr(message.channel, 'name') else 'Direct Message'}\n\n"
                    
            if conversation_history:
                context += "**Recalled Conversation:**\n\n"
                for msg in conversation_history[-20:]:  # Show last 10 interactions
                    truncated_user_message = truncate_middle(msg['user_message'], max_tokens=256)
                    truncated_ai_response = truncate_middle(msg['ai_response'], max_tokens=256)
                    context += f"***{msg['user_name']}***: {truncated_user_message}\n"
                    context += f"***Discord LLM Agent***: {truncated_ai_response}\n\n"
            else:
                context += f"This is the first interaction with {user_name} (User ID: {user_id}).\n\n"
            
            context += "**Ongoing Chatroom Conversation:**\n\n"
            messages = []
            async for msg in message.channel.history(limit=10):
                truncated_content = truncate_middle(msg.content, max_tokens=256)
                messages.append(f"***{msg.author.name}***: {truncated_content}")
            
            # Reverse the order of messages and add them to the context
            for msg in reversed(messages):
                context += f"{msg}\n"
            context += "\n"
            
            if relevant_memories:
                context += "**Relevant memories:**\n"
                for memory, score in relevant_memories:
                    truncated_memory = truncate_middle(memory, max_tokens=256)
                    context += f"[Relevance: {score:.2f}] {truncated_memory}\n"
                context += "\n"
            
            prompt = prompt_formats['chat_with_memory'].format(
                context=context,
                user_name=user_name,
                user_message=content
            )

            system_prompt = system_prompts['default_chat'].replace('{persona_intensity}', str(bot.persona_intensity))

            response_content = await call_api(prompt, context=context, system_prompt=system_prompt, temperature=TEMPERATURE)

        await send_long_message(message.channel, response_content)
        logging.info(f"Sent response to {user_name} (ID: {user_id}): {response_content[:1000]}...")

        memory_text = f"User {user_name} in {message.channel.name if hasattr(message.channel, 'name') else 'DM'}: {content}\nYour response: {response_content}"
        memory_index.add_memory(user_id, memory_text)
        # Use the new function here
        await generate_and_save_thought(memory_index, user_id, memory_text, prompt_formats, system_prompts, bot)

        cache_manager.append_to_conversation(user_id, {
            'user_name': user_name,
            'user_message': content,
            'ai_response': response_content
        })

        log_to_jsonl({
            'event': 'chat_interaction',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'channel': message.channel.name if hasattr(message.channel, 'name') else 'DM',
            'user_message': content,
            'ai_response': response_content
        })

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await message.channel.send(error_message)
        logging.error(f"Error in message processing for {user_name} (ID: {user_id}): {str(e)}")
        log_to_jsonl({
            'event': 'chat_error',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'channel': message.channel.name if hasattr(message.channel, 'name') else 'DM',
            'error': str(e)
        })


async def process_file(message, memory_index, prompt_formats, system_prompts, user_message="", bot=None, temperature=0.7):
    """
    Process a file attachment from a Discord message, handling both images and text files.

    Args:
        message (discord.Message): The Discord message containing the file attachment
        memory_index (UserMemoryIndex): Index for storing user interaction memories
        prompt_formats (dict): Dictionary of prompt templates for different scenarios
        system_prompts (dict): Dictionary of system prompts for different scenarios
        user_message (str, optional): Additional message from user about the file. Defaults to empty string.
        bot (Bot, optional): Bot instance for accessing persona settings. Defaults to None.
        temperature (float, optional): Temperature parameter for API calls. Defaults to 0.7.

    Raises:
        ValueError: If no attachment is found or if image processing fails
        FileNotFoundError: If temporary image file cannot be saved
        Exception: For other processing errors

    The function:
    1. Detects if the attachment is an image or text file
    2. For images:
        - Downloads and verifies the image
        - Saves to temporary file
        - Calls API for image analysis
        - Cleans up temporary file
    3. For text files:
        - Reads and processes content
        - Calls API for text analysis
    4. Stores interaction in memory index
    5. Logs the interaction
    6. Sends response back to Discord channel
    """
    user_id = str(message.author.id)
    user_name = message.author.name
    
    attachment = message.attachments[0] if message.attachments else None
    if not attachment:
        raise ValueError("No attachment found in message")

    logging.info(f"Processing file '{attachment.filename}' from {user_name} (ID: {user_id}) with message: {user_message}")

    try:
        is_image = attachment.content_type and attachment.content_type.startswith('image/')
        
        # Get persona intensity early to ensure consistent usage
        persona_intensity = str(bot.persona_intensity if bot else DEFAULT_PERSONA_INTENSITY)
        logging.info(f"Using persona intensity: {persona_intensity}")
        
        # Build context once, before the image/text handling split
        context = f"Current channel: {message.channel.name if hasattr(message.channel, 'name') else 'Direct Message'}\n\n"
        context += "**Ongoing Chatroom Conversation:**\n\n"
        messages = []
        async for msg in message.channel.history(limit=10):
            truncated_content = truncate_middle(msg.content, max_tokens=256)
            messages.append(f"***{msg.author.name}***: {truncated_content}")
        
        # Reverse the order of messages and add them to the context
        for msg in reversed(messages):
            context += f"{msg}\n"
        context += "\n"

        async with message.channel.typing():
            if is_image:
                temp_path = f"temp_{attachment.filename}"
                try:
                    # Download the image
                    image_data = await attachment.read()
                    logging.info(f"Downloaded image data: {len(image_data)} bytes")

                    # Verify image data
                    try:
                        img = Image.open(io.BytesIO(image_data))
                        img.verify()
                        logging.info(f"Image verified: {img.format}, {img.size}, {img.mode}")
                    except Exception as img_error:
                        logging.error(f"Image verification failed: {str(img_error)}")
                        logging.error(traceback.format_exc())
                        raise ValueError(f"Invalid image data: {str(img_error)}")

                    # Save image to temporary file
                    with open(temp_path, 'wb') as f:
                        f.write(image_data)
                    logging.info(f"Saved image to temporary path: {temp_path}")
                    
                    if not os.path.exists(temp_path):
                        raise FileNotFoundError(f"Failed to save image: {temp_path} not found")
                    
                    image_prompt = prompt_formats.get('analyze_image', "").format(
                        context=context,
                        filename=attachment.filename,
                        user_message=user_message if user_message else "Please analyze this image."
                    )
                    
                    # Add persona intensity to image analysis system prompt
                    system_prompt = system_prompts.get('image_analysis', "").replace(
                        '{persona_intensity}',
                        persona_intensity
                    )
                    
                    logging.info(f"Calling API with image path: {temp_path}")
                    logging.info(f"Image prompt: {image_prompt}")
                    logging.info(f"System prompt: {system_prompt}")
                    try:
                        response_content = await call_api(
                            prompt=image_prompt,
                            system_prompt=system_prompt,
                            image_paths=[temp_path],
                            temperature=temperature
                        )
                        logging.info("API call with image successful")
                        logging.info(f"API response: {response_content[:100]}...")
                    except Exception as api_error:
                        logging.error(f"Error in API call: {str(api_error)}")
                        logging.error(traceback.format_exc())
                        raise ValueError(f"API call failed: {str(api_error)}")
                except Exception as e:
                    logging.error(f"Error processing image: {str(e)}")
                    logging.error(traceback.format_exc())
                    raise
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        logging.info(f"Removed temporary image file: {temp_path}")
            else:
                # Text file handling
                try:
                    file_content = await attachment.read()
                    file_content = file_content.decode('utf-8')
                    logging.info(f"Read text file content: {len(file_content)} characters")
                    
                    file_prompt = prompt_formats['analyze_file'].format(
                        context=context,
                        filename=attachment.filename,
                        file_content=file_content[:],
                        user_message=user_message
                    )

                    # Use consistent persona intensity for text file analysis
                    system_prompt = system_prompts['file_analysis'].replace(
                        '{persona_intensity}', 
                        persona_intensity
                    )

                    response_content = await call_api(
                        prompt=file_prompt,
                        system_prompt=system_prompt,
                        temperature=temperature
                    )
                except Exception as e:
                    logging.error(f"Error processing text file: {str(e)}")
                    raise

        await send_long_message(message.channel, response_content)
        
        # Memory and logging
        memory_text = f"Analyzed {'image' if is_image else 'file'} '{attachment.filename}' for User {user_name}. User's message: {user_message}. Analysis: {response_content}"
        await generate_and_save_thought(memory_index, user_id, memory_text, prompt_formats, system_prompts, bot)

        log_to_jsonl({
            'event': 'file_analysis',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'filename': attachment.filename,
            'file_type': 'image' if is_image else 'document',
            'user_message': user_message,
            'ai_response': response_content
        })

    except Exception as e:
        error_message = f"An error occurred while analyzing the {'image' if is_image else 'file'} '{attachment.filename}': {str(e)}"
        await message.channel.send(error_message)
        logging.error(f"Error in {'image' if is_image else 'file'} analysis for {user_name} (ID: {user_id}): {str(e)}")
        logging.error(traceback.format_exc())

async def send_long_message(channel, message):
    """
    Splits and sends a long message into smaller chunks to Discord to handle message length limits.
    
    Args:
        channel: The Discord channel to send the message to
        message: The full message text to be sent
        
    The function splits the message into sentences and creates chunks under Discord's 2000 character limit,
    preserving sentence boundaries. Each chunk is sent sequentially to maintain message flow.
    """
    # First, split the message into sentences
    sentences = sent_tokenize(message)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit, start a new chunk
        if len(current_chunk) + len(sentence) > 1900:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Send each chunk
    for chunk in chunks:
        await channel.send(chunk)

def truncate_middle(text, max_tokens=256):
    """
    Truncates text to a maximum number of tokens while preserving content from both ends.
    
    Args:
        text (str): The input text to truncate
        max_tokens (int, optional): Maximum number of tokens to keep. Defaults to 256.
        
    Returns:
        str: The truncated text with ellipsis (...) in the middle if truncation was needed,
             or the original text if it was already within the token limit.
             
    The function preserves roughly equal amounts of tokens from the start and end of the text,
    placing an ellipsis in the middle to indicate truncation. It uses the cl100k_base tokenizer
    from tiktoken for token counting and manipulation.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    keep_tokens = max_tokens - 3  # Account for the ellipsis (...) in the middle
    side_tokens = keep_tokens // 2
    end_tokens = side_tokens + (keep_tokens % 2)  # Add the remainder to the end if odd
    
    truncated_tokens = tokens[:side_tokens] + [tokenizer.encode('...')[0]] + tokens[-end_tokens:]
    return tokenizer.decode(truncated_tokens)

async def generate_and_save_thought(memory_index, user_id, memory_text, prompt_formats, system_prompts, bot):
    """
    Generates a thought about a memory and saves both to the memory index.
    
    Args:
        memory_index (UserMemoryIndex): The memory index to store memories and thoughts
        user_id (str): ID of the user the memory/thought is about
        memory_text (str): The original memory text to generate a thought about
        prompt_formats (dict): Dictionary containing prompt templates
        system_prompts (dict): Dictionary containing system prompt templates
        bot (commands.Bot): The bot instance containing persona settings
        
    The function:
    1. Generates a thought about the memory using the AI API
    2. Saves both the original memory and generated thought to the memory index
    3. Uses the bot's persona intensity for thought generation
    """
    thought_prompt = prompt_formats['generate_thought'].format(
        memory_text=memory_text
    )
    
    thought_system_prompt = system_prompts['thought_generation'].replace('{persona_intensity}', str(bot.persona_intensity))
    
    thought_response = await call_api(thought_prompt, context="", system_prompt=thought_system_prompt, temperature=TEMPERATURE)
    
    # Save both the original memory and the thought
    memory_index.add_memory(user_id, memory_text)
    memory_index.add_memory(user_id, f"Thought about recent interaction: {thought_response}")


# Bot setup

def setup_bot():
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    bot = commands.Bot(command_prefix='!', intents=intents)

    user_memory_index = UserMemoryIndex(cache_type='user_memory_index')
    repo_index = RepoIndex(cache_type='repo_index')
    cache_manager = CacheManager('conversation_history')
    github_repo = GitHubRepo(GITHUB_TOKEN, REPO_NAME)

    with open(os.path.join(script_dir, 'prompts', 'prompt_formats.yaml'), 'r') as file:
        prompt_formats = yaml.safe_load(file)
    
    with open(os.path.join(script_dir, 'prompts', 'system_prompts.yaml'), 'r') as file:
        system_prompts = yaml.safe_load(file)

    # Add this variable to store the current persona intensity
    bot.persona_intensity = DEFAULT_PERSONA_INTENSITY


    @bot.event
    async def on_ready():
        logging.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
        logging.info('------')
        log_to_jsonl({
            'event': 'bot_ready',
            'timestamp': datetime.now().isoformat(),
            'bot_name': bot.user.name,
            'bot_id': bot.user.id
        })

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        # Check if the message is a command
        ctx = await bot.get_context(message)
        if ctx.valid:
            await bot.invoke(ctx)
            return

        if isinstance(message.channel, discord.DMChannel) or bot.user in message.mentions:
            if message.attachments:
                attachment = message.attachments[0]
                if attachment.size <= 1000000:  # 1MB limit
                    try:
                        await process_file(
                            message=message,
                            memory_index=user_memory_index,
                            prompt_formats=prompt_formats,
                            system_prompts=system_prompts,
                            user_message=message.content,
                            bot=bot
                        )
                    except Exception as e:
                        await message.channel.send(f"Error processing file: {str(e)}")
                else:
                    await message.channel.send("File is too large. Please upload a file smaller than 1 MB.")
            else:
                await process_message(message, user_memory_index, prompt_formats, system_prompts, cache_manager, github_repo)
                
    @bot.command(name='persona')
    async def set_persona_intensity(ctx, intensity: int = None):
        """Set or get the AI's persona intensity (0-100). The intensity can be steered through in context prompts and it also adjusts the temperature of the API calls."""
        if intensity is None:
            await ctx.send(f"Current persona intensity is {bot.persona_intensity}%.")
            logging.info(f"Persona intensity queried by user {ctx.author.name} (ID: {ctx.author.id})")
        elif 0 <= intensity <= 100:
            bot.persona_intensity = intensity
            update_temperature(intensity)  # Update the temperature
            await ctx.send(f"Persona intensity set to {intensity}%.")
            logging.info(f"Persona intensity set to {intensity}% by user {ctx.author.name} (ID: {ctx.author.id})")
        else:
            await ctx.send("Please provide a valid intensity between 0 and 100.")


    @bot.command(name='add_memory')
    async def add_memory(ctx, *, memory_text):
        """Add a new memory to the AI."""
        user_memory_index.add_memory(str(ctx.author.id), memory_text)
        await ctx.send("Memory added successfully.")
        log_to_jsonl({
            'event': 'add_memory',
            'timestamp': datetime.now().isoformat(),
            'user_id': str(ctx.author.id),
            'user_name': ctx.author.name,
            'memory_text': memory_text
        })

    @bot.command(name='clear_memories')
    async def clear_memories(ctx):
        """Clear all memories of the invoking user."""
        user_id = str(ctx.author.id)
        user_memory_index.clear_user_memories(user_id)
        await ctx.send("Your memories have been cleared.")
        log_to_jsonl({
            'event': 'clear_user_memories',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': ctx.author.name
        })

    @bot.command(name='analyze_file')
    async def analyze_file(ctx):
        """Analyze an uploaded file."""
        if not ctx.message.attachments:
            await ctx.send("Please upload a file to analyze.")
            return

        attachment = ctx.message.attachments[0]
        
        if attachment.size > 1000000:  # 1 MB limit
            await ctx.send("File is too large. Please upload a file smaller than 1 MB.")
            return

        try:
            file_content = await attachment.read()
            file_content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            await ctx.send("Unable to read the file. Please ensure it's a text file.")
            return

        await process_file(ctx, file_content, attachment.filename, user_memory_index, prompt_formats, system_prompts)

    @bot.command(name='summarize')
    async def summarize(ctx, *, args=None):
        """Summarize the last n messages in a specified channel and send the summary to DM."""
        try:
            n = 100  # Default value
            channel = None

            if args:
                parts = args.split()
                if len(parts) >= 1:
                    # Check if the first part is a channel mention or ID
                    if parts[0].startswith('<#') and parts[0].endswith('>'):
                        channel_id = int(parts[0][2:-1])
                    elif parts[0].isdigit():
                        channel_id = int(parts[0])
                    else:
                        await ctx.send("Please provide a valid channel ID or mention.")
                        return
                    
                    channel = bot.get_channel(channel_id)
                    if channel is None:
                        await ctx.send(f"Invalid channel. Channel ID: {channel_id}")
                        return
                    parts = parts[1:]  # Remove the channel mention/ID from parts

                    # Check if there's a number provided
                    if parts:
                        try:
                            n = int(parts[0])
                        except ValueError:
                            await ctx.send("Invalid input. Please provide a number for the amount of messages to summarize.")
                            return
            else:
                await ctx.send("Please specify a channel ID or mention to summarize.")
                return

            # Log the attempt
            logging.info(f"Attempting to summarize {n} messages from channel {channel.name} (ID: {channel.id})")

            # Check permissions
            member = channel.guild.get_member(ctx.author.id)
            if member is None or not channel.permissions_for(member).read_messages:
                await ctx.send(f"You don't have permission to read messages in the specified channel.")
                return
            
            if not channel.permissions_for(channel.guild.me).read_message_history:
                await ctx.send(f"I don't have permission to read message history in the specified channel.")
                return

            summarizer = ChannelSummarizer(bot, cache_manager, prompt_formats, system_prompts, max_entries=n)
            summary = await summarizer.summarize_channel(channel.id)
            
            # Send the summary as a DM to the user
            try:
                await send_long_message(ctx.author, f"**Channel Summary for #{channel.name} (Last {n} messages)**\n\n{summary}")
                
                # Confirm that the summary was sent
                if isinstance(ctx.channel, discord.DMChannel):
                    await ctx.send(f"I've sent you the summary of #{channel.name}.")
                else:
                    await ctx.send(f"{ctx.author.mention}, I've sent you a DM with the summary of #{channel.name}.")
            except discord.Forbidden:
                await ctx.send("I couldn't send you a DM. Please check your privacy settings and try again.")

            # Save and generate thought
            memory_text = f"Summarized {n} messages from #{channel.name}. Summary: {summary}"
            await generate_and_save_thought(user_memory_index, str(ctx.author.id), memory_text, prompt_formats, system_prompts, bot)

        except discord.Forbidden as e:
            await ctx.send(f"I don't have permission to perform this action. Error: {str(e)}")
        except Exception as e:
            error_message = f"An error occurred while summarizing the channel: {str(e)}"
            await ctx.send(error_message)
            logging.error(f"Error in channel summarization: {str(e)}")

    @bot.command(name='index_repo')
    async def index_repo(ctx, option: str = None, branch: str = 'main'):
        """Index the GitHub repository contents, list indexed files, or check indexing status."""
        global repo_processing_event

        if option == 'list':
            if repo_processing_event.is_set():
                indexed_files = set()
                for file_paths in repo_index.repo_index.values():
                    indexed_files.update(file_paths)
                
                if indexed_files:
                    file_list = f"# Indexed Repository Files (Branch: {branch})\n\n"
                    for file in sorted(indexed_files):
                        file_list += f"- `{file}`\n"
                    
                    temp_file = 'indexed_files.md'
                    with open(temp_file, 'w') as f:
                        f.write(file_list)
                    
                    await ctx.send(f"Here's the list of indexed files from the '{branch}' branch:", file=discord.File(temp_file))
                    
                    os.remove(temp_file)
                else:
                    await ctx.send(f"No files have been indexed yet on the '{branch}' branch.")
            else:
                await ctx.send(f"Repository indexing has not been completed for the '{branch}' branch. Please run `!index_repo` first.")
        elif option == 'status':
            if repo_processing_event.is_set():
                await ctx.send("Repository indexing is complete.")
            else:
                await ctx.send("Repository indexing is still in progress.")
        else:
            if not repo_processing_event.is_set():
                try:
                    await ctx.send(f"Starting to index the repository on the '{branch}' branch... This may take a while.")
                    
                    # Clear existing cache before starting new indexing
                    repo_index.clear_cache()
                    
                    start_background_processing_thread(github_repo.repo, repo_index, max_depth=None, branch=branch)
                    await ctx.send(f"Repository indexing has started in the background for the '{branch}' branch. You will be notified when it's complete.")
                except Exception as e:
                    error_message = f"An error occurred while starting the repository indexing on the '{branch}' branch: {str(e)}"
                    await ctx.send(error_message)
                    logging.error(error_message)
            else:
                await ctx.send(f"Repository indexing is already in progress or completed for the '{branch}' branch. Use `!index_repo list` to see indexed files or `!index_repo status` to check the current status.")

    @bot.command(name='generate_prompt')
    async def generate_prompt_command(ctx, *, input_text):
        parts = input_text.split(maxsplit=1)
        if len(parts) < 2:
            await ctx.send("Error: Please provide both a file path and a task description.")
            return

        file_path = parts[0]
        user_task_description = parts[1]

        logging.info(f"Received generate_prompt command: {file_path}, {user_task_description}")
        
        if not repo_processing_event.is_set():
            await ctx.send("Repository indexing is not complete. Please run !index_repo first.")
            return

        try:
            # Normalize the file path
            file_path = file_path.strip().replace('\\', '/')
            if file_path.startswith('/'):
                file_path = file_path[1:]  # Remove leading slash if present
            
            logging.info(f"Normalized file path: {file_path}")

            # Check if the file is in the indexed repository
            indexed_files = set()
            for file_set in repo_index.repo_index.values():
                indexed_files.update(file_set)
            
            if file_path not in indexed_files:
                await ctx.send(f"Error: The file '{file_path}' is not in the indexed repository.")
                return

            # Fetch the file content using GitHubRepo class
            repo_code = github_repo.get_file_content(file_path)
            
            if repo_code.startswith("Error fetching file:"):
                await ctx.send(f"Error: {repo_code}")
                return
            elif repo_code == "File is too large to fetch content directly.":
                await ctx.send(repo_code)
                return

            logging.info(f"Successfully fetched file: {file_path}")

            # Determine the code type based on file extension
            _, file_extension = os.path.splitext(file_path)
            code_type = mimetypes.types_map.get(file_extension, "").split('/')[-1] or "plaintext"
            
            # Check if the necessary keys exist in prompt_formats and system_prompts
            if 'generate_prompt' not in prompt_formats:
                await ctx.send("Error: 'generate_prompt' template is missing from prompt_formats.")
                return
            if 'generate_prompt' not in system_prompts:
                await ctx.send("Error: 'generate_prompt' is missing from system_prompts.")
                return
            
            # Add channel context
            context = f"Current channel: {ctx.channel.name if hasattr(ctx.channel, 'name') else 'Direct Message'}\n\n"
            context += "**Ongoing Chatroom Conversation:**\n\n"
            messages = []
            async for msg in ctx.channel.history(limit=10):
                truncated_content = truncate_middle(msg.content, max_tokens=256)
                messages.append(f"***{msg.author.name}***: {truncated_content}")
            
            # Reverse the order of messages and add them to the context
            for msg in reversed(messages):
                context += f"{msg}\n"
            context += "\n"

            prompt = prompt_formats['generate_prompt'].format(
                file_path=file_path,
                code_type=code_type,
                repo_code=repo_code,
                user_task_description=user_task_description,
                context=context
            )
            
            # Replace persona_intensity in the system prompt
            system_prompt = system_prompts['generate_prompt'].replace('{persona_intensity}', str(bot.persona_intensity))
            
            response_content = await call_api(prompt, system_prompt=system_prompt)
             
            # Create a safe filename from the user_task_description
            safe_filename = re.sub(r'[^\w\-_\. ]', '_', user_task_description)
            safe_filename = safe_filename[:50]  # Limit filename length
            md_filename = os.path.join(TEMP_DIR, f"{safe_filename}.md")
            
            # Write the response to a Markdown file
            with open(md_filename, 'w', encoding='utf-8') as md_file:
                md_file.write(f"# Generated Prompt for: {user_task_description}\n\n")
                md_file.write(f"File: `{file_path}`\n\n")
                md_file.write(response_content)
            
            # Send the Markdown file to the chat
            await ctx.send(f"Generated response for: {user_task_description}", file=discord.File(md_filename))
            
            logging.info(f"Sent AI response as Markdown file: {md_filename}")

            # Optionally, remove the file after sending
            os.remove(md_filename)

            # Generate and save thought
            memory_text = f"Generated prompt for file '{file_path}' with task description '{user_task_description}'. Response: {response_content}"
            await generate_and_save_thought(user_memory_index, str(ctx.author.id), memory_text, prompt_formats, system_prompts, bot)

        except Exception as e:
            error_message = f"Error generating prompt and querying AI: {str(e)}"
            await ctx.send(error_message)
            logging.error(error_message)

    @bot.command(name='ask_repo')
    async def ask_repo(ctx, *, question):
        """Chat about the GitHub repository contents."""
        try:
            if not repo_processing_event.is_set():
                await ctx.send("Repository indexing is not complete. Please wait or run !index_repo first.")
                return

            relevant_files = repo_index.search_repo(question)
            if not relevant_files:
                await ctx.send("No relevant files found in the repository for this question.")
                return

            context = "Relevant files in the repository:\n"
            file_links = []  # List to store file links
            for file_path, score in relevant_files:
                context += f"- {file_path} (Relevance: {score:.2f})\n"
                file_content = github_repo.get_file_content(file_path)
                context += f"Content preview: {truncate_middle(file_content, 1000)}\n\n"
                file_links.append(f"{file_path}")

            # Add channel context
            context += f"\nCurrent channel: {ctx.channel.name if hasattr(ctx.channel, 'name') else 'Direct Message'}\n\n"
            context += "**Ongoing Chatroom Conversation:**\n\n"
            messages = []
            async for msg in ctx.channel.history(limit=10):
                truncated_content = truncate_middle(msg.content, max_tokens=256)
                messages.append(f"***{msg.author.name}***: {truncated_content}")
            
            # Reverse the order of messages and add them to the context
            for msg in reversed(messages):
                context += f"{msg}\n"
            context += "\n"

            prompt = prompt_formats['ask_repo'].format(
                context=context,
                question=question
            )

            system_prompt = system_prompts['ask_repo']

            # Replace persona_intensity in the system prompt
            system_prompt = system_prompts['ask_repo'].replace('{persona_intensity}', str(bot.persona_intensity))
            
            response = await call_api(prompt, context=context, system_prompt=system_prompt)

            # Append file links to the response wrapped in a Markdown code block
            response += "\n\nReferenced Files:\n```md\n" + "\n".join(file_links) + "\n```"
            
            await send_long_message(ctx, response)
            logging.info(f"Sent repo chat response for question: {question[:100]}...")

            # Generate and save thought
            memory_text = f"Asked repo question '{question}'. Response: {response}"
            await generate_and_save_thought(user_memory_index, str(ctx.author.id), memory_text, prompt_formats, system_prompts, bot)

        except Exception as e:
            error_message = f"An error occurred while processing the repo chat: {str(e)}"
            await ctx.send(error_message)
            logging.error(f"Error in repo chat: {str(e)}")

    @bot.command(name='search_memories')
    async def search_memories(ctx, *, query):
        """Test the memory search function."""
        is_dm = isinstance(ctx.channel, discord.DMChannel)
        user_id = str(ctx.author.id) if is_dm else None
        
        results = user_memory_index.search(query, user_id=user_id)
        
        if not results:
            await ctx.send("No results found.")
            return
            
        current_chunk = f"Search results for '{query}':\n"
        
        for memory, score in results:
            # First truncate the memory content
            truncated_memory = truncate_middle(memory, 800)
            result_line = f"[Relevance: {score:.2f}] {truncated_memory}\n"
            
            # Ensure single result doesn't exceed limit
            if len(result_line) > 1900:
                result_line = result_line[:1896] + "...\n"
            
            # Check if adding this line would exceed Discord's limit
            if len(current_chunk) + len(result_line) > 1900:
                await ctx.send(current_chunk)
                current_chunk = result_line
            else:
                current_chunk += result_line
        
        # Send any remaining content
        if current_chunk:
            await ctx.send(current_chunk)

    return bot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Discord bot with selected API and model')
    parser.add_argument('--api', choices=['ollama', 'openai', 'anthropic'], default='ollama', help='Choose the API to use (default: ollama)')
    parser.add_argument('--model', type=str, help='Specify the model to use. If not provided, defaults will be used based on the API.')
    args = parser.parse_args()

    initialize_api_client(args)

    bot = setup_bot()
    bot.run(TOKEN)
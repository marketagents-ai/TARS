from collections import defaultdict, Counter
import logging
import pickle
import os
import re
import string
import asyncio
import threading
import tiktoken
from cache_manager import CacheManager

# Set up logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define ALLOWED_EXTENSIONS
ALLOWED_EXTENSIONS = {'.py', '.js', '.ts', '.html', '.css', '.md', '.txt', '.json', '.yaml', '.yml'}

class RepoIndex:
    def __init__(self, cache_type, max_depth=3):
        self.cache_manager = CacheManager('repo_index')
        self.cache_dir = self.cache_manager.get_cache_dir(cache_type)
        self.max_depth = max_depth
        self.repo_index = defaultdict(list)
        self.inverted_index = defaultdict(list)
        self.memories = []
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.load_cache()
        self.indexing_status = "Not started"

    def index_repo_file(self, file_path, content):
        cleaned_content = self.clean_text(content)
        words = cleaned_content.split()
        for word in words:
            if word not in self.repo_index:
                self.repo_index[word] = set()
            self.repo_index[word].add(file_path)

    def search_repo(self, query, k=5):
        cleaned_query = self.clean_text(query)
        query_words = cleaned_query.split()
        file_scores = Counter()

        for word in query_words:
            for file_path in self.repo_index.get(word, set()):
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
        self.inverted_index.clear()
        self.memories.clear()
        self.repo_index.clear()
        for file_name in ['inverted_index.pkl', 'memories.pkl', 'repo_index.pkl']:
            cache_file = os.path.join(self.cache_dir, file_name)
            if os.path.exists(cache_file):
                os.remove(cache_file)
        logging.info("Memory index cache cleared")

    def save_cache(self):
        with open(os.path.join(self.cache_dir, 'inverted_index.pkl'), 'wb') as f:
            pickle.dump(self.inverted_index, f)
        with open(os.path.join(self.cache_dir, 'memories.pkl'), 'wb') as f:
            pickle.dump(self.memories, f)
        with open(os.path.join(self.cache_dir, 'repo_index.pkl'), 'wb') as f:
            pickle.dump(self.repo_index, f)
        logging.info("Memory cache saved successfully.")

    def load_cache(self):
        repo_index_path = os.path.join(self.cache_dir, 'repo_index.pkl')
        if os.path.exists(repo_index_path):
            with open(repo_index_path, 'rb') as f:
                self.repo_index = pickle.load(f)
            logging.info("Repository index cache loaded successfully.")
            return True
        return False

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def get_repo_status(self):
        return self.indexing_status

    def generate_prompt(self):
        # This is a placeholder implementation. You might want to implement a more sophisticated prompt generation method.
        return "This is a generated prompt based on the repository content."

# Create an event for signaling when repository processing is complete
repo_processing_event = asyncio.Event()

async def fetch_and_chunk_repo_contents(repo, memory_index, max_depth=None):
    # Start with an empty path to get the root directory
    structure = repo.get_directory_structure(max_depth=max_depth)
    if not structure:
        logging.error("Failed to fetch repository contents.")
        return

    logging.info(f"Starting to fetch contents for repo: {repo.repo.full_name}")
    memory_index.indexing_status = "In progress"

    async def process_contents(contents):
        tasks = []
        for item in contents:
            if item.endswith('/'):  # It's a directory
                dir_contents = repo.get_directory_structure(item.rstrip('/'), max_depth=1)
                await process_contents(dir_contents)
            else:  # It's a file
                tasks.append(asyncio.create_task(process_file(item)))

            if len(tasks) >= 10:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)

    async def process_file(file_path):
        try:
            _, file_extension = os.path.splitext(file_path)
            if file_extension.lower() in ALLOWED_EXTENSIONS:
                logging.debug(f"Processing file: {file_path}")
                file_content = repo.get_file_content(file_path)
                if isinstance(file_content, str) and file_content.startswith("Error fetching file:"):
                    logging.error(f"Failed to fetch {file_path}: {file_content}")
                    return
                memory_index.index_repo_file(file_path, file_content)
                logging.info(f"Successfully processed file: {file_path}")
            else:
                logging.debug(f"Skipping file with unsupported extension: {file_path}")
        except Exception as e:
            logging.error(f"Unexpected error processing {file_path}: {str(e)}")

    await process_contents(structure)

    memory_index.save_cache()
    memory_index.indexing_status = "Completed"
    logging.info(f"Finished processing repo: {repo.repo.full_name}")

async def start_background_processing(repo, memory_index, max_depth=None):
    global repo_processing_event
    repo_processing_event.clear()
    
    try:
        await fetch_and_chunk_repo_contents(repo, memory_index, max_depth)
    except Exception as e:
        logging.error(f"Error in background processing: {str(e)}")
        memory_index.indexing_status = "Failed"
    finally:
        repo_processing_event.set()

def run_background_processing(repo, memory_index, max_depth=None):
    asyncio.run(start_background_processing(repo, memory_index, max_depth))

def start_background_processing_thread(repo, memory_index, max_depth=None):
    thread = threading.Thread(target=run_background_processing, args=(repo, memory_index, max_depth))
    thread.start()
    logging.info(f"Started background processing of repository contents in a separate thread (Max Depth: {max_depth if max_depth is not None else 'Unlimited'})")

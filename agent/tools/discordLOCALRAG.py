import os
import base64
import logging
import asyncio
import pickle
import string
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LocalDirectory:
    """Handles interaction with local directory structure."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            raise ValueError(f"Directory {root_path} does not exist")

    def get_file_content(self, file_path: str) -> str:
        """Fetch content of a file with size checking."""
        try:
            full_path = self.root_path / file_path
            if not full_path.exists():
                return f"File {file_path} does not exist"
            
            if full_path.stat().st_size > 1_000_000:  # 1MB limit
                return "File is too large to fetch content directly."
                
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(full_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"
        except Exception as e:
            return f"Error fetching file: {str(e)}"

    def get_directory_structure(self, 
                              path: str = "", 
                              prefix: str = "", 
                              max_depth: int = 2, 
                              current_depth: int = 0) -> List[str]:
        """Generate a hierarchical view of the directory structure."""
        if current_depth > max_depth:
            return []

        full_path = self.root_path / path
        structure = []
        
        try:
            for item in sorted(full_path.iterdir()):
                if item.is_dir():
                    structure.append(f"{prefix}{item.name}/")
                    if current_depth < max_depth:
                        structure.extend(
                            self.get_directory_structure(
                                str(item.relative_to(self.root_path)),
                                prefix + "  ",
                                max_depth,
                                current_depth + 1
                            )
                        )
                else:
                    structure.append(f"{prefix}{item.name}")
        except Exception as e:
            logging.error(f"Error reading directory {path}: {str(e)}")
            
        return structure

class DirectoryIndex:
    """Provides indexing and search functionality for directory contents."""
    
    def __init__(self, cache_dir: str, max_depth: int = 3):
        """Initialize the DirectoryIndex.
        
        Args:
            cache_dir (str): Directory to store cache files
            max_depth (int, optional): Maximum depth for directory traversal
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_depth = max_depth
        self.file_index = defaultdict(list)
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 
                            'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        self.load_cache()

    def index_file(self, file_path: str, content: str) -> None:
        """Index a file's content."""
        cleaned_content = self.clean_text(content)
        words = cleaned_content.split()
        
        for word in words:
            if word and file_path not in self.file_index[word]:
                self.file_index[word].append(file_path)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, int]]:
        """Search indexed files based on query."""
        cleaned_query = self.clean_text(query)
        query_words = cleaned_query.split()
        file_scores = Counter()

        for word in query_words:
            for file_path in self.file_index.get(word, []):
                file_scores[file_path] += 1

        return file_scores.most_common(k)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for indexing/searching."""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        return ' '.join(words)

    def clear_cache(self) -> None:
        """Clear the index cache."""
        self.file_index.clear()
        cache_file = self.cache_dir / 'directory_index.pkl'
        if cache_file.exists():
            cache_file.unlink()
        logging.info("Directory index cache cleared")

    def save_cache(self) -> None:
        """Save the current index to cache."""
        cache_file = self.cache_dir / 'directory_index.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump(dict(self.file_index), f)
        logging.info("Directory index cache saved successfully")

    def load_cache(self) -> bool:
        """Load the index from cache if it exists."""
        cache_file = self.cache_dir / 'directory_index.pkl'
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                self.file_index = defaultdict(list, pickle.load(f))
            logging.info("Directory index cache loaded successfully")
            return True
        return False

async def index_directory(directory: LocalDirectory, 
                         index: DirectoryIndex, 
                         allowed_extensions: Set[str],
                         max_depth: int = None) -> None:
    """Index a directory asynchronously."""
    
    async def process_file(file_path: Path) -> None:
        try:
            if file_path.suffix.lower() in allowed_extensions:
                logging.debug(f"Processing file: {file_path}")
                content = directory.get_file_content(str(file_path.relative_to(directory.root_path)))
                if not isinstance(content, str) or content.startswith("Error"):
                    logging.error(f"Failed to read {file_path}: {content}")
                    return
                    
                index.index_file(str(file_path.relative_to(directory.root_path)), content)
                logging.info(f"Successfully processed file: {file_path}")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

    async def process_directory(path: Path, current_depth: int = 0) -> None:
        if max_depth is not None and current_depth > max_depth:
            return

        tasks = []
        try:
            for item in path.iterdir():
                if item.is_file():
                    tasks.append(asyncio.create_task(process_file(item)))
                elif item.is_dir():
                    await process_directory(item, current_depth + 1)

                if len(tasks) >= 10:  # Process in batches of 10
                    await asyncio.gather(*tasks)
                    tasks = []

            if tasks:  # Process remaining tasks
                await asyncio.gather(*tasks)
        except Exception as e:
            logging.error(f"Error processing directory {path}: {str(e)}")

    await process_directory(directory.root_path)
    index.save_cache()
    logging.info(f"Finished indexing directory: {directory.root_path}")

# Example usage
async def main():
    # Initialize with root directory and allowed file extensions
    root_dir = "./my_directory"
    allowed_extensions = {'.txt', '.py', '.md', '.json', '.yml', '.yaml'}
    
    # Create directory handler and index
    local_dir = LocalDirectory(root_dir)
    dir_index = DirectoryIndex("./cache", max_depth=3)
    
    # Index the directory
    await index_directory(local_dir, dir_index, allowed_extensions, max_depth=3)
    
    # Search example
    results = dir_index.search("python async", k=5)
    for file_path, score in results:
        print(f"File: {file_path}, Score: {score}")

if __name__ == "__main__":
    asyncio.run(main())
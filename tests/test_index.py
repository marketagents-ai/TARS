import unittest
import os
import logging
from dotenv import load_dotenv
from tars_bot.inverted_index import InvertedIndexSearch

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestInvertedIndexSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()  # Load environment variables from .env file
        cls.repo_name = os.getenv('GITHUB_REPO')
        cls.index_search = InvertedIndexSearch(repo_name=cls.repo_name)

    def test_process_and_index_content(self):
        with open("principles.md", "r") as f:
            content = f.read()
        
        logging.info("Processing and indexing content...")
        self.index_search.process_and_index_content(content, "principles.md")
        
        logging.info("Inverted Index: %s", self.index_search.inverted_index)
        
        self.assertGreater(len(self.index_search.chunks), 0, "No chunks were created.")
        self.assertGreater(len(self.index_search.inverted_index), 0, "Inverted index is empty.")

    def test_clear_cache(self):
        logging.info("Clearing cache...")
        self.index_search.clear_cache()
        self.assertEqual(len(self.index_search.inverted_index), 0, "Inverted index should be empty after clearing.")
        self.assertEqual(len(self.index_search.chunks), 0, "Chunks should be empty after clearing.")

    
if __name__ == '__main__':
    unittest.main()
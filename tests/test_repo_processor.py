import unittest
import logging
import os
from unittest.mock import MagicMock
from dotenv import load_dotenv
from tars_bot.repo_processor import count_files, fetch_and_chunk_repo_contents, start_background_processing
from github import Github  # Assuming you're using PyGithub

load_dotenv()  # Load environment variables from .env file

class TestRepoProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.github_token = os.getenv('GITHUB_TOKEN')
        cls.repo_name = os.getenv('GITHUB_REPO')
        cls.github = Github(cls.github_token)
        cls.repo = cls.github.get_repo(cls.repo_name)
        cls.inverted_index_search = MagicMock()  # Mock the inverted index search object

    async def test_count_files(self):
        logging.info("Testing count_files function...")
        total_files = await count_files(self.repo, path="", max_depth=None)
        logging.info("Total files counted: %d", total_files)

        self.assertEqual(total_files, 3, "Should count 3 files with allowed extensions.")

    async def test_fetch_and_chunk_repo_contents(self):
        logging.info("Testing fetch_and_chunk_repo_contents function...")
        await fetch_and_chunk_repo_contents(self.repo, self.inverted_index_search, max_depth=None)

        self.assertTrue(self.inverted_index_search.process_and_index_content.called, "Should process and index content.")
        logging.info("Content fetched and processed successfully.")

    async def test_start_background_processing(self):
        logging.info("Testing start_background_processing function...")
        await start_background_processing(self.repo, self.inverted_index_search, max_depth=None)

        self.assertTrue(self.inverted_index_search.load_cache.called, "Should load cache during processing.")
        logging.info("Background processing started successfully.")

if __name__ == "__main__":
    unittest.main()
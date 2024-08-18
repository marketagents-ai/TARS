import os
import logging
import discord
import unittest
from unittest.mock import AsyncMock
from dotenv import load_dotenv
from tars_bot.bot import TarsBot
from tars_bot.github_utils import GithubConnector  # Import the GithubConnector

load_dotenv()  # Load environment variables from .env file

# Set up logging
logging.basicConfig(level=logging.INFO)

class TestTarsBot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load environment variables
        GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
        REPO_NAME = os.getenv('GITHUB_REPO')

        # Initialize GitHub connector
        gh_conn = GithubConnector(GITHUB_TOKEN, REPO_NAME)
        
        # Initialize other components
        inverted_index_search = None  # Replace with actual inverted index search if needed
        api = None  # Replace with actual API if needed
        
        # Initialize the bot
        cls.bot = TarsBot(gh_conn, inverted_index_search, api)

    async def get_context(self):
        # Create a mock context for testing
        ctx = AsyncMock()
        ctx.send = AsyncMock()  # Mock the send method
        ctx.author = discord.Object(id=123456789)  # Mock author
        ctx.message = discord.Object()  # Mock message
        return ctx

    async def test_ai_chat(self):
        logging.info("Testing ai_chat command...")
        ctx = await self.get_context()
        await self.bot.ai_chat(ctx, question="What is AI?")
        
        # Capture the response
        response = ctx.send.call_args[0][0]
        logging.info(f"ai_chat response: {response[:100]}...")  # Log the response (truncated)
        self.assertIn("AI", response, "Response should contain information about AI.")

    async def test_analyze_code(self):
        logging.info("Testing analyze_code command...")
        ctx = await self.get_context()
        await self.bot.analyze_code(ctx, code="print('Hello, World!')")
        
        # Capture the response
        response = ctx.send.call_args[0][0]
        logging.info(f"analyze_code response: {response[:100]}...")  # Log the response (truncated)
        self.assertIn("prints", response, "Response should contain analysis of the code.")

    async def test_repo_chat(self):
        logging.info("Testing repo_chat command...")
        ctx = await self.get_context()
        await self.bot.repo_chat(ctx, question="Tell me about the repository.")
        
        # Capture the response
        response = ctx.send.call_args[0][0]
        logging.info(f"repo_chat response: {response[:100]}...")  # Log the response (truncated)
        self.assertIn("repository", response, "Response should contain information about the repository.")

    async def test_dir_command(self):
        logging.info("Testing dir command...")
        ctx = await self.get_context()
        await self.bot.dir_command(ctx, max_depth=1)  # Ensure the command name matches
        
        # Capture the response
        response = ctx.send.call_args[0][0]
        logging.info(f"dir_command response: {response[:100]}...")  # Log the response (truncated)
        self.assertIn("file", response, "Response should list files in the directory.")

    async def test_clear_history(self):
        logging.info("Testing clear_history command...")
        ctx = await self.get_context()
        await self.bot.clear_history(ctx)
        
        # Capture the response
        response = ctx.send.call_args[0][0]
        logging.info(f"clear_history response: {response[:100]}...")  # Log the response (truncated)
        self.assertIn("cleared", response, "Response should confirm that history has been cleared.")

    async def test_channel_summary(self):
        logging.info("Testing channel_summary command...")
        ctx = await self.get_context()
        await self.bot.channel_summary(ctx, message_count=5)
        
        # Capture the response
        response = ctx.send.call_args[0][0]
        logging.info(f"channel_summary response: {response[:100]}...")  # Log the response (truncated)
        self.assertIn("summary", response, "Response should contain a summary of the channel.")

    async def test_re_index(self):
        logging.info("Testing re_index command...")
        ctx = await self.get_context()
        await self.bot.re_index(ctx)
        
        # Capture the response
        response = ctx.send.call_args[0][0]
        logging.info(f"re_index response: {response[:100]}...")  # Log the response (truncated)
        self.assertIn("re-indexing", response, "Response should confirm that re-indexing has started.")

if __name__ == "__main__":
    unittest.main()
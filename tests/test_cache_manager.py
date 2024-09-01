import os
import unittest
import logging
from tars_bot.cache_manager import CacheManager

class TestCacheManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.repo_name = "test_repo"
        cls.cache_manager = CacheManager(repo_name=cls.repo_name)

    def test_initialization(self):
        logging.info("Testing CacheManager initialization...")
        self.assertEqual(self.cache_manager.repo_name, self.repo_name)
        self.assertEqual(self.cache_manager.max_history, 100)
        self.assertTrue(os.path.exists(self.cache_manager.conversation_dir))
        self.assertTrue(os.path.exists(self.cache_manager.base_cache_dir))

    def test_append_and_get_conversation(self):
        logging.info("Testing append and get conversation...")
        user_id = "user1"
        message = {"text": "Hello, world!"}
        
        self.cache_manager.append_to_conversation(user_id, message)
        history = self.cache_manager.get_conversation_history(user_id=user_id)

        logging.info("Conversation history: %s", history)
        self.assertGreater(len(history), 0, "History should not be empty after appending.")
        self.assertEqual(history[-1], message, "Last message should match the appended message.")

    def test_clear_conversation(self):
        logging.info("Testing clear conversation...")
        user_id = "user1"
        message = {"text": "This will be cleared."}
        
        self.cache_manager.append_to_conversation(user_id, message)
        self.cache_manager.clear_conversation(user_id=user_id)
        history = self.cache_manager.get_conversation_history(user_id=user_id)

        logging.info("Conversation history after clearing: %s", history)
        self.assertEqual(len(history), 0, "History should be empty after clearing.")

    def test_get_ai_chat_history(self):
        logging.info("Testing AI chat history retrieval...")
        user_id = "ai_user"  # Define a user ID for AI chat
        message = {"text": "AI message."}
        
        self.cache_manager.append_to_conversation(user_id=user_id, is_ai_chat=True, message=message)
        history = self.cache_manager.get_conversation_history(is_ai_chat=True)

        logging.info("AI chat history: %s", history)
        self.assertGreater(len(history), 0, "AI chat history should not be empty after appending.")

if __name__ == "__main__":
    unittest.main()
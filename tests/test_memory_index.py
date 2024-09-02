import unittest
from unittest.mock import patch, MagicMock
from TARS.tars_bot.memory_index import UserMemoryIndex

class TestUserMemoryIndex(unittest.TestCase):

    @patch('TARS.tars_bot.memory_index.CacheManager')
    def setUp(self, MockCacheManager):
        self.mock_cache_manager = MockCacheManager.return_value
        self.mock_cache_manager.get_cache_dir.return_value = '/mock/cache/dir'
        self.memory_index = UserMemoryIndex('mock_cache_type', 1000, 5)

    def test_clean_text(self):
        text = "Hello, World! 123"
        cleaned_text = self.memory_index.clean_text(text)
        self.assertEqual(cleaned_text, "hello world")

    def test_add_memory(self):
        user_id = 'user1'
        memory_text = "This is a test memory."
        self.memory_index.add_memory(user_id, memory_text)
        self.assertIn(memory_text, self.memory_index.memories)
        self.assertIn(0, self.memory_index.user_memories[user_id])
        self.assertIn('test', self.memory_index.inverted_index)
        self.assertIn(0, self.memory_index.inverted_index['test'])

    def test_clear_user_memories(self):
        user_id = 'user1'
        memory_text = "This is a test memory."
        self.memory_index.add_memory(user_id, memory_text)
        self.memory_index.clear_user_memories(user_id)
        self.assertNotIn(user_id, self.memory_index.user_memories)
        self.assertIsNone(self.memory_index.memories[0])
        self.assertNotIn(0, self.memory_index.inverted_index['test'])

    def test_search(self):
        user_id = 'user1'
        memory_text = "This is a test memory."
        self.memory_index.add_memory(user_id, memory_text)
        results = self.memory_index.search("test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], memory_text)

    @patch('TARS.tars_bot.memory_index.pickle.dump')
    def test_save_cache(self, mock_pickle_dump):
        self.memory_index.save_cache()
        mock_pickle_dump.assert_called_once()

    @patch('TARS.tars_bot.memory_index.pickle.load')
    @patch('os.path.exists')
    def test_load_cache(self, mock_exists, mock_pickle_load):
        mock_exists.return_value = True
        mock_pickle_load.return_value = {
            'inverted_index': {'test': [0]},
            'memories': ["This is a test memory."],
            'user_memories': {'user1': [0]}
        }
        self.memory_index.load_cache()
        self.assertIn('test', self.memory_index.inverted_index)
        self.assertIn("This is a test memory.", self.memory_index.memories)
        self.assertIn('user1', self.memory_index.user_memories)

if __name__ == '__main__':
    unittest.main()
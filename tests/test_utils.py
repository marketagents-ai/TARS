import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import json
import os
import re
from TARS.tars_bot.utils import (
    log_to_jsonl, process_message, process_file, simple_sent_tokenize,
    send_long_message, truncate_middle, create_markdown_file, send_markdown_file
)

class TestUtils(unittest.TestCase):

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_log_to_jsonl(self, mock_open):
        data = {'key': 'value'}
        log_to_jsonl(data)
        mock_open.assert_called_once_with('bot_log.jsonl', 'a')
        mock_open().write.assert_called_once_with(json.dumps(data) + '\n')

    @patch('TARS.tars_bot.utils.call_api', new_callable=AsyncMock)
    @patch('TARS.tars_bot.utils.send_long_message', new_callable=AsyncMock)
    @patch('TARS.tars_bot.utils.log_to_jsonl')
    @patch('TARS.tars_bot.utils.memory_index')
    @patch('TARS.tars_bot.utils.cache_manager')
    async def test_process_message(self, mock_cache_manager, mock_memory_index, mock_log_to_jsonl, mock_send_long_message, mock_call_api):
        message = MagicMock()
        message.author.id = 123
        message.author.name = 'test_user'
        message.content = '!test command'
        message.channel.name = 'test_channel'
        
        memory_index = MagicMock()
        cache_manager = MagicMock()
        prompt_formats = {'chat_with_memory': '{context} {user_name} {user_message}'}
        system_prompts = {'default_chat': 'default system prompt'}
        
        mock_memory_index.search.return_value = []
        mock_cache_manager.get_conversation_history.return_value = []
        mock_call_api.return_value = 'response from API'
        
        await process_message(message, memory_index, prompt_formats, system_prompts, cache_manager, None)
        
        mock_call_api.assert_called_once()
        mock_send_long_message.assert_called_once()
        mock_log_to_jsonl.assert_called()

    @patch('TARS.tars_bot.utils.call_api', new_callable=AsyncMock)
    @patch('TARS.tars_bot.utils.send_long_message', new_callable=AsyncMock)
    @patch('TARS.tars_bot.utils.log_to_jsonl')
    async def test_process_file(self, mock_log_to_jsonl, mock_send_long_message, mock_call_api):
        ctx = MagicMock()
        ctx.author.id = 123
        ctx.author.name = 'test_user'
        ctx.channel.name = 'test_channel'
        
        file_content = 'file content'
        filename = 'test_file.txt'
        memory_index = MagicMock()
        prompt_formats = {'analyze_file': '{filename} {file_content}'}
        system_prompts = {'file_analysis': 'file analysis system prompt'}
        
        mock_call_api.return_value = 'response from API'
        
        await process_file(ctx, file_content, filename, memory_index, prompt_formats, system_prompts)
        
        mock_call_api.assert_called_once()
        mock_send_long_message.assert_called_once()
        mock_log_to_jsonl.assert_called()

    def test_simple_sent_tokenize(self):
        text = "Hello world! How are you? I'm fine."
        expected = ["Hello world!", "How are you?", "I'm fine."]
        result = simple_sent_tokenize(text)
        self.assertEqual(result, expected)

    @patch('TARS.tars_bot.utils.simple_sent_tokenize')
    @patch('TARS.tars_bot.utils.discord.TextChannel.send', new_callable=AsyncMock)
    async def test_send_long_message(self, mock_send, mock_simple_sent_tokenize):
        channel = MagicMock()
        message = "This is a long message."
        mock_simple_sent_tokenize.return_value = ["This is a long message."]
        
        await send_long_message(channel, message)
        
        mock_send.assert_called_once()

    def test_truncate_middle(self):
        text = "This is a very long text that needs to be truncated."
        result = truncate_middle(text, max_length=20)
        self.assertEqual(result, "This is a...uncated.")

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('os.path.join', return_value='TEMP_DIR/test_file.md')
    def test_create_markdown_file(self, mock_path_join, mock_open):
        filename = "test_file"
        content = "This is the content of the file."
        result = create_markdown_file(filename, content)
        self.assertEqual(result, 'TEMP_DIR/test_file.md')
        mock_open.assert_called_once_with('TEMP_DIR/test_file.md', 'w', encoding='utf-8')
        mock_open().write.assert_called_once_with(content)

    @patch('TARS.tars_bot.utils.create_markdown_file', return_value='TEMP_DIR/test_file.md')
    @patch('TARS.tars_bot.utils.discord.File')
    @patch('TARS.tars_bot.utils.os.remove')
    @patch('TARS.tars_bot.utils.discord.TextChannel.send', new_callable=AsyncMock)
    async def test_send_markdown_file(self, mock_send, mock_remove, mock_file, mock_create_markdown_file):
        ctx = MagicMock()
        filename = "test_file"
        content = "This is the content of the file."
        
        await send_markdown_file(ctx, filename, content)
        
        mock_create_markdown_file.assert_called_once_with(filename, content)
        mock_send.assert_called_once()
        mock_remove.assert_called_once_with('TEMP_DIR/test_file.md')

if __name__ == '__main__':
    unittest.main()
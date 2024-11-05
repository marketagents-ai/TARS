import os
import json
from collections import deque

class CacheManager:
    def __init__(self, repo_name, max_history=5):
        self.repo_name = repo_name
        self.max_history = max_history
        self.base_cache_dir = os.path.join('cache', self.repo_name)
        self.conversation_dir = os.path.join(self.base_cache_dir, 'conversations')
        self.memory_file = os.path.join(self.base_cache_dir, 'memories.json')
        os.makedirs(self.conversation_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

    def get_conversation_history(self, user_id):
        """Retrieves conversation history for a user from JSONL file, up to max_history messages."""
        file_path = os.path.join(self.conversation_dir, f"{user_id}.jsonl")
        history = deque(maxlen=self.max_history)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    history.append(json.loads(line.strip()))
        return list(history)

    def append_to_conversation(self, user_id, message):
        """Appends message to user's conversation history and trims to max history items."""
        history = self.get_conversation_history(user_id)
        history.append(message)
        if len(history) > self.max_history:
            history = history[-self.max_history:]
        file_path = os.path.join(self.conversation_dir, f"{user_id}.jsonl")
        with open(file_path, 'w') as f:
            for item in history:
                f.write(json.dumps(item) + '\n')

    def clear_conversation(self, user_id):
        """Clears conversation history for a user."""
        file_path = os.path.join(self.conversation_dir, f"{user_id}.jsonl")
        if os.path.exists(file_path):
            os.remove(file_path)

    def get_memories(self):
        """Retrieves all memories from the memory file."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {}

    def save_memory(self, key, value):
        """Saves a memory to the memory file."""
        memories = self.get_memories()
        memories[key] = value
        with open(self.memory_file, 'w') as f:
            json.dump(memories, f)

    def get_cache_dir(self, cache_type):
        """Creates and returns a cache directory for a given type."""
        cache_dir = os.path.join(self.base_cache_dir, cache_type)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
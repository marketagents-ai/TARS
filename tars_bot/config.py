import os
from dotenv import load_dotenv

# Force reload of .env file
if os.path.exists('.env'):
    load_dotenv(override=True)

# Bot configuration
TOKEN = os.getenv('DISCORD_TOKEN')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_NAME = os.getenv('GITHUB_REPO')

# API configuration
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')

OLLAMA_API_BASE = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'nousresearch/hermes-3-llama-3.1-405b')

LOCALAI_API_BASE = os.getenv('LOCALAI_API_BASE', 'https://demo.localai.io/v1')
LOCALAI_MODEL = os.getenv('LOCALAI_MODEL', 'Hermes-3-Llama-3.1-8B:vllm')

# Discord configuration
DISCORD_CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')

# File extensions
ALLOWED_EXTENSIONS = {'.md', '.py', '.txt'}

# Inverted Index Search configuration
MAX_TOKENS = 1000
CONTEXT_CHUNKS = 4
CHUNK_PERCENTAGE = 10

# Conversation history
MAX_CONVERSATION_HISTORY = 5

# Temporary directory
TEMP_DIR = os.getenv('TEMP_DIR', 'temp_files')

# Make sure the directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Notion configuration - WIP UNIMPLEMENTED
NOTION_API_KEY = os.getenv('NOTION_API_KEY')
CALENDAR_DB_ID = os.getenv('CALENDAR_DB_ID')
PROJECTS_DB_ID = os.getenv('PROJECTS_DB_ID')
TASKS_DB_ID = os.getenv('TASKS_DB_ID')
KANBAN_DB_ID = os.getenv('KANBAN_DB_ID')

# Polling configuration
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', 120))

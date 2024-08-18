# TARS
A witty, sarcastic and humorous discord AI bot for server and coding assistance
<p align="center">
  <img src="assets/serain-tars4.jpg" alt="Image Alt Text" width="80%" height="80%">
</p>

## Project Summary:

The core pipeline of this project involves:
1. Initializing the Discord bot and connecting to a GitHub repository
2. Processing and indexing the repository contents for efficient searching
3. Handling user interactions through Discord commands and messages
4. Utilizing AI services (Azure, Ollama, or OpenRouter) to generate responses
5. Managing conversation history and search indexing through caching

## User Interaction:

Users can interact with the agent in chat through the following methods:
1. Direct Messages: Users can send private messages to the bot, which will be processed and responded to using the AI service.

2. Commands in Discord channels:
   - !ai_chat: Get assistance from the AI model
   - !analyze_code: Analyze code using the AI model
   - !repo_chat: Chat with the repository using the inverted index search
   - !generate_prompt: Generate a goal-oriented prompt based on repository code and principles
   - !dir: Display the repository file structure
   - !clear_history: Clear the user's conversation history
   - !channel_summary: Summarize the last n messages in the channel
   - !re_index: Re-index the repository and update the cache

These commands allow users to interact with the AI, analyze code, search the repository, and manage their interaction history. The bot processes these commands, retrieves relevant information from the repository when necessary, and generates responses using the configured AI service.

Certainly! I'll provide a high-level overview of your project's flow, note the Input-Process-Output (IPO) for key components, and summarize how users can interact with the agent in chat.

## High-Level Overview:

1. Main Script (main.py):
   - Initializes the environment, logging, and parses command-line arguments
   - Sets up the GitHub repository connection
   - Initializes the API client, cache manager, and inverted index search
   - Sets up the Discord bot
   - Starts background processing of the repository
   - Runs the Discord bot

2. Bot Setup (bot.py):
   - Configures the Discord bot with various commands
   - Handles message events and command processing

3. Repository Processing (repo_processor.py):
   - Fetches and chunks repository contents
   - Indexes the content for efficient searching

4. API Client (api_client.py):
   - Handles communication with different AI APIs (Azure, Ollama, OpenRouter)

5. Inverted Index Search (inverted_index.py):
   - Manages the search functionality for repository contents

6. Cache Manager (cache_manager.py):
   - Handles caching of conversation history and search index

Key IPO and Related Modules:

1. Main Script (main.py):
   - Input: Command-line arguments, environment variables
   - Process: Initialization of components
   - Output: Running Discord bot
   - Related: All other modules

2. Bot Setup (bot.py):
   - Input: Discord events, user messages
   - Process: Command handling, message processing
   - Output: Bot responses, API calls
   - Related: api_client.py, github_utils.py, inverted_index.py, cache_manager.py

3. Repository Processing (repo_processor.py):
   - Input: GitHub repository contents
   - Process: Fetching, chunking, and indexing content
   - Output: Indexed repository content
   - Related: github_utils.py, inverted_index.py

4. API Client (api_client.py):
   - Input: User prompts, context
   - Process: API calls to AI services
   - Output: AI-generated responses
   - Related: bot.py

5. Inverted Index Search (inverted_index.py):
   - Input: Repository content, search queries
   - Process: Indexing, searching
   - Output: Relevant content chunks
   - Related: repo_processor.py, bot.py

6. Cache Manager (cache_manager.py):
   - Input: Conversation data, search index
   - Process: Storing and retrieving cached data
   - Output: Cached conversations, search index
   - Related: bot.py, inverted_index.py

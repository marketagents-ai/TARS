# TARS
A witty, sarcastic and humorous discord AI bot for server and coding assistance
<p align="center">
  <img src="assets/serain-tars4.jpg" alt="Image Alt Text" width="80%" height="80%">
</p>

## Project Summary:

# AI-Powered Discord Bot with Repository Integration

This project implements a Discord bot that integrates with GitHub repositories, provides AI-powered responses, and offers various utility functions for users. The bot is designed to assist with code analysis, repository searching, and can maintain long-term user-specific memories.

## Key Components

### 1. Main Script (main.py)
- Entry point for the application
- Parses command-line arguments for API selection
- Initializes the API client and sets up the Discord bot

### 2. Bot Setup (bot_setup.py)
- Configures the Discord bot with various commands
- Handles message events and command processing
- Integrates with other components like UserMemoryIndex, RepoIndex, and ChannelSummarizer

### 3. API Client (api_client.py)
- Supports multiple AI services: Azure, Ollama, OpenRouter, and LocalAI
- Handles API calls to generate responses
- Implements logging for API calls

### 4. Repository Index (repo_index.py)
- Indexes and searches repository contents
- Implements background processing for repository indexing
- Caches indexed data for improved performance

### 5. GitHub Tools (github_tools.py)
- Interfaces with GitHub API
- Retrieves file contents and directory structures from repositories

### 6. User Memory Index (memory_index.py)
- Manages user-specific memories
- Implements search functionality for memories
- Handles caching of user memories

### 7. Channel Summarizer (channel_summarizer.py)
- Summarizes Discord channel conversations
- Generates Markdown files with summaries

### 8. Utility Functions (utils.py)
- Provides helper functions like text truncation and Markdown file creation

## Key Features

1. **Multi-API Support**: The bot can use different AI services (Azure, Ollama, OpenRouter, LocalAI) for generating responses.

2. **Repository Integration**: 
   - Indexes and searches GitHub repository contents
   - Analyzes code and generates prompts based on repository content

3. **User Memory Management**:
   - Stores and retrieves user-specific memories
   - Implements search functionality for memories

4. **Channel Summarization**:
   - Summarizes Discord channel conversations
   - Generates and sends Markdown files with summaries

5. **Command System**:
   - `!summarize`: Summarizes channel conversations
   - `!add_memory`: Adds user-specific memories
   - `!clear_memories`: Clears user memories
   - `!search_memories`: Searches user memories
   - `!ask_repo`: Queries the repository
   - `!index_repo`: Initiates repository indexing
   - `!repo_status`: Checks repository indexing status
   - `!generate_prompt`: Generates prompts based on repository content

6. **Background Processing**: 
   - Implements asynchronous processing for repository indexing

7. **Caching**: 
   - Implements caching mechanisms for repository index and user memories

8. **Logging**: 
   - Comprehensive logging for debugging and monitoring

## Configuration
- Uses YAML files for prompt formats and system prompts
- Utilizes environment variables for API keys and other sensitive information

## Extensibility
The modular design allows for easy addition of new features or integration with other services. The bot's architecture separates concerns well, making it maintainable and scalable.

## Usage
Users can interact with the bot through Discord commands or by mentioning the bot in channels. The bot processes these interactions, leverages the appropriate components (e.g., memory index, repo index), and generates responses using the configured AI service.


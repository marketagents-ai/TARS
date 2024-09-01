from pathlib import Path
import discord
from discord.ext import commands
from discord import TextChannel, DMChannel
import logging
import yaml
from github_tools import GitHubRepo
from memory_index import UserMemoryIndex
from repo_index import RepoIndex, start_background_processing_thread
from channel_summarizer import ChannelSummarizer
from cache_manager import CacheManager
from config import GITHUB_TOKEN, REPO_NAME, MAX_TOKENS, CONTEXT_CHUNKS, TEMP_DIR
from utils import log_to_jsonl, process_message, process_file, send_long_message, truncate_middle
from api_client import call_api

import discord
from discord.ext import commands
from channel_summarizer import ChannelSummarizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory of the current file
current_dir = Path(__file__).parent

def setup_bot():
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    bot = commands.Bot(command_prefix='!', intents=intents)

    user_memory_index = UserMemoryIndex('user_memory_index', MAX_TOKENS, CONTEXT_CHUNKS)
    repo_index = RepoIndex('repo_index')
    cache_manager = CacheManager('conversation_history')
    github_repo = GitHubRepo(GITHUB_TOKEN, REPO_NAME)

    with open(f'{current_dir}/prompts/prompt_formats.yaml', 'r') as file:
        prompt_formats = yaml.safe_load(file)
    
    with open(f'{current_dir}/prompts/system_prompts.yaml', 'r') as file:
        system_prompts = yaml.safe_load(file)

    @bot.event
    async def on_ready():
        logging.info(f'{bot.user} has connected to Discord!')

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        # Check if the message is a command
        ctx = await bot.get_context(message)
        if ctx.valid:
            # If it's a valid command, process it only if it's not in a DM
            if not isinstance(message.channel, discord.DMChannel):
                await bot.invoke(ctx)
            return

        if message.attachments:  # Check if there are attachments
            for attachment in message.attachments:
                if attachment.size <= 1000000:  # Check file size limit
                    try:
                        file_content = await attachment.read()
                        # Try to decode as UTF-8, if it fails, use latin-1
                        try:
                            file_content = file_content.decode('utf-8')
                        except UnicodeDecodeError:
                            file_content = file_content.decode('latin-1')
                        await process_file(message, file_content, attachment.filename, user_memory_index, prompt_formats, system_prompts)
                    except Exception as e:
                        await message.channel.send(f"Error processing file: {str(e)}")
                else:
                    await message.channel.send("File is too large. Please upload a file smaller than 1 MB.")
            return

        # Process non-command messages in DMs or when mentioned
        if isinstance(message.channel, discord.DMChannel):
            # In DMs, process all messages that are not commands
            if not message.content.startswith('!'):
                await process_message(message, user_memory_index, prompt_formats, system_prompts, cache_manager, github_repo)
        elif bot.user in message.mentions:
            # In servers, only process messages when mentioned
            await process_message(message, user_memory_index, prompt_formats, system_prompts, cache_manager, github_repo)

    @bot.command(name='add_memory')
    async def add_memory(ctx, *, memory_text):
        """Add a new memory to the AI."""
        user_memory_index.add_memory(str(ctx.author.id), memory_text)
        await ctx.send("Memory added successfully.")
        log_to_jsonl({
            'event': 'add_memory',
            'user_id': str(ctx.author.id),
            'user_name': ctx.author.name,
            'memory_text': memory_text
        })

    @bot.command(name='clear_memories')
    async def clear_memories(ctx):
        """Clear all memories of the invoking user."""
        user_id = str(ctx.author.id)
        user_memory_index.clear_user_memories(user_id)
        await ctx.send("Your memories have been cleared.")
        log_to_jsonl({
            'event': 'clear_user_memories',
            'user_id': user_id,
            'user_name': ctx.author.name
        })

    @bot.command(name='summarize')
    async def summarize(ctx, *, channel_info: str):
        """Summarize a channel and send the summary as a Markdown file via DM.
        Usage from DM: !summarize channel_id|#channel_name [number_of_messages]
        or: !summarize #channel_name [number_of_messages]
        Usage in server: !summarize #channel_name [number_of_messages]
        Default number of messages is 100."""
        
        logging.info(f"Summarize command invoked by {ctx.author} with arguments: {channel_info}")
        
        parts = channel_info.split()
        channel_identifier = parts[0]
        entries = 100  # default value

        if len(parts) > 1:
            try:
                entries = int(parts[1])
            except ValueError:
                await ctx.send("Invalid number of messages. Using default of 100.")
        
        entries = max(10, min(entries, 1000))  # Minimum 10, Maximum 1000

        if '|' in channel_identifier:
            # Detailed format: channel_id|#channel_name
            try:
                channel_id, channel_name = channel_identifier.split('|')
                channel_id = int(channel_id.strip())
                channel_name = channel_name.strip().lstrip('#')
                channel = bot.get_channel(channel_id)
            except ValueError:
                await ctx.send("Invalid format. Please use: !summarize channel_id|#channel_name [number_of_messages]")
                logging.warning(f"Invalid detailed format provided by {ctx.author}: {channel_identifier}")
                return
        else:
            # Simple format: #channel_name
            channel_name = channel_identifier.lstrip('#')
            channel = None
            for guild in bot.guilds:
                channel = discord.utils.get(guild.text_channels, name=channel_name)
                if channel:
                    break
        
        if not channel:
            await ctx.send(f"Channel #{channel_name} not found. Make sure you've provided the correct information.")
            logging.warning(f"Channel not found: {channel_name}")
            return
        
        logging.info(f"Summarizing channel {channel.name} (ID: {channel.id}) in {channel.guild.name}, last {entries} messages")
        
        summarizer = ChannelSummarizer(bot, cache_manager, max_entries=entries)
        success = await summarizer.summarize_and_send(channel.id, ctx.author)
        
        if success:
            await ctx.send(f"Channel #{channel.name} in {channel.guild.name} summarized successfully. Check your DMs for the summary file.")
        else:
            await ctx.send("I couldn't send you a DM. Please check your privacy settings and try again.")
        
        logging.info(f"Summary {'sent to' if success else 'failed to send to'} {ctx.author} for channel {channel.name} (ID: {channel.id})")

    @bot.command(name='ask_repo')
    async def ask_repo(ctx, *, query: str):
        """Ask a question about the repository."""
        results = repo_index.search_repo(query, k=5)
        await ctx.send("Repository search results:")
        for file_path, score in results:
            await ctx.send(f"[Score: {score}] {file_path}")

    @bot.command(name='index_repo')
    async def index_repo(ctx):
        """Index the repository."""
        start_background_processing_thread(github_repo, repo_index)
        await ctx.send("Repository indexing has started in the background.")

    @bot.command(name='repo_status')
    async def repo_status(ctx):
        """Get the status of the repository indexing."""
        if hasattr(repo_index, 'get_repo_status'):
            status = repo_index.get_repo_status()
            await ctx.send(f"Repository indexing status: {status}")
        else:
            await ctx.send("Repository status checking is not implemented.")

    @bot.command(name='generate_prompt')
    async def generate_prompt(ctx):
        """Generate a prompt based on the repository content."""
        if hasattr(repo_index, 'generate_prompt'):
            prompt = repo_index.generate_prompt()
            await ctx.send(f"Generated prompt: {prompt}")
        else:
            await ctx.send("Prompt generation is not implemented.")

    @bot.command(name='search_memories')
    async def search_memories(ctx, *, query: str):
        """Search memories."""
        results = user_memory_index.search(query)
        response = f"Search results for '{query}':\n"
        for memory, score in results:
            result_line = f"[Relevance: {score:.2f}] {truncate_middle(memory, 1000)}\n"
            if len(response) + len(result_line) > 1900:  # Leave some buffer
                await ctx.send(response)
                response = result_line
            else:
                response += result_line
        
        if response:
            await ctx.send(response)
        
        if not results:
            await ctx.send("No results found.")

    return bot

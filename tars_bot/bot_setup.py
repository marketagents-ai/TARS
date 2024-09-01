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
import datetime
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory of the current file
current_dir = Path(__file__).parent
def log_to_jsonl(data):
    with open('bot_log.jsonl', 'a') as f:
        json.dump(data, f)
        f.write('\n')

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging():
    log_level = os.getenv('LOGLEVEL', 'INFO')
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def get_discord_config():
    return {
        'token': os.getenv('DISCORD_TOKEN'),
        'channel_id': int(os.getenv('DISCORD_CHANNEL_ID'))
    }


def setup_bot():
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    bot = commands.Bot(command_prefix='!', intents=intents)

    memory_index = UserMemoryIndex(cache_type='memory_index')
    cache_manager = CacheManager('conversation_history')

    prompt_formats = load_yaml_config('prompt_formats.yaml')
    system_prompts = load_yaml_config('system_prompts.yaml')

    @bot.event
    async def on_ready():
        logging.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
        logging.info('------')
        log_to_jsonl({
            'event': 'bot_ready',
            'timestamp': datetime.now().isoformat(),
            'bot_name': bot.user.name,
            'bot_id': bot.user.id
        })

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return
        if isinstance(message.channel, discord.DMChannel) or bot.user in message.mentions:
            await process_message(message, memory_index, prompt_formats, system_prompts, cache_manager)
        await bot.process_commands(message)

    @bot.command(name='add_memory')
    async def add_memory(ctx, *, memory_text):
        memory_index.add_memory(memory_text)
        await ctx.send("Memory added successfully.")
        log_to_jsonl({
            'event': 'add_memory',
            'timestamp': datetime.now().isoformat(),
            'user_id': str(ctx.author.id),
            'user_name': ctx.author.name,
            'memory_text': memory_text
        })

    @bot.command(name='clear_memories')
    async def clear_memories(ctx):
        memory_index.clear_cache()
        await ctx.send("All memories have been cleared.")
        log_to_jsonl({
            'event': 'clear_memories',
            'timestamp': datetime.now().isoformat(),
            'user_id': str(ctx.author.id),
            'user_name': ctx.author.name
        })

    @bot.command(name='save_memories')
    async def save_memories(ctx):
        memory_index.save_cache()
        await ctx.send("Memories saved successfully.")
        log_to_jsonl({
            'event': 'save_memories',
            'timestamp': datetime.now().isoformat(),
            'user_id': str(ctx.author.id),
            'user_name': ctx.author.name
        })

    @bot.command(name='analyze_file')
    async def analyze_file(ctx):
        if not ctx.message.attachments:
            await ctx.send("Please upload a file to analyze.")
            return
        attachment = ctx.message.attachments[0]
        if attachment.size > 1000000:
            await ctx.send("File is too large. Please upload a file smaller than 1 MB.")
            return
        try:
            file_content = await attachment.read()
            file_content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            await ctx.send("Unable to read the file. Please ensure it's a text file.")
            return
        await process_file(ctx, file_content, attachment.filename, memory_index, prompt_formats, system_prompts)

    @bot.command(name='debug_memories')
    async def debug_memories(ctx):
        memories = "\n".join(memory_index.memories)
        await ctx.send(f"Stored memories:\n{memories[:1900]}...")

    @bot.command(name='debug_search')
    async def debug_search(ctx, *, query):
        results = memory_index.search(query)
        response = f"Search results for '{query}':\n"
        for memory, score in results:
            result_line = f"[Relevance: {score:.2f}] {memory[:100]}...\n"
            if len(response) + len(result_line) > 1900:
                await ctx.send(response)
                response = result_line
            else:
                response += result_line
        if response:
            await ctx.send(response)
        if not results:
            await ctx.send("No results found.")

    async def send_summary(ctx, summary):
        chunks = [summary[i:i+1900] for i in range(0, len(summary), 1900)]
        await ctx.send(f"**Channel Summary**\n\n{chunks[0]}")
        for chunk in chunks[1:]:
            await ctx.send(chunk)

    @bot.command(name='summarize')
    async def summarize(ctx, n: int = 100):
        try:
            cache_manager = CacheManager('channel_summaries', max_history=10)
            summarizer = ChannelSummarizer(bot, cache_manager, max_entries=n)
            summary = await summarizer.summarize_channel(ctx.channel.id)
            await send_summary(ctx, summary)
        except Exception as e:
            error_message = f"An error occurred while summarizing the channel: {str(e)}"
            await ctx.send(error_message)
            logging.error(f"Error in channel summarization: {str(e)}")

    @bot.command(name='debug_context')
    async def debug_context(ctx, *, message):
        user_id = str(ctx.author.id)
        user_name = ctx.author.name
        relevant_memories = memory_index.search(message)
        conversation_history = cache_manager.get_conversation_history(user_id)
        context = f"Current channel: {ctx.channel.name}\n"
        if conversation_history:
            context += f"Previous conversation history with {user_name} (User ID: {user_id}):\n"
            for i, msg in enumerate(reversed(conversation_history[-5:]), 1):
                context += f"Interaction {i}:\n{msg['user_name']}: {msg['user_message']}\nAI: {msg['ai_response']}\n\n"
        else:
            context += f"This is the first interaction with {user_name} (User ID: {user_id}).\n"
        if relevant_memories:
            context += "Relevant memories:\n" + "\n".join(relevant_memories) + "\n\n"
        await ctx.send(f"Debug context for message '{message}':\n{context[:1900]}...")

    return bot

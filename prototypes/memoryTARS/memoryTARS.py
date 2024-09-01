import discord
from discord.ext import commands
import logging
import asyncio
import os
import mimetypes
import tiktoken
import tempfile
import json
from datetime import datetime
from discord import TextChannel
import re
import pickle
from collections import defaultdict
import yaml
import argparse
from dotenv import load_dotenv
from collections import Counter
import string
# Load environment variables
load_dotenv()

# Configuration imports
from config import *
from api_client import initialize_api_client, call_api
from cache_manager import CacheManager

# Set up logging
log_level = os.getenv('LOGLEVEL', 'INFO')
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

# Discord configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
DISCORD_CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID'))

# JSONL logging setup
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

class ChannelSummarizer:
    def __init__(self, bot, cache_manager: CacheManager, max_entries=100):
        self.bot = bot
        self.cache_manager = cache_manager
        self.max_entries = max_entries
        self.prompt_formats = load_yaml_config('prompt_formats.yaml')
        self.system_prompts = load_yaml_config('system_prompts.yaml')

    async def summarize_channel(self, channel_id):
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return "Channel not found."

        main_messages, threads = await self._fetch_channel_history(channel)

        summary = f"Summary of #{channel.name}:\n\n"
        summary += await self._summarize_messages(main_messages, "Main Channel")

        for thread_id, thread_messages in threads.items():
            thread = channel.get_thread(thread_id)
            if thread:
                thread_summary = await self._summarize_messages(thread_messages, f"Thread: {thread.name}")
                summary += f"\n{thread_summary}"

        self.cache_manager.append_to_conversation(str(channel_id), {"summary": summary}, is_ai_chat=True)
        return summary

    async def _fetch_channel_history(self, channel):
        main_messages = []
        threads = defaultdict(list)
        async for message in channel.history(limit=self.max_entries):
            if message.thread:
                threads[message.thread.id].append(message)
            else:
                main_messages.append(message)
        return main_messages, threads

    async def _summarize_messages(self, messages, context):
        user_message_counts, file_types, content_chunks = self._process_messages(messages)
        summary = self._generate_summary(context, user_message_counts, file_types)
        content_summary = await self._process_chunks(content_chunks, context)
        summary += f"\nContent Summary:\n{content_summary}\n"
        return summary

    def _process_messages(self, messages):
        user_message_counts = defaultdict(int)
        file_types = defaultdict(int)
        content_chunks = []
        for message in messages:
            user_message_counts[message.author.name] += 1
            for attachment in message.attachments:
                file_type = attachment.filename.split('.')[-1].lower()
                file_types[file_type] += 1
            content_chunks.append(f"{message.author.name}: {message.content}")
        return user_message_counts, file_types, content_chunks

    def _generate_summary(self, context, user_message_counts, file_types):
        summary = f"{context}\nParticipants:\n"
        for user, count in user_message_counts.items():
            summary += f"- {user}: {count} messages\n"
        if file_types:
            summary += "\nShared Files:\n"
            for file_type, count in file_types.items():
                summary += f"- {file_type}: {count} files\n"
        return summary

    async def _process_chunks(self, chunks, context):
        prompt = self.prompt_formats['summarize_channel'].format(context=context, content="\n".join(chunks))
        system_prompt = self.system_prompts['channel_summarization']
        try:
            return await call_api(prompt, context="", system_prompt=system_prompt)
        except Exception as e:
            return f"Error in generating summary: {str(e)}"

class MemoryIndex:
    def __init__(self, cache_type, max_tokens=1000, context_chunks=4):
        self.cache_manager = CacheManager('memory_index')
        self.cache_dir = self.cache_manager.get_cache_dir(cache_type)
        self.max_tokens = max_tokens
        self.context_chunks = context_chunks
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.inverted_index = defaultdict(list)
        self.memories = []
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        self.load_cache()

    def clean_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        return ' '.join(words)

    def add_memory(self, memory_text):
        memory_id = len(self.memories)
        self.memories.append(memory_text)
        cleaned_text = self.clean_text(memory_text)
        words = cleaned_text.split()
        for word in words:
            self.inverted_index[word].append(memory_id)
        logging.info(f"Added new memory: {memory_text[:100]}...")
        self.save_cache()

    def search(self, query, k=5):
        cleaned_query = self.clean_text(query)
        query_words = cleaned_query.split()
        memory_scores = Counter()
        for word in query_words:
            for memory_id in self.inverted_index.get(word, []):
                memory_scores[memory_id] += 1
        for memory_id, score in memory_scores.items():
            memory_scores[memory_id] = score / len(self.clean_text(self.memories[memory_id]).split())
        sorted_memories = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        results, total_tokens = self._get_top_k_memories(sorted_memories, k)
        logging.info(f"Found {len(results)} relevant memories for query: {query[:100]}...")
        return results

    def _get_top_k_memories(self, sorted_memories, k):
        results = []
        total_tokens = 0
        for memory_id, score in sorted_memories[:k]:
            memory = self.memories[memory_id]
            memory_tokens = self.count_tokens(memory)
            if total_tokens + memory_tokens > self.max_tokens:
                break
            results.append((memory, score))
            total_tokens += memory_tokens
        return results, total_tokens

    def clear_cache(self):
        self.inverted_index.clear()
        self.memories.clear()
        for file_name in ['inverted_index.pkl', 'memories.pkl']:
            cache_file = os.path.join(self.cache_dir, file_name)
            if os.path.exists(cache_file):
                os.remove(cache_file)
        logging.info("Memory index cache cleared")

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def save_cache(self):
        with open(os.path.join(self.cache_dir, 'inverted_index.pkl'), 'wb') as f:
            pickle.dump(self.inverted_index, f)
        with open(os.path.join(self.cache_dir, 'memories.pkl'), 'wb') as f:
            pickle.dump(self.memories, f)
        logging.info("Memory cache saved successfully.")

    def load_cache(self):
        inverted_index_path = os.path.join(self.cache_dir, 'inverted_index.pkl')
        memories_path = os.path.join(self.cache_dir, 'memories.pkl')
        if os.path.exists(inverted_index_path) and os.path.exists(memories_path):
            with open(inverted_index_path, 'rb') as f:
                self.inverted_index = pickle.load(f)
            with open(memories_path, 'rb') as f:
                self.memories = pickle.load(f)
            logging.info("Cache loaded successfully.")
            return True
        return False

def setup_bot():
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    bot = commands.Bot(command_prefix='!', intents=intents)

    memory_index = MemoryIndex(cache_type='memory_index')
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

def truncate_middle(text, max_length=100):
    if len(text) <= max_length:
        return text
    side_length = (max_length - 3) // 2
    end_length = side_length + (max_length - 3) % 2
    return f"{text[:side_length]}...{text[-end_length:]}"

async def process_message(message, memory_index, prompt_formats, system_prompts, cache_manager, is_command=False):
    user_id = str(message.author.id)
    user_name = message.author.name
    content = message.content.split(maxsplit=1)[1] if is_command else message.content.strip()
    if message.guild and message.guild.me:
        content = content.replace(f'<@!{message.guild.me.id}>', '').strip()
    logging.info(f"Received message from {user_name} (ID: {user_id}): {content}")
    try:
        relevant_memories = memory_index.search(content)
        logging.info(f"Found {len(relevant_memories)} relevant memories")
        conversation_history = cache_manager.get_conversation_history(user_id)
        context = build_context(message, user_name, user_id, conversation_history, relevant_memories)
        prompt = prompt_formats['chat_with_memory'].format(context=context, user_name=user_name, user_message=content)
        system_prompt = system_prompts['default_chat']
        logging.info(f"Calling API with prompt length: {len(prompt)} characters")
        response_content = await call_api(prompt, context=context, system_prompt=system_prompt)
        await send_long_message(message.channel, response_content)
        logging.info(f"Sent response to {user_name} (ID: {user_id}): {response_content[:100]}...")
        memory_text = f"User {user_name} in {message.channel.name if hasattr(message.channel, 'name') else 'DM'}: {content}\nAI: {response_content}"
        memory_index.add_memory(memory_text)
        cache_manager.append_to_conversation(user_id, {'user_name': user_name, 'user_message': content, 'ai_response': response_content})
        log_to_jsonl({'event': 'chat_interaction', 'timestamp': datetime.now().isoformat(), 'user_id': user_id, 'user_name': user_name, 'channel': message.channel.name if hasattr(message.channel, 'name') else 'DM', 'user_message': content, 'ai_response': response_content})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await message.channel.send(error_message)
        logging.error(f"Error in message processing for {user_name} (ID: {user_id}): {str(e)}")
        log_to_jsonl({'event': 'chat_error', 'timestamp': datetime.now().isoformat(), 'user_id': user_id, 'user_name': user_name, 'channel': message.channel.name if hasattr(message.channel, 'name') else 'DM', 'error': str(e)})

def build_context(message, user_name, user_id, conversation_history, relevant_memories):
    context = f"Current channel: {message.channel.name if hasattr(message.channel, 'name') else 'Direct Message'}\n"
    if conversation_history:
        context += f"Previous conversation history with {user_name} (User ID: {user_id}):\n"
        for i, msg in enumerate(reversed(conversation_history[-5:]), 1):
            truncated_user_message = truncate_middle(msg['user_message'], max_length=100)
            truncated_ai_response = truncate_middle(msg['ai_response'], max_length=100)
            context += f"Interaction {i}:\n{msg['user_name']}: {truncated_user_message}\nAI: {truncated_ai_response}\n\n"
    else:
        context += f"This is the first interaction with {user_name} (User ID: {user_id}).\n"
    if relevant_memories:
        context += "Relevant memories:\n"
        for memory, score in relevant_memories:
            context += f"[Relevance: {score:.2f}] {memory}\n"
        context += "\n"
    logging.info(f"Prepared context. Length: {len(context)} characters")
    return context

async def process_file(ctx, file_content, filename, memory_index, prompt_formats, system_prompts):
    user_id = str(ctx.author.id)
    user_name = ctx.author.name
    logging.info(f"Processing file '{filename}' from {user_name} (ID: {user_id})")
    try:
        file_prompt = prompt_formats['analyze_file'].format(filename=filename, file_content=file_content[:1000])
        system_prompt = system_prompts['file_analysis']
        response_content = await call_api(file_prompt, context="", system_prompt=system_prompt)
        await send_long_message(ctx, response_content)
        logging.info(f"Sent file analysis response to {user_name} (ID: {user_id}): {response_content[:100]}...")
        memory_index.add_memory(f"Analyzed file '{filename}' for User {user_name}. Analysis: {response_content}")
        log_to_jsonl({'event': 'file_analysis', 'timestamp': datetime.now().isoformat(), 'user_id': user_id, 'user_name': user_name, 'filename': filename, 'ai_response': response_content})
    except Exception as e:
        error_message = f"An error occurred while analyzing the file: {str(e)}"
        await ctx.send(error_message)
        logging.error(f"Error in file analysis for {user_name} (ID: {user_id}): {str(e)}")
        log_to_jsonl({'event': 'file_analysis_error', 'timestamp': datetime.now().isoformat(), 'user_id': user_id, 'user_name': user_name, 'filename': filename, 'error': str(e)})

async def send_long_message(channel, message):
    chunks = [message[i:i+1900] for i in range(0, len(message), 1900)]
    for chunk in chunks:
        await channel.send(chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Discord bot with selected API and model')
    parser.add_argument('--api', choices=['azure', 'ollama', 'openrouter', 'localai'], default='ollama', help='Choose the API to use (default: ollama)')
    parser.add_argument('--model', type=str, help='Specify the model to use. If not provided, defaults will be used based on the API.')
    args = parser.parse_args()

    initialize_api_client(args)
    bot = setup_bot()
    bot.run(DISCORD_TOKEN)

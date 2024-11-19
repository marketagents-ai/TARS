import discord
from discord import TextChannel
from discord.ext import commands

import logging
import asyncio
import os
import mimetypes
import tiktoken
import json
from datetime import datetime

import re
import pickle
from collections import defaultdict, Counter
import yaml
import argparse
from github import Github
from github import GithubException, UnknownObjectException  

import base64
import string
import threading
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Configuration imports
from agent.bot_config import *
from agent.api_client import initialize_api_client, call_api
from agent.cache_manager import CacheManager
from market_agents.orchestrators.discord_orchestrator import MessageProcessor

# image handling
from PIL import Image
import io
import traceback

# import tools
from agent.tools.discordSUMMARISER import ChannelSummarizer
from agent.tools.discordGITHUB import *

# import memory module
from agent.memory import UserMemoryIndex

script_dir = os.path.dirname(os.path.abspath(__file__))

# Set up logging
log_level = os.getenv('LOGLEVEL', 'INFO')
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

# JSONL logging setup
def log_to_jsonl(data):
    with open('bot_log.jsonl', 'a') as f:
        json.dump(data, f)
        f.write('\n')

# Persona intensity handling

def update_temperature(intensity):
    """Updates the global temperature parameter for LLM sampling based on persona intensity.
    
    The persona intensity (0-100) is mapped linearly to temperature (0.0-1.0).
    Higher temperatures result in more random/creative outputs, while lower 
    temperatures make the LLM responses more focused and deterministic.
    
    Args:
        intensity (int): Persona intensity value between 0-100
            0 = Most focused/deterministic (temp = 0.0)
            100 = Most random/creative (temp = 1.0)
    """
    global TEMPERATURE
    TEMPERATURE = intensity / 100.0
    logging.info(f"Updated temperature to {TEMPERATURE}")


def start_background_processing_thread(repo, memory_index, max_depth=None, branch='main'):
    """
    Start a background thread to process and index repository contents.

    Args:
        repo (GitHubRepo): The GitHub repository interface to process
        memory_index (UserMemoryIndex): Index for storing repository content memories
        max_depth (int, optional): Maximum directory depth to process. None means unlimited. Defaults to None.
        branch (str, optional): Git branch to process. Defaults to 'main'.

    The function:
    1. Creates a new thread targeting run_background_processing()
    2. Starts the thread to process repo contents asynchronously 
    3. Logs the start of background processing
    """
    thread = threading.Thread(target=run_background_processing, args=(repo, memory_index, max_depth, branch))
    thread.start()
    logging.info(f"Started background processing of repository contents in a separate thread (Branch: {branch}, Max Depth: {max_depth if max_depth is not None else 'Unlimited'})")

# Update the run_background_processing function to include the branch parameter
def run_background_processing(repo, memory_index, max_depth=None, branch='main'):
    global repo_processing_event
    repo_processing_event.clear()
    try:
        asyncio.run(process_repo_contents(repo, '', memory_index, max_depth, branch))
        memory_index.save_cache()  # Save the cache after indexing
    except Exception as e:
        logging.error(f"Error in background processing for branch '{branch}': {str(e)}")
    finally:
        repo_processing_event.set()

# Message processing

async def process_message(message, memory_index, prompt_formats, system_prompts, cache_manager, github_repo, is_command=False):
    """
    Process an incoming Discord message and generate an appropriate response.

    Args:
        message (discord.Message): The Discord message to process
        memory_index (UserMemoryIndex): Index for storing and retrieving user interaction memories
        prompt_formats (dict): Dictionary of prompt templates for different scenarios
        system_prompts (dict): Dictionary of system prompts for different scenarios 
        cache_manager (CacheManager): Manager for caching conversation history
        github_repo (GitHubRepo): GitHub repository interface
        is_command (bool, optional): Whether message is a command. Defaults to False.

    The function:
    1. Extracts message content and user info
    2. Retrieves relevant conversation history and memories
    3. Builds context from channel history and memories
    4. Generates response using LLM API
    5. Saves interaction to memory and conversation history
    6. Logs the interaction
    7. Sends response back to Discord channel

    Raises:
        Exception: For errors in message processing, API calls, or Discord operations
    """
    user_id = str(message.author.id)
    user_name = message.author.name
    
    # Get conversation history early to determine if this is first interaction
    conversation_history = cache_manager.get_conversation_history(user_id)
    is_first_interaction = not bool(conversation_history)

    if is_command:
        content = message.content.split(maxsplit=1)[1]
    else:
        if message.guild and message.guild.me:
            content = message.content.replace(f'<@!{message.guild.me.id}>', '').strip()
        else:
            content = message.content.strip()
    
    logging.info(f"Received message from {user_name} (ID: {user_id}): {content}")

    try:
        async with message.channel.typing():
            is_dm = isinstance(message.channel, discord.DMChannel)
            
            relevant_memories = memory_index.search(
                content, 
                user_id=(user_id if is_dm else None)
            )
            conversation_history = cache_manager.get_conversation_history(user_id)
            
            context = f"Current channel: {message.channel.name if hasattr(message.channel, 'name') else 'Direct Message'}\n\n"
                    
            if conversation_history:
                context += "**Recalled Conversation:**\n\n"
                for msg in conversation_history[-10:]:  # Show last 10 interactions
                    truncated_user_message = truncate_middle(msg['user_message'], max_tokens=256)
                    truncated_ai_response = truncate_middle(msg['ai_response'], max_tokens=256)
                    context += f"***{msg['user_name']}***: {truncated_user_message}\n"
                    context += f"***Discord LLM Agent***: {truncated_ai_response}\n\n"
            else:
                context += f"This is the first interaction with {user_name} (User ID: {user_id}).\n\n"
            
            context += "**Ongoing Chatroom Conversation:**\n\n"
            messages = []
            async for msg in message.channel.history(limit=10):
                truncated_content = truncate_middle(msg.content, max_tokens=256)
                messages.append(f"***{msg.author.name}***: {truncated_content}")
            
            # Reverse the order of messages and add them to the context
            for msg in reversed(messages):
                context += f"{msg}\n"
            context += "\n"
            
            if relevant_memories:
                context += "**Relevant memories:**\n"
                for memory, score in relevant_memories:
                    truncated_memory = truncate_middle(memory, max_tokens=256)
                    context += f"[Relevance: {score:.2f}] {truncated_memory}\n"
                context += "\n"
            
            # Select appropriate prompt template based on interaction type
            prompt_key = 'introduction' if is_first_interaction else 'chat_with_memory'
            prompt = prompt_formats[prompt_key].format(
                context=context,
                user_name=user_name,
                user_message=content
            )

            # Select appropriate system prompt based on interaction type
            system_prompt_key = 'default_chat'
            system_prompt = system_prompts[system_prompt_key].replace('{persona_intensity}', str(bot.persona_intensity))

            response_content = await call_api(prompt, context=context, system_prompt=system_prompt, temperature=TEMPERATURE)

        await send_long_message(message.channel, response_content)
        logging.info(f"Sent response to {user_name} (ID: {user_id}): {response_content[:1000]}...")

        memory_text = f"User {user_name} in {message.channel.name if hasattr(message.channel, 'name') else 'DM'}: {content}\nYour response: {response_content}"
        memory_index.add_memory(user_id, memory_text)
        
        # Moved outside typing block
        await generate_and_save_thought(
            memory_index=memory_index,
            user_id=user_id,
            user_name=user_name,
            memory_text=memory_text,
            prompt_formats=prompt_formats,
            system_prompts=system_prompts,
            bot=bot
        )

        cache_manager.append_to_conversation(user_id, {
            'user_name': user_name,
            'user_message': content,
            'ai_response': response_content
        })

        log_to_jsonl({
            'event': 'chat_interaction',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'channel': message.channel.name if hasattr(message.channel, 'name') else 'DM',
            'user_message': content,
            'ai_response': response_content
        })

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await message.channel.send(error_message)
        logging.error(f"Error in message processing for {user_name} (ID: {user_id}): {str(e)}")
        log_to_jsonl({
            'event': 'chat_error',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'channel': message.channel.name if hasattr(message.channel, 'name') else 'DM',
            'error': str(e)
        })

async def process_files(message, memory_index, prompt_formats, system_prompts, user_message="", bot=None, temperature=TEMPERATURE):
    """Process multiple files from a Discord message, handling combinations of images and text files."""
    user_id = str(message.author.id)
    user_name = message.author.name
    
    if not message.attachments:
        raise ValueError("No attachments found in message")

    # Track files for combined analysis
    image_files = []
    text_contents = []
    temp_paths = []
    
    # Track detected types for prompt selection
    has_images = False
    has_text = False

    logging.info(f"Processing {len(message.attachments)} files from {user_name} (ID: {user_id}) with message: {user_message}")

    try:
        persona_intensity = str(bot.persona_intensity if bot else DEFAULT_PERSONA_INTENSITY)
        logging.info(f"Using persona intensity: {persona_intensity}")
        
        # Build context once
        context = f"Current channel: {message.channel.name if hasattr(message.channel, 'name') else 'Direct Message'}\n\n"
        context += "**Ongoing Chatroom Conversation:**\n\n"
        messages = []
        async for msg in message.channel.history(limit=10):
            truncated_content = truncate_middle(msg.content, max_tokens=256)
            messages.append(f"***{msg.author.name}***: {truncated_content}")
        
        for msg in reversed(messages):
            context += f"{msg}\n"
        context += "\n"

        async with message.channel.typing():
            # First pass: Check all file types before processing
            for attachment in message.attachments:
                if attachment.size > 1000000:
                    await message.channel.send(f"Skipping {attachment.filename} - file too large (>1MB)")
                    continue
                
                ext = os.path.splitext(attachment.filename.lower())[1]
                is_image = (attachment.content_type and 
                          attachment.content_type.startswith('image/') and 
                          ext in ALLOWED_IMAGE_EXTENSIONS)
                          
                is_text = ext in ALLOWED_EXTENSIONS
                
                if is_image:
                    has_images = True
                if is_text:
                    has_text = True
                    
                if not (is_image or is_text):
                    await message.channel.send(
                        f"Skipping {attachment.filename} - unsupported type. "
                        f"Supported types: {', '.join(ALLOWED_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS)}"
                    )
                    continue

                # Now process the file based on its confirmed type
                if is_image:
                    temp_path = f"temp_{attachment.filename}"
                    try:
                        image_data = await attachment.read()
                        logging.info(f"Downloaded image data for {attachment.filename}: {len(image_data)} bytes")

                        try:
                            img = Image.open(io.BytesIO(image_data))
                            img.verify()
                            logging.info(f"Image verified: {img.format}, {img.size}, {img.mode}")
                        except Exception as img_error:
                            logging.error(f"Image verification failed: {str(img_error)}")
                            logging.error(traceback.format_exc())
                            raise ValueError(f"Invalid image data: {str(img_error)}")

                        with open(temp_path, 'wb') as f:
                            f.write(image_data)
                        logging.info(f"Saved image to temporary path: {temp_path}")
                        
                        if not os.path.exists(temp_path):
                            raise FileNotFoundError(f"Failed to save image: {temp_path} not found")
                        
                        image_files.append(attachment.filename)
                        temp_paths.append(temp_path)
                        
                    except Exception as e:
                        logging.error(f"Error processing image {attachment.filename}: {str(e)}")
                        logging.error(traceback.format_exc())
                        continue

                elif is_text:
                    try:
                        content = await attachment.read()
                        try:
                            text_content = content.decode('utf-8')
                            text_contents.append({
                                'filename': attachment.filename,
                                'content': text_content
                            })
                            logging.info(f"Successfully processed text file: {attachment.filename}")
                        except UnicodeDecodeError as e:
                            logging.error(f"Error decoding text file {attachment.filename}: {str(e)}")
                            await message.channel.send(
                                f"Warning: {attachment.filename} couldn't be decoded. "
                                "Please ensure it's properly encoded as UTF-8."
                            )
                            continue
                    except Exception as e:
                        logging.error(f"Error processing text file {attachment.filename}: {str(e)}")
                        continue

            # Verify we have files to process
            if not (image_files or text_contents):
                raise ValueError("No valid files to analyze")

            # Update flags based on actual processed content
            has_images = bool(image_files)
            has_text = bool(text_contents)

            # Validate required prompts based on file types
            if has_images and has_text:
                if 'analyze_combined' not in prompt_formats or 'combined_analysis' not in system_prompts:
                    raise ValueError("Missing required combined analysis prompts")
            elif has_images:
                if 'analyze_image' not in prompt_formats or 'image_analysis' not in system_prompts:
                    raise ValueError("Missing required image analysis prompts")
            else:  # has_text
                if 'analyze_file' not in prompt_formats or 'file_analysis' not in system_prompts:
                    raise ValueError("Missing required file analysis prompts")

            # Select and format appropriate prompt
            if image_files and text_contents:
                prompt = prompt_formats['analyze_combined'].format(
                    context=context,
                    image_files="\n".join(image_files),
                    text_files="\n".join(f"{t['filename']}: {truncate_middle(t['content'], 1000)}" for t in text_contents),
                    user_message=user_message if user_message else "Please analyze these files."
                )
                system_prompt = system_prompts['combined_analysis'].replace(
                    '{persona_intensity}',
                    persona_intensity
                )
            elif image_files:
                prompt = prompt_formats['analyze_image'].format(
                    context=context,
                    filename=", ".join(image_files),
                    user_message=user_message if user_message else "Please analyze these images."
                )
                system_prompt = system_prompts['image_analysis'].replace(
                    '{persona_intensity}',
                    persona_intensity
                )
            else:
                combined_text = "\n\n".join(f"=== {t['filename']} ===\n{t['content']}" for t in text_contents)
                prompt = prompt_formats['analyze_file'].format(
                    context=context,
                    filename=", ".join(t['filename'] for t in text_contents),
                    file_content=combined_text,
                    user_message=user_message
                )
                system_prompt = system_prompts['file_analysis'].replace(
                    '{persona_intensity}',
                    persona_intensity
                )

            logging.info(f"Using prompt type: {'combined' if has_images and has_text else 'image' if has_images else 'text'}")
            
            try:
                response_content = await call_api(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    image_paths=temp_paths if temp_paths else None,
                    temperature=TEMPERATURE
                )
                logging.info(f"API call successful. Response preview: {response_content[:100]}...")
            except Exception as api_error:
                logging.error(f"Error in API call: {str(api_error)}")
                logging.error(traceback.format_exc())
                raise ValueError(f"API call failed: {str(api_error)}")

            await send_long_message(message.channel, response_content)
            
            # Generate memory text
            files_description = []
            if image_files:
                files_description.append(f"{len(image_files)} images: {', '.join(image_files)}")
            if text_contents:
                files_description.append(f"{len(text_contents)} text files: {', '.join(t['filename'] for t in text_contents)}")
                
            memory_text = f"Analyzed {' and '.join(files_description)} for User {user_name}. User's message: {user_message}. Analysis: {response_content}"
            
            await generate_and_save_thought(
                memory_index=memory_index,
                user_id=user_id,
                user_name=user_name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot
            )

            # Log the analysis
            log_to_jsonl({
                'event': 'file_analysis',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'user_name': user_name,
                'files_processed': {
                    'images': image_files,
                    'text_files': [t['filename'] for t in text_contents]
                },
                'user_message': user_message,
                'ai_response': response_content
            })

    except Exception as e:
        error_message = f"An error occurred while analyzing files: {str(e)}"
        await message.channel.send(error_message)
        logging.error(f"Error in file analysis for {user_name} (ID: {user_id}): {str(e)}")
        logging.error(traceback.format_exc())
        
    finally:
        # Clean up temp files
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logging.info(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logging.error(f"Error removing temporary file {temp_path}: {str(e)}")

async def send_long_message(channel, message, max_length=1800):
    """
    Splits and sends a long message into smaller chunks to Discord, preserving original formatting.
    
    Args:
        channel: The Discord channel to send the message to
        message: The full message text to be sent
        max_length: Maximum length for each chunk (default 1800 to stay under Discord's limit)
    """
    # First split into segments that are either sentences or newlines
    segments = []
    current_text = ""
    
    # Iterate through message character by character to properly handle consecutive newlines
    for char in message:
        if char == '\n':
            if current_text:
                segments.append(('text', current_text))
                current_text = ""
            segments.append(('newline', '\n'))
        else:
            current_text += char
    
    # Add any remaining text
    if current_text:
        segments.append(('text', current_text))
    
    # Now build chunks while preserving formatting
    chunks = []
    current_chunk = ""
    
    for type_, content in segments:
        if type_ == 'newline':
            # Always add newlines if there's room
            if len(current_chunk) + 1 <= max_length:
                current_chunk += content
            else:
                chunks.append(current_chunk)
                current_chunk = content
        else:
            # Split text into sentences
            sentences = sent_tokenize(content)
            for sentence in sentences:
                sentence = sentence.strip()
                # Add proper spacing based on context
                if current_chunk and not current_chunk.endswith('\n'):
                    sentence = ' ' + sentence
                
                # Check if this sentence fits
                if len(current_chunk) + len(sentence) <= max_length:
                    current_chunk += sentence
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence
    
    # Add the final chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # Send each chunk
    for chunk in chunks:
        await channel.send(chunk)

def truncate_middle(text, max_tokens=256):
    """
    Truncates text to a maximum number of tokens while preserving content from both ends.
    
    Args:
        text (str): The input text to truncate
        max_tokens (int, optional): Maximum number of tokens to keep. Defaults to 256.
        
    Returns:
        str: The truncated text with ellipsis (...) in the middle if truncation was needed,
             or the original text if it was already within the token limit.
             
    The function preserves roughly equal amounts of tokens from the start and end of the text,
    placing an ellipsis in the middle to indicate truncation. It uses the cl100k_base tokenizer
    from tiktoken for token counting and manipulation.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    keep_tokens = max_tokens - 3  # Account for the ellipsis (...) in the middle
    side_tokens = keep_tokens // 2
    end_tokens = side_tokens + (keep_tokens % 2)  # Add the remainder to the end if odd
    
    truncated_tokens = tokens[:side_tokens] + [tokenizer.encode('...')[0]] + tokens[-end_tokens:]
    return tokenizer.decode(truncated_tokens)

# There is an issue where the bot will indicate typing because processing files, images and other calls haz thought gen within the api loop. This needs changing to match the prcoess message style.

async def generate_and_save_thought(memory_index, user_id, user_name, memory_text, prompt_formats, system_prompts, bot):
    """
    Generates a thought about a memory and saves both to the memory index.
    
    Args:
        memory_index (UserMemoryIndex): The memory index to store memories and thoughts
        user_id (str): ID of the user the memory/thought is about
        memory_text (str): The original memory text to generate a thought about
        prompt_formats (dict): Dictionary containing prompt templates
        system_prompts (dict): Dictionary containing system prompt templates
        bot (commands.Bot): The bot instance containing persona settings
        
    The function:
    1. Generates a thought about the memory using the AI API
    2. Saves both the original memory and the thought to the memory index
    3. Uses the bot's persona intensity for thought generation
    """

    # Get current timestamp and format it as hh:mm [dd/mm/yy]
    timestamp = datetime.now().strftime("%H:%M [%d/%m/%y]")

    # Generate thought
    thought_prompt = prompt_formats['generate_thought'].format(
        user_name=user_name,
        memory_text=memory_text,
        timestamp=timestamp
    )
    
    thought_system_prompt = system_prompts['thought_generation'].replace('{persona_intensity}', str(bot.persona_intensity))
    
    thought_response = await call_api(thought_prompt, context="", system_prompt=thought_system_prompt, temperature=TEMPERATURE)
    
    # Save both the original memory and the thought
    memory_index.add_memory(user_id, memory_text)
    memory_index.add_memory(user_id, f"Priors on interactions with @{user_name}: {thought_response} (Timestamp: {timestamp})")


# Bot setup

def setup_bot():
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    bot = commands.Bot(command_prefix='!', intents=intents, status=discord.Status.online)  

    # Initialize with specific cache types
    user_memory_index = UserMemoryIndex('user_memory_index')
    repo_index = RepoIndex('repo_index')
    cache_manager = CacheManager('conversation_history')  # Keep existing name for compatibility
    github_repo = GitHubRepo(GITHUB_TOKEN, REPO_NAME)

    with open(os.path.join(script_dir, 'prompts', 'prompt_formats.yaml'), 'r') as file:
        prompt_formats = yaml.safe_load(file)

    with open(os.path.join(script_dir, 'prompts', 'system_prompts.yaml'), 'r') as file:
        system_prompts = yaml.safe_load(file)

    # Add this variable to store the current persona intensity
    bot.persona_intensity = DEFAULT_PERSONA_INTENSITY

    # Initialize cache managers for different purposes
    conversation_cache = CacheManager('conversation_history')
    file_cache = CacheManager('file_cache')
    prompt_cache = CacheManager('prompt_cache')

    # Store cache managers on the bot instance for access in commands
    bot.cache_managers = {
        'conversation': conversation_cache,
        'file': file_cache,
        'prompt': prompt_cache
    }

    # Store other necessary objects on the bot instance
    bot.user_memory_index = user_memory_index
    bot.repo_index = repo_index
    bot.cache_manager = cache_manager
    bot.github_repo = github_repo
    bot.prompt_formats = prompt_formats
    bot.system_prompts = system_prompts

    # Initialize the MessageProcessor with the bot instance
    bot.message_processor = MessageProcessor(bot)

    # Initialize message cache for accumulating messages
    bot.message_cache = {}

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

        # Now that the bot is ready, set bot_id in MessageProcessor
        bot.message_processor.bot_id = str(bot.user.id)

        # Set up the agent asynchronously
        await bot.message_processor.setup_agent()

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        # Check if the message is a command
        ctx = await bot.get_context(message)
        if ctx.valid:
            await bot.invoke(ctx)
            return

        # Handle direct messages or mentions
        if isinstance(message.channel, discord.DMChannel) or bot.user in message.mentions:
            if message.attachments:
                attachment = message.attachments[0]
                if attachment.size <= 1000000:  # 1MB limit
                    try:
                        await process_files(
                            message=message,
                            memory_index=bot.user_memory_index,
                            prompt_formats=bot.prompt_formats,
                            system_prompts=bot.system_prompts,
                            user_message=message.content,
                            bot=bot
                        )
                    except Exception as e:
                        await message.channel.send(f"Error processing file: {str(e)}")
                else:
                    await message.channel.send("File is too large. Please upload a file smaller than 1 MB.")
            else:
                await process_message(
                    message,
                    bot.user_memory_index,
                    bot.prompt_formats,
                    bot.system_prompts,
                    bot.cache_manager,
                    bot.github_repo
                )
        else:
            # Accumulate messages from other channels
            channel_id = message.channel.id
            if not hasattr(bot, 'message_cache'):
                bot.message_cache = {}

            if channel_id not in bot.message_cache:
                bot.message_cache[channel_id] = []

            bot.message_cache[channel_id].append({
                "content": message.content,
                "author_id": str(message.author.id),
                "author_name": message.author.name,
                "timestamp": message.created_at.isoformat()
            })

            # Keep the cache size limited per channel
            if len(bot.message_cache[channel_id]) > 20:
                bot.message_cache[channel_id].pop(0)

            # If the number of messages accumulated reaches a threshold, process them
            if len(bot.message_cache[channel_id]) >= 10:
                # Process the messages through the agent
                channel_info = {
                    "id": str(message.channel.id),
                    "name": message.channel.name
                }

                # Process the messages using the MessageProcessor
                results = await bot.message_processor.process_messages(channel_info, bot.message_cache[channel_id])

                if results:
                    # Process the agent's action
                    action_result = results.get('action')
                    if action_result and action_result.get('content'):
                        # Send the agent's message to the channel
                        await message.channel.send(action_result['content'])
                else:
                    logging.error("Message processing failed")

                # Clear the message cache for this channel
                bot.message_cache[channel_id] = []
                
    @bot.command(name='persona')
    async def set_persona_intensity(ctx, intensity: int = None):
        """Set or get the AI's persona intensity (0-100). The intensity can be steered through in context prompts and it also adjusts the temperature of the API calls."""
        if intensity is None:
            await ctx.send(f"Current persona intensity is {bot.persona_intensity}%.")
            logging.info(f"Persona intensity queried by user {ctx.author.name} (ID: {ctx.author.id})")
        elif 0 <= intensity <= 100:
            bot.persona_intensity = intensity
            update_temperature(intensity)  # Update the temperature
            await ctx.send(f"Persona intensity set to {intensity}%.")
            logging.info(f"Persona intensity set to {intensity}% by user {ctx.author.name} (ID: {ctx.author.id})")
        else:
            await ctx.send("Please provide a valid intensity between 0 and 100.")


    @bot.command(name='add_memory')
    async def add_memory(ctx, *, memory_text):
        """Add a new memory to the AI."""
        user_memory_index.add_memory(str(ctx.author.id), memory_text)
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
        """Clear all memories of the invoking user."""
        user_id = str(ctx.author.id)
        user_memory_index.clear_user_memories(user_id)
        await ctx.send("Your memories have been cleared.")
        log_to_jsonl({
            'event': 'clear_user_memories',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': ctx.author.name
        })

    @bot.command(name='analyze_file')
    async def analyze_file(ctx):
        """Analyze an uploaded file."""
        if not ctx.message.attachments:
            await ctx.send("Please upload a file to analyze.")
            return

        attachment = ctx.message.attachments[0]
        
        if attachment.size > 1000000:  # 1 MB limit
            await ctx.send("File is too large. Please upload a file smaller than 1 MB.")
            return

        try:
            file_content = await attachment.read()
            file_content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            await ctx.send("Unable to read the file. Please ensure it's a text file.")
            return

        await process_files(ctx, file_content, attachment.filename, user_memory_index, prompt_formats, system_prompts)

    @bot.command(name='summarize')
    async def summarize(ctx, *, args=None):
        """Summarize the last n messages in a specified channel and send the summary to DM."""
        try:
            n = 100  # Default value
            channel = None

            if args:
                parts = args.split()
                if len(parts) >= 1:
                    # Check if the first part is a channel mention or ID
                    if parts[0].startswith('<#') and parts[0].endswith('>'):
                        channel_id = int(parts[0][2:-1])
                    elif parts[0].isdigit():
                        channel_id = int(parts[0])
                    else:
                        await ctx.send("Please provide a valid channel ID or mention.")
                        return
                    
                    channel = bot.get_channel(channel_id)
                    if channel is None:
                        await ctx.send(f"Invalid channel. Channel ID: {channel_id}")
                        return
                    parts = parts[1:]  # Remove the channel mention/ID from parts

                    # Check if there's a number provided
                    if parts:
                        try:
                            n = int(parts[0])
                        except ValueError:
                            await ctx.send("Invalid input. Please provide a number for the amount of messages to summarize.")
                            return
            else:
                await ctx.send("Please specify a channel ID or mention to summarize.")
                return

            # Log the attempt
            logging.info(f"Attempting to summarize {n} messages from channel {channel.name} (ID: {channel.id})")

            # Check permissions
            member = channel.guild.get_member(ctx.author.id)
            if member is None or not channel.permissions_for(member).read_messages:
                await ctx.send(f"You don't have permission to read messages in the specified channel.")
                return
            
            if not channel.permissions_for(channel.guild.me).read_message_history:
                await ctx.send(f"I don't have permission to read message history in the specified channel.")
                return

            async with ctx.channel.typing():
                summarizer = ChannelSummarizer(bot, cache_manager, prompt_formats, system_prompts, max_entries=n)
                summary = await summarizer.summarize_channel(channel.id)
                
            # Send the summary as a DM to the user
            try:
                await send_long_message(ctx.author, f"**Channel Summary for #{channel.name} (Last {n} messages)**\n\n{summary}")
                
                # Confirm that the summary was sent
                if isinstance(ctx.channel, discord.DMChannel):
                    await ctx.send(f"I've sent you the summary of #{channel.name}.")
                else:
                    await ctx.send(f"{ctx.author.mention}, I've sent you a DM with the summary of #{channel.name}.")
            except discord.Forbidden:
                await ctx.send("I couldn't send you a DM. Please check your privacy settings and try again.")

            # Save and generate thought
            memory_text = f"Summarized {n} messages from #{channel.name}. Summary: {summary}"
            await generate_and_save_thought(
                memory_index=user_memory_index,
                user_id=str(ctx.author.id),
                user_name=ctx.author.name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot
            )

        except discord.Forbidden as e:
            await ctx.send(f"I don't have permission to perform this action. Error: {str(e)}")
        except Exception as e:
            error_message = f"An error occurred while summarizing the channel: {str(e)}"
            await ctx.send(error_message)
            logging.error(f"Error in channel summarization: {str(e)}")

    @bot.command(name='index_repo')
    async def index_repo(ctx, option: str = None, branch: str = 'main'):
        """Index the GitHub repository contents, list indexed files, or check indexing status."""
        global repo_processing_event

        if option == 'list':
            if repo_processing_event.is_set():
                indexed_files = set()
                for file_paths in repo_index.repo_index.values():
                    indexed_files.update(file_paths)
                
                if indexed_files:
                    file_list = f"# Indexed Repository Files (Branch: {branch})\n\n"
                    for file in sorted(indexed_files):
                        file_list += f"- `{file}`\n"
                    
                    temp_file = 'indexed_files.md'
                    with open(temp_file, 'w') as f:
                        f.write(file_list)
                    
                    await ctx.send(f"Here's the list of indexed files from the '{branch}' branch:", file=discord.File(temp_file))
                    
                    os.remove(temp_file)
                else:
                    await ctx.send(f"No files have been indexed yet on the '{branch}' branch.")
            else:
                await ctx.send(f"Repository indexing has not been completed for the '{branch}' branch. Please run `!index_repo` first.")
        elif option == 'status':
            if repo_processing_event.is_set():
                await ctx.send("Repository indexing is complete.")
            else:
                await ctx.send("Repository indexing is still in progress.")
        else:
            if not repo_processing_event.is_set():
                try:
                    await ctx.send(f"Starting to index the repository on the '{branch}' branch... This may take a while.")
                    
                    # Clear existing cache before starting new indexing
                    repo_index.clear_cache()
                    
                    start_background_processing_thread(github_repo.repo, repo_index, max_depth=None, branch=branch)
                    await ctx.send(f"Repository indexing has started in the background for the '{branch}' branch. You will be notified when it's complete.")
                except Exception as e:
                    error_message = f"An error occurred while starting the repository indexing on the '{branch}' branch: {str(e)}"
                    await ctx.send(error_message)
                    logging.error(error_message)
            else:
                await ctx.send(f"Repository indexing is already in progress or completed for the '{branch}' branch. Use `!index_repo list` to see indexed files or `!index_repo status` to check the current status.")

    @bot.command(name='generate_prompt')
    async def generate_prompt_command(ctx, *, input_text):
        parts = input_text.split(maxsplit=1)
        if len(parts) < 2:
            await ctx.send("Error: Please provide both a file path and a task description.")
            return

        file_path = parts[0]
        user_task_description = parts[1]

        logging.info(f"Received generate_prompt command: {file_path}, {user_task_description}")
        
        if not repo_processing_event.is_set():
            await ctx.send("Repository indexing is not complete. Please run !index_repo first.")
            return

        try:
            # Normalize the file path
            file_path = file_path.strip().replace('\\', '/')
            if file_path.startswith('/'):
                file_path = file_path[1:]  # Remove leading slash if present
            
            logging.info(f"Normalized file path: {file_path}")

            # Check if the file is in the indexed repository
            indexed_files = set()
            for file_set in repo_index.repo_index.values():
                indexed_files.update(file_set)
            
            if file_path not in indexed_files:
                await ctx.send(f"Error: The file '{file_path}' is not in the indexed repository.")
                return

            # Fetch the file content using GitHubRepo class
            async with ctx.channel.typing():
                repo_code = github_repo.get_file_content(file_path)
                
                if repo_code.startswith("Error fetching file:"):
                    await ctx.send(f"Error: {repo_code}")
                    return
                elif repo_code == "File is too large to fetch content directly.":
                    await ctx.send(repo_code)
                    return

                logging.info(f"Successfully fetched file: {file_path}")

            # Determine the code type based on file extension
            _, file_extension = os.path.splitext(file_path)
            code_type = mimetypes.types_map.get(file_extension, "").split('/')[-1] or "plaintext"
            
            # Check if the necessary keys exist in prompt_formats and system_prompts
            if 'generate_prompt' not in prompt_formats:
                await ctx.send("Error: 'generate_prompt' template is missing from prompt_formats.")
                return
            if 'generate_prompt' not in system_prompts:
                await ctx.send("Error: 'generate_prompt' is missing from system_prompts.")
                return
            
            # Add channel context
            context = f"Current channel: {ctx.channel.name if hasattr(ctx.channel, 'name') else 'Direct Message'}\n\n"
            context += "**Ongoing Chatroom Conversation:**\n\n"
            messages = []
            async for msg in ctx.channel.history(limit=10):
                truncated_content = truncate_middle(msg.content, max_tokens=256)
                messages.append(f"***{msg.author.name}***: {truncated_content}")
            
            # Reverse the order of messages and add them to the context
            for msg in reversed(messages):
                context += f"{msg}\n"
            context += "\n"

            prompt = prompt_formats['generate_prompt'].format(
                file_path=file_path,
                code_type=code_type,
                repo_code=repo_code,
                user_task_description=user_task_description,
                context=context
            )
            
            # Replace persona_intensity in the system prompt
            system_prompt = system_prompts['generate_prompt'].replace('{persona_intensity}', str(bot.persona_intensity))
            
            async with ctx.channel.typing():
                response_content = await call_api(prompt, system_prompt=system_prompt)
             
            # Create temporary markdown file using cache manager
            cache_manager = CacheManager('prompt_cache')
            safe_filename = re.sub(r'[^\w\-_\. ]', '_', user_task_description)
            safe_filename = safe_filename[:50]  # Limit filename length
            
            content = f"# Generated Prompt for: {user_task_description}\n\n"
            content += f"File: `{file_path}`\n\n"
            content += response_content
            
            temp_path, file_id = cache_manager.create_temp_file(
                user_id=str(ctx.author.id),
                prefix='prompt',
                suffix='.md',
                content=content
            )
            
            # Send the Markdown file to the chat
            await ctx.send(
                f"Generated response for: {user_task_description}", 
                file=discord.File(temp_path)
            )
            
            # Clean up the temporary file
            cache_manager.remove_temp_file(str(ctx.author.id), file_id)
            
            logging.info(f"Sent AI response as Markdown file: {temp_path}")

            # Generate and save thought
            memory_text = f"Generated prompt for file '{file_path}' with task description '{user_task_description}'. Response: {response_content}"
            await generate_and_save_thought(
                memory_index=user_memory_index,
                user_id=str(ctx.author.id),
                user_name=ctx.author.name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot
            )

        except Exception as e:
            error_message = f"Error generating prompt and querying AI: {str(e)}"
            await ctx.send(error_message)
            logging.error(error_message)

    @bot.command(name='ask_repo')
    async def ask_repo(ctx, *, question):
        """Chat about the GitHub repository contents."""
        try:
            if not repo_processing_event.is_set():
                await ctx.send("Repository indexing is not complete. Please wait or run !index_repo first.")
                return

            relevant_files = repo_index.search_repo(question)
            if not relevant_files:
                await ctx.send("No relevant files found in the repository for this question.")
                return

            context = "Relevant files in the repository:\n"
            file_links = []  # List to store file links
            for file_path, score in relevant_files:
                context += f"- {file_path} (Relevance: {score:.2f})\n"
                file_content = github_repo.get_file_content(file_path)
                context += f"Content preview: {truncate_middle(file_content, 1000)}\n\n"
                file_links.append(f"{file_path}")

            # Add channel context
            context += f"\nCurrent channel: {ctx.channel.name if hasattr(ctx.channel, 'name') else 'Direct Message'}\n\n"
            context += "**Ongoing Chatroom Conversation:**\n\n"
            messages = []
            async for msg in ctx.channel.history(limit=10):
                truncated_content = truncate_middle(msg.content, max_tokens=256)
                messages.append(f"***{msg.author.name}***: {truncated_content}")
            
            # Reverse the order of messages and add them to the context
            for msg in reversed(messages):
                context += f"{msg}\n"
            context += "\n"

            prompt = prompt_formats['ask_repo'].format(
                context=context,
                question=question
            )

            system_prompt = system_prompts['ask_repo']

            # Replace persona_intensity in the system prompt
            system_prompt = system_prompts['ask_repo'].replace('{persona_intensity}', str(bot.persona_intensity))
            
            async with ctx.channel.typing():
                response = await call_api(prompt, context=context, system_prompt=system_prompt)

            # Append file links to the response wrapped in a Markdown code block
            response += "\n\nReferenced Files:\n```md\n" + "\n".join(file_links) + "\n```"
            
            await send_long_message(ctx, response)
            logging.info(f"Sent repo chat response for question: {question[:100]}...")

            # Generate and save thought
            memory_text = f"Asked repo question '{question}'. Response: {response}"
            await generate_and_save_thought(
                memory_index=user_memory_index,
                user_id=str(ctx.author.id),
                user_name=ctx.author.name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot
            )

        except Exception as e:
            error_message = f"An error occurred while processing the repo chat: {str(e)}"
            await ctx.send(error_message)
            logging.error(f"Error in repo chat: {str(e)}")

    @bot.command(name='search_memories')
    async def search_memories(ctx, *, query):
        """Test the memory search function."""
        is_dm = isinstance(ctx.channel, discord.DMChannel)
        user_id = str(ctx.author.id) if is_dm else None
        
        results = user_memory_index.search(query, user_id=user_id)
        
        if not results:
            await ctx.send("No results found.")
            return
            
        current_chunk = f"Search results for '{query}':\n"
        
        for memory, score in results:
            # First truncate the memory content
            truncated_memory = truncate_middle(memory, 800)
            result_line = f"[Relevance: {score:.2f}] {truncated_memory}\n"
            
            # Ensure single result doesn't exceed limit
            if len(result_line) > 1900:
                result_line = result_line[:1896] + "...\n"
            
            # Check if adding this line would exceed Discord's limit
            if len(current_chunk) + len(result_line) > 1900:
                await ctx.send(current_chunk)
                current_chunk = result_line
            else:
                current_chunk += result_line
        
        # Send any remaining content
        if current_chunk:
            await ctx.send(current_chunk)

    return bot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Discord bot with selected API and model')
    parser.add_argument('--api', choices=['ollama', 'openai', 'anthropic', 'vllm'], default='ollama', help='Choose the API to use (default: ollama)')
    parser.add_argument('--model', type=str, help='Specify the model to use. If not provided, defaults will be used based on the API.')
    args = parser.parse_args()

    initialize_api_client(args)

    bot = setup_bot()
    bot.run(TOKEN)
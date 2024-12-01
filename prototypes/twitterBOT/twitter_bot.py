# twitter_bot.py
import tweepy
import logging
import asyncio
import os
import tiktoken
import json
from datetime import datetime
import yaml
from functools import wraps
import traceback
import re
from PIL import Image
import io
import aiohttp
import argparse

# Configuration imports
from agent.bot_config import *
from agent.api_client import initialize_api_client, call_api
from agent.cache_manager import CacheManager
from agent.memory import UserMemoryIndex

script_dir = os.path.dirname(os.path.abspath(__file__))

# Set up logging
log_level = os.getenv('LOGLEVEL', 'INFO')
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def log_to_jsonl(data):
    """Log events to JSONL file"""
    with open('twitter_bot_log.jsonl', 'a') as f:
        json.dump(data, f)
        f.write('\n')

def handle_rate_limit(func):
    """Decorator to handle Twitter rate limits"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except tweepy.TooManyRequests as e:
            wait_seconds = int(e.response.headers.get('x-rate-limit-reset', 60))
            logging.warning(f"Rate limit hit, waiting {wait_seconds} seconds")
            await asyncio.sleep(wait_seconds)
            return await func(*args, **kwargs)
    return wrapper

class TwitterMediaHandler:
    """Handle Twitter media attachments"""
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.temp_dir = os.path.join(os.getcwd(), 'temp_twitter_media')
        os.makedirs(self.temp_dir, exist_ok=True)

    async def process_media(self, media_items):
        """Process media from a tweet"""
        image_files = []
        text_files = []
        temp_paths = []

        try:
            for media in media_items:
                if media.type == 'photo':
                    temp_path = await self.save_image(media)
                    if temp_path:
                        image_files.append(media.url)
                        temp_paths.append(temp_path)
                elif media.type == 'text':
                    text_content = await self.get_text_content(media)
                    if text_content:
                        text_files.append({
                            'filename': f'tweet_text_{media.id}.txt',
                            'content': text_content
                        })

            return image_files, text_files, temp_paths

        except Exception as e:
            logging.error(f"Error processing media: {str(e)}")
            self.cleanup_temp_files(temp_paths)
            raise

    async def save_image(self, media):
        """Save image to temporary file"""
        try:
            temp_path = os.path.join(self.temp_dir, f"temp_img_{media.id}.jpg")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(media.url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        
                        # Verify image
                        img = Image.open(io.BytesIO(image_data))
                        img.verify()
                        
                        # Save image
                        with open(temp_path, 'wb') as f:
                            f.write(image_data)
                            
                        logging.info(f"Saved image to {temp_path}")
                        return temp_path
                        
        except Exception as e:
            logging.error(f"Error saving image: {str(e)}")
            return None

    def cleanup_temp_files(self, temp_paths):
        """Clean up temporary files"""
        for path in temp_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logging.info(f"Removed temporary file: {path}")
            except Exception as e:
                logging.error(f"Error removing temporary file {path}: {str(e)}")

class TwitterMessageHandler:
    """Handle tweet processing and responses"""
    def __init__(self, bot):
        self.bot = bot
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def process_tweet(self, tweet):
        """Process incoming tweet"""
        user_id = str(tweet.author_id)
        user = await self.bot.get_user_info(user_id)
        user_name = user.username

        # Get conversation history
        conversation_history = self.bot.cache_manager.get_conversation_history(user_id)
        is_first_interaction = not bool(conversation_history)

        # Clean tweet content
        content = self.clean_tweet_content(tweet.text)

        logging.info(f"Processing tweet from {user_name} (ID: {user_id}): {content}")

        try:
            # Handle media if present
            media_items = getattr(tweet, 'attachments', {}).get('media', [])
            if media_items:
                await self.process_tweet_with_media(tweet, media_items)
                return

            # Process regular tweet
            relevant_memories = self.bot.memory_index.search(
                content,
                user_id=user_id
            )

            # Build context
            context = self.build_tweet_context(
                user_name=user_name,
                conversation_history=conversation_history,
                relevant_memories=relevant_memories
            )

            # Generate response
            prompt_key = 'introduction' if is_first_interaction else 'chat_with_memory'
            response = await self.generate_response(
                content=content,
                context=context,
                prompt_key=prompt_key,
                user_name=user_name
            )

            # Send response
            await self.bot.send_thread_response(tweet.id, response)

            # Update memory and cache
            await self.update_memory_and_cache(
                user_id=user_id,
                user_name=user_name,
                content=content,
                response=response
            )

        except Exception as e:
            error_message = f"Error processing tweet: {str(e)}"
            logging.error(f"{error_message}\n{traceback.format_exc()}")
            await self.bot.send_error_dm(user_id, error_message)

    async def process_tweet_with_media(self, tweet, media_items):
        """Process tweet containing media"""
        user_id = str(tweet.author_id)
        user_name = tweet.author.username
        content = self.clean_tweet_content(tweet.text)

        try:
            # Process media
            image_files, text_files, temp_paths = await self.bot.media_handler.process_media(media_items)

            # Build context
            context = self.build_media_context(
                user_name=user_name,
                content=content,
                image_files=image_files,
                text_files=text_files
            )

            # Generate response
            response = await self.generate_media_response(
                context=context,
                has_images=bool(image_files),
                has_text=bool(text_files)
            )

            # Send response
            await self.bot.send_thread_response(tweet.id, response)

            # Update memory
            memory_text = self.build_media_memory(
                user_name=user_name,
                content=content,
                image_files=image_files,
                text_files=text_files,
                response=response
            )
            
            self.bot.memory_index.add_memory(user_id, memory_text)

        finally:
            # Cleanup
            self.bot.media_handler.cleanup_temp_files(temp_paths)

    def clean_tweet_content(self, text):
        """Clean tweet text"""
        # Remove mentions
        text = re.sub(r'@\w+\s*', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+\s*', '', text)
        # Clean up whitespace
        text = ' '.join(text.split())
        return text.strip()

    def build_tweet_context(self, user_name, conversation_history, relevant_memories):
        """Build context for tweet response"""
        context = f"Platform: Twitter\n"
        context += f"User: @{user_name}\n\n"

        if conversation_history:
            context += "Previous interactions:\n"
            for msg in conversation_history[-5:]:  # Limited due to Twitter context
                truncated_msg = self.truncate_middle(msg['user_message'], 128)
                truncated_response = self.truncate_middle(msg['bot_response'], 128)
                context += f"User: {truncated_msg}\n"
                context += f"Bot: {truncated_response}\n\n"
        else:
            context += f"First interaction with @{user_name}\n\n"

        if relevant_memories:
            context += "Relevant memories:\n"
            for memory, score in relevant_memories:
                truncated_memory = self.truncate_middle(memory, 128)
                context += f"[{score:.2f}] {truncated_memory}\n"

        return context

    def build_media_context(self, user_name, content, image_files, text_files):
        """Build context for media response"""
        context = f"Platform: Twitter\n"
        context += f"User: @{user_name}\n"
        context += f"Message: {content}\n\n"

        if image_files:
            context += f"Attached images ({len(image_files)}):\n"
            for img in image_files:
                context += f"- {img}\n"

        if text_files:
            context += f"\nAttached text files ({len(text_files)}):\n"
            for txt in text_files:
                truncated_content = self.truncate_middle(txt['content'], 256)
                context += f"=== {txt['filename']} ===\n{truncated_content}\n\n"

        return context

    def truncate_middle(self, text, max_tokens=256):
        """Truncate text while preserving start and end content"""
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            return text

        keep_tokens = max_tokens - 3  # Account for ellipsis
        side_tokens = keep_tokens // 2
        end_tokens = side_tokens + (keep_tokens % 2)

        truncated_tokens = tokens[:side_tokens] + [self.tokenizer.encode('...')[0]] + tokens[-end_tokens:]
        return self.tokenizer.decode(truncated_tokens)

    def split_response(self, response, max_length=280):
        """Split response into tweet-sized chunks"""
        if len(response) <= max_length:
            return [response]

        chunks = []
        current_chunk = ""
        sentences = response.split('. ')

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence.endswith('.'):
                sentence += '.'

            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += ' ' + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

class TwitterBot:
    """Main Twitter bot class"""
    def __init__(self):
        # Initialize core components
        self.memory_index = UserMemoryIndex('twitter_memory_index')
        self.cache_manager = CacheManager('twitter_conversation_history')
        
        # Load prompts
        with open(os.path.join(script_dir, 'prompts', 'prompt_formats.yaml'), 'r') as file:
            self.prompt_formats = yaml.safe_load(file)
        with open(os.path.join(script_dir, 'prompts', 'system_prompts.yaml'), 'r') as file:
            self.system_prompts = yaml.safe_load(file)

        # Initialize Twitter client
        self.client = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_SECRET,
            wait_on_rate_limit=True
        )

        # Initialize handlers
        self.media_handler = TwitterMediaHandler(self.cache_manager)
        self.message_handler = TwitterMessageHandler(self)

        # Set default persona intensity
        self.persona_intensity = DEFAULT_PERSONA_INTENSITY
        
        # Track last checked tweet ID
        self.last_checked_id = None

    async def start(self):
        """Start the Twitter bot"""
        try:
            logging.info("Twitter bot started successfully")
            while True:
                await self.check_mentions()
                await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logging.error(f"Failed to start Twitter bot: {str(e)}")
            raise

    @handle_rate_limit
    async def check_mentions(self):
        """Check for new mentions"""
        try:
            # Get mentions
            mentions = self.client.get_users_mentions(
                id=self.client.get_me().data.id,
                since_id=self.last_checked_id,
                tweet_fields=['referenced_tweets', 'author_id', 'attachments'],
                expansions=['referenced_tweets.id', 'attachments.media_keys'],
                media_fields=['url', 'type']
            )

            if not mentions.data:
                return

            # Update last checked ID
            self.last_checked_id = mentions.data[0].id

            # Process mentions
            for tweet in mentions.data:
                await self.message_handler.process_tweet(tweet)

        except Exception as e:
            logging.error(f"Error checking mentions: {str(e)}")

    @handle_rate_limit
    async def send_thread_response(self, reply_to_id, response):
        """Send response as a thread if needed"""
        try:
            chunks = self.message_handler.split_response(response)
            previous_id = reply_to_id

            for chunk in chunks:
                tweet = await self.client.create_tweet(
                    text=chunk,
                    in_reply_to_tweet_id=previous_id
                )
                previous_id = tweet.data['id']
                await asyncio.sleep(1)  # Rate limit compliance

        except Exception as e:
            logging.error(f"Error sending thread response: {str(e)}")
            raise

    @handle_rate_limit
    async def send_error_dm(self, user_id, error_message):
        """Send error message via DM"""
        try:
            await self.client.create_direct_message(
                participant_id=user_id,
                text=f"Error processing your tweet: {error_message}"
            )
        except Exception as e:
            logging.error(f"Error sending DM: {str(e)}")

def setup_twitter_bot():
    """Setup and return Twitter bot instance"""
    return TwitterBot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Twitter bot with selected API and model')
    parser.add_argument('--api', choices=['ollama', 'openai', 'anthropic', 'vllm'],
                       default='ollama', help='Choose the API to use (default: ollama)')

    parser.add_argument('--model', type=str,
                       help='Specify the model to use. If not provided, defaults will be used based on the API.')
    args = parser.parse_args()
    
    initialize_api_client(args)
    
    bot = setup_twitter_bot()
    asyncio.run(bot.start())
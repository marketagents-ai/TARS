# web_bot.py
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import logging
import asyncio
import os
import mimetypes
import tiktoken
import json
from datetime import datetime
import yaml
import io
import traceback
from PIL import Image
from typing import List, Optional, Dict, Any
import argparse
from dataclasses import dataclass
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi import WebSocket, WebSocketDisconnect
from nltk.tokenize import sent_tokenize

# Add this at the start of the file, before other imports
import sys
import os

# Get the project root directory (one level up from prototypes)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'agent'))  # Add this line

# Define script_dir for use in WebBot class
script_dir = os.path.dirname(os.path.abspath(__file__))

# Now imports should work
from agent.bot_config import *
from agent.api_client import initialize_api_client, call_api
from agent.cache_manager import CacheManager
from agent.memory import UserMemoryIndex

# Add constants here, before the RateLimiter class
RATE_LIMIT_REQUESTS = 100  # Adjust these values as needed
RATE_LIMIT_WINDOW = 3600  # Time window in seconds (1 hour)
FASTAPI_HOST = "localhost"  # Default host
FASTAPI_PORT = 8000       # Default port
FASTAPI_CORS_ORIGINS = ["*"]  # Adjust based on your security requirements
DEFAULT_PERSONA_INTENSITY = 0.7  # Matching discord_bot.py pattern

# Set up logging - matches discord_bot.py pattern
log_level = os.getenv('LOGLEVEL', 'INFO')
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

# JSONL logging setup - matches discord_bot.py pattern
def log_to_jsonl(data):
    with open('web_bot_log.jsonl', 'a') as f:
        json.dump(data, f)
        f.write('\n')

# Rate limiting handler - similar to twitter_bot.py pattern
class RateLimiter:
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, 
                 window: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}
        
    async def check_rate_limit(self, user_id: str):
        now = datetime.now().timestamp()
        user_requests = self.requests.get(user_id, [])
        
        # Remove old requests
        user_requests = [req for req in user_requests 
                        if now - req < self.window]
        
        if len(user_requests) >= self.max_requests:
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. Try again in {self.window} seconds."
            )
            
        user_requests.append(now)
        self.requests[user_id] = user_requests
        return True

@dataclass
class ChatRequest:
    user_id: str
    user_name: str
    message: str

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logging.info(f"WebSocket client {client_id} connected")

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logging.info(f"WebSocket client {client_id} disconnected")

    async def send_personal(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

async def process_message(message, memory_index, prompt_formats, system_prompts, cache_manager, github_repo=None):
    """Process incoming messages similar to Discord bot's message handling"""
    try:
        # Handle message input whether it's a dict or string
        if isinstance(message, dict):
            user_id = message.get('user_id')
            user_name = message.get('user_name')
            content = message.get('message')
        else:
            # If message is a string, we need these values from somewhere
            content = message
            user_id = "default_user"
            user_name = "default_user"

        # Get conversation history
        conversation_history = cache_manager.get_conversation_history(user_id)
        is_first_interaction = not bool(conversation_history)

        # Build context
        context = f"User {user_name} says: {content}\n\n"
        
        if conversation_history:
            context += "**Previous Conversation:**\n"
            for msg in conversation_history[-5:]:  # Show last 5 interactions
                context += f"User: {msg.get('user_message', '')}\n"
                context += f"Assistant: {msg.get('ai_response', '')}\n"
            context += "\n"

        # Get relevant memories
        relevant_memories = memory_index.search(content, user_id=user_id)
        if relevant_memories:
            context += "**Relevant memories:**\n"
            for memory, score in relevant_memories:
                context += f"[Relevance: {score:.2f}] {memory}\n"
            context += "\n"

        # Select appropriate prompt template and handle missing keys
        prompt_key = 'introduction_web' if is_first_interaction else 'chat_with_memory'
        if prompt_key not in prompt_formats:
            logging.warning(f"Missing prompt format key: {prompt_key}, falling back to default")
            prompt = f"Context:\n{context}\n\nUser Message: {content}\n\nPlease provide a helpful response."
        else:
            try:
                prompt = prompt_formats[prompt_key].format(
                    context=context,
                    user_message=content,
                    user_name=user_name
                )
            except KeyError as e:
                logging.error(f"Error formatting prompt: {str(e)}")
                prompt = f"Context:\n{context}\n\nUser Message: {content}\n\nPlease provide a helpful response."

        # Get system prompt
        system_prompt = system_prompts.get('default_web_chat', "You are a helpful AI assistant.")
        system_prompt = system_prompt.replace('{persona_intensity}', str(DEFAULT_PERSONA_INTENSITY))

        # Call API with both prompts
        response = await call_api(prompt, system_prompt=system_prompt)

        # Save to conversation history
        cache_manager.append_to_conversation(user_id, {
            'user_name': user_name,
            'user_message': content,
            'ai_response': response
        })

        # Save to memory
        memory_text = f"User {user_name}: {content}\nAssistant: {response}"
        memory_index.add_memory(user_id, memory_text)

        # Generate and save thought
        await generate_and_save_thought(
            memory_index=memory_index,
            user_id=user_id,
            user_name=user_name,
            memory_text=memory_text,
            prompt_formats=prompt_formats,
            system_prompts=system_prompts,
            bot=None  # Pass None since we don't have a bot instance
        )

        # Log interaction
        log_to_jsonl({
            'event': 'message_processed',
            'user_id': user_id,
            'user_name': user_name,
            'message': content,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })

        return response

    except Exception as e:
        logging.error(f"Error processing message: {str(e)}")
        traceback.print_exc()  # Add this for more detailed error information
        return f"Error: {str(e)}"

async def process_files(files, memory_index, prompt_formats, system_prompts, user_message, temperature):
    """Process file uploads similar to Discord bot's file handling"""
    try:
        # Process each file
        responses = []
        for file in files:
            filename = file['filename']
            content = file['content']
            content_type = file['content_type']

            # Handle different file types like Discord bot
            if content_type.startswith('image/'):
                # Process image
                image = Image.open(io.BytesIO(content))
                # Add image processing logic here
                responses.append(f"Processed image: {filename}")
            else:
                # Process other file types
                text_content = content.decode('utf-8')
                responses.append(f"Processed file: {filename}")

        return "\n".join(responses)

    except Exception as e:
        logging.error(f"Error processing files: {str(e)}")
        return f"Error: {str(e)}"

async def generate_and_save_thought(memory_index, user_id, user_name, memory_text, prompt_formats, system_prompts, bot):
    """
    Generates a thought about a memory and saves both to the memory index.
    """
    # Get current timestamp and format it as hh:mm [dd/mm/yy]
    timestamp = datetime.now().strftime("%H:%M [%d/%m/%y]")

    # Generate thought
    thought_prompt = prompt_formats['generate_thought'].format(
        user_name=user_name,
        memory_text=memory_text,
        timestamp=timestamp
    )
    
    thought_system_prompt = system_prompts['thought_generation'].replace('{persona_intensity}', str(DEFAULT_PERSONA_INTENSITY))
    
    thought_response = await call_api(thought_prompt, context="", system_prompt=thought_system_prompt, temperature=TEMPERATURE)
    
    # Save both the original memory and the thought
    memory_index.add_memory(user_id, memory_text)
    memory_index.add_memory(user_id, f"Priors on interactions with @{user_name}: {thought_response} (Timestamp: {timestamp})")

def truncate_middle(text: str, max_length: int) -> str:
    """Truncate text in the middle if it exceeds max_length, preserving start and end."""
    if len(text) <= max_length:
        return text
    half = (max_length - 3) // 2
    return text[:half] + "..." + text[-half:]

class WebBot:
    """Main web bot class matching structure of other bots"""
    def __init__(self):
        # Initialize with same components as Discord bot
        self.memory_index = UserMemoryIndex('web_memory_index')
        self.cache_manager = CacheManager('web_conversation_history')
        
        # Load prompts from agent directory
        agent_dir = os.path.join(project_root, 'agent')
        with open(os.path.join(agent_dir, 'prompts', 'prompt_formats.yaml'), 'r') as file:
            self.prompt_formats = yaml.safe_load(file)
        with open(os.path.join(agent_dir, 'prompts', 'system_prompts.yaml'), 'r') as file:
            self.system_prompts = yaml.safe_load(file)

        # Set persona intensity like Discord bot
        self.persona_intensity = DEFAULT_PERSONA_INTENSITY
        
        # Initialize FastAPI components
        self.app = FastAPI(title="AI Chat Bot")
        
        # Update static files path to use gui/web_bot
        static_dir = os.path.join(project_root, "prototypes", "web_bot")
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        # Store index file path for root route
        self.index_file = os.path.join(static_dir, "index.html")
        
        # Setup other components
        self.setup_cors()
        self.ws_manager = WebSocketManager()
        self.rate_limiter = RateLimiter()
        
        # Initialize cache managers like Discord bot
        self.cache_managers = {
            'conversation': CacheManager('web_conversation_history'),
            'file': CacheManager('web_file_cache'),
            'prompt': CacheManager('web_prompt_cache')
        }
        
        self.setup_routes()

    def setup_cors(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=FASTAPI_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        """Setup API routes reusing core processing functions"""
        
        @self.app.get("/")
        async def root():
            # Serve the index.html file at the root
            if os.path.exists(self.index_file):
                return FileResponse(self.index_file)
            else:
                return {"error": "index.html not found"}
        
        @self.app.post("/chat")
        async def chat_endpoint(request: ChatRequest):
            # Rate limiting check
            await self.rate_limiter.check_rate_limit(request.user_id)
            
            # Convert to standard message format
            message = self.convert_request_to_message(request)
            
            # Use shared process_message function
            return await process_message(
                message=message,
                memory_index=self.memory_index,
                prompt_formats=self.prompt_formats,
                system_prompts=self.system_prompts,
                cache_manager=self.cache_manager,
                github_repo=None
            )

        @self.app.post("/upload")
        async def upload_files(
            files: List[UploadFile],
            user_id: str,
            user_name: str,
            message: Optional[str] = None
        ):
            # Rate limiting check
            await self.rate_limiter.check_rate_limit(user_id)
            
            # Use shared process_files function
            return await process_files(
                files=self.convert_files_to_standard(files),
                memory_index=self.memory_index,
                prompt_formats=self.prompt_formats,
                system_prompts=self.system_prompts,
                user_message=message,
                temperature=self.temperature
            )

        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self.ws_manager.connect(websocket, client_id)
            try:
                while True:
                    try:
                        data = await websocket.receive_json()
                        logging.info(f"Received WebSocket message from {client_id}: {data}")
                        
                        # Process message and get response
                        response = await process_message(
                            message=data,  # Pass the entire data dictionary
                            memory_index=self.memory_index,
                            prompt_formats=self.prompt_formats,
                            system_prompts=self.system_prompts,
                            cache_manager=self.cache_manager,
                            github_repo=None
                        )
                        
                        # Send response
                        await self.ws_manager.send_personal(client_id, response)
                        
                    except WebSocketDisconnect:
                        logging.info(f"WebSocket client {client_id} disconnected")
                        break
                    except Exception as e:
                        logging.error(f"WebSocket error for {client_id}: {str(e)}")
                        error_message = f"Error: {str(e)}"
                        await self.ws_manager.send_personal(client_id, error_message)
            finally:
                await self.ws_manager.disconnect(client_id)

        # Memory management endpoints matching discord_bot.py commands
        @self.app.get("/memories/{user_id}")
        async def get_memories(user_id: str, query: Optional[str] = None):
            await self.rate_limiter.check_rate_limit(user_id)
            if query:
                return self.memory_index.search(query, user_id=user_id)
            else:
                return self.memory_index.get_user_memories(user_id)

        @self.app.post("/memories/{user_id}")
        async def add_memory(user_id: str, memory: str):
            await self.rate_limiter.check_rate_limit(user_id)
            self.memory_index.add_memory(user_id, memory)
            return {"status": "success"}

        @self.app.delete("/memories/{user_id}")
        async def clear_memories(user_id: str):
            await self.rate_limiter.check_rate_limit(user_id)
            self.memory_index.clear_user_memories(user_id)
            return {"status": "success"}

        @self.app.get("/search_memories/{user_id}")
        async def search_memories(user_id: str, query: str):
            # Rate limiting check
            await self.rate_limiter.check_rate_limit(user_id)
            
            results = self.memory_index.search(query, user_id=user_id)
            if not results:
                return {"message": "No results found"}
                
            formatted_results = [
                {
                    "memory": truncate_middle(memory, 800),
                    "relevance": float(score)
                }
                for memory, score in results
            ]
            
            return {"results": formatted_results}

    def convert_request_to_message(self, request):
        """Convert web request to standard message format matching Discord's"""
        return type('WebMessage', (), {
            'author': type('Author', (), {
                'id': request.user_id,
                'name': request.user_name
            }),
            'content': request.message,
            'channel': type('Channel', (), {
                'name': 'web',
                'send': self.send_response
            })
        })

    def convert_files_to_standard(self, files):
        """Convert web uploads to standard format matching Discord's"""
        return [
            {
                'filename': file.filename,
                'content': file.file.read(),
                'content_type': file.content_type
            }
            for file in files
        ]

    async def process_websocket_message(self, message_data: dict) -> str:
        try:
            # Process the message and return a string response
            response = await process_message(
                message=message_data.get('message', ''),
                memory_index=self.memory_index,
                prompt_formats=self.prompt_formats,
                system_prompts=self.system_prompts,
                cache_manager=self.cache_manager,
                github_repo=None
            )
            # Ensure response is a string
            if isinstance(response, dict):
                return json.dumps(response)
            return str(response)
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            return f"Error processing message: {str(e)}"

def setup_web_bot():
    """Setup and return web bot instance matching other bots' pattern"""
    return WebBot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the web bot with selected API and model')
    parser.add_argument('--api', choices=['ollama', 'openai', 'anthropic', 'vllm'],
                       default='ollama', help='Choose the API to use (default: ollama)')
    parser.add_argument('--model', type=str,
                       help='Specify the model to use. If not provided, defaults will be used based on the API.')
    parser.add_argument('--host', type=str, default=FASTAPI_HOST,
                       help='Host to run the server on')
    parser.add_argument('--port', type=int, default=FASTAPI_PORT,
                       help='Port to run the server on')
    
    args = parser.parse_args()
    
    initialize_api_client(args)
    
    bot = setup_web_bot()
    
    import uvicorn
    uvicorn.run(bot.app, host=args.host, port=args.port)
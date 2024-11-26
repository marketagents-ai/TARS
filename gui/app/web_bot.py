# web_bot.py

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request, WebSocket, WebSocketDisconnect
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
from typing import List, Optional, Dict, Any, Union
import argparse
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize
from argparse import Namespace

# Add project root to path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import internal modules
from agent.bot_config import *
from agent.api_client import initialize_api_client, call_api
from agent.cache_manager import CacheManager
from gui.app.memory_manager import MemoryManager
from gui.app.models import MemoryCreate, MemoryUpdate, MemoryResponse

# Constants
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600
FASTAPI_HOST = "localhost"  
FASTAPI_PORT = 8000       
FASTAPI_CORS_ORIGINS = ["*"]
DEFAULT_PERSONA_INTENSITY = 0.7
DEFAULT_TEMPERATURE = 0.7

# Configure logging
log_level = os.getenv('LOGLEVEL', 'INFO')
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('web_bot.log')
    ]
)
logger = logging.getLogger(__name__)

def log_to_jsonl(data: dict):
    """Log events to JSONL file with error handling"""
    try:
        with open('web_bot_log.jsonl', 'a') as f:
            json.dump(data, f)
            f.write('\n')
    except Exception as e:
        logger.error(f"Error writing to log file: {str(e)}")

# Data Models
@dataclass
class ChatRequest:
    user_id: str
    user_name: str
    message: str

class FileUploadRequest(BaseModel):
    user_id: str
    user_name: str
    message: Optional[str] = None


# Rate Limiter and WebSocket Manager

class RateLimiter:
    """Handles rate limiting for API requests"""
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, 
                 window: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, List[float]] = {}
        
    async def check_rate_limit(self, user_id: str) -> bool:
        now = datetime.now().timestamp()
        user_requests = self.requests.get(user_id, [])
        
        # Clean old requests
        user_requests = [req for req in user_requests if now - req < self.window]
        
        if len(user_requests) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {self.window} seconds."
            )
        
        user_requests.append(now)
        self.requests[user_id] = user_requests
        return True

class WebSocketManager:
    """Manages WebSocket connections and message handling"""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client {client_id} connected")

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket client {client_id} disconnected")

    async def send_personal(self, client_id: str, message: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {str(e)}")
                await self.disconnect(client_id)
    
    async def process_and_send(self, 
                             client_id: str,
                             message_data: dict,
                             memory_index: MemoryManager,
                             prompt_formats: dict,
                             system_prompts: dict,
                             cache_manager: CacheManager):
        """Process and send message with proper prompt ordering"""
        try:
            # Process message with correct prompt order
            response = await process_message(
                message=message_data,
                memory_index=memory_index,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                cache_manager=cache_manager
            )
            
            # Send response only after all processing is complete
            if client_id in self.active_connections:
                await self.active_connections[client_id].send_text(response)
                
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc()
            if client_id in self.active_connections:
                await self.active_connections[client_id].send_text(error_msg)

async def generate_and_save_thought(
    memory_index: MemoryManager,
    user_id: str,
    user_name: str,
    memory_text: str,
    prompt_formats: dict,
    system_prompts: dict,
    bot: Optional[Any] = None
) -> Optional[str]:
    """Generate and save a thought about an interaction"""
    try:
        timestamp = datetime.now().strftime("%H:%M [%d/%m/%y]")
        
        # Generate thought using the correct prompts
        thought_prompt = prompt_formats['generate_thought'].format(
            user_name=user_name,
            memory_text=memory_text,
            timestamp=timestamp
        )
        
        thought_system_prompt = system_prompts['thought_generation'].replace(
            '{persona_intensity}', 
            str(DEFAULT_PERSONA_INTENSITY)
        )
        
        # Generate thought with error handling
        try:
            thought_response = await call_api(
                thought_prompt,
                system_prompt=thought_system_prompt,
                temperature=DEFAULT_TEMPERATURE
            )
        except Exception as e:
            logger.error(f"Error generating thought: {str(e)}")
            return None

        # Save both the interaction and thought memories
        try:
            memory_index.add_memory(user_id, memory_text)
            thought_memory = f"Thought about @{user_name}: {thought_response} (Timestamp: {timestamp})"
            memory_index.add_memory(user_id, thought_memory)
            
            logger.info(f"Saved thought for user {user_name}: {thought_response[:100]}...")
            return thought_response
            
        except Exception as e:
            logger.error(f"Error saving memories: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"Error in generate_and_save_thought: {str(e)}")
        traceback.print_exc()
        return None

async def process_message(
    message: Union[dict, str],
    memory_index: MemoryManager,
    prompt_formats: dict,
    system_prompts: dict,
    cache_manager: CacheManager,
    github_repo: Optional[str] = None
) -> str:
    """Process incoming messages with proper prompt ordering"""
    try:
        # 1. Extract message details
        if isinstance(message, dict):
            user_id = message.get('user_id')
            user_name = message.get('user_name')
            content = message.get('message')
        else:
            content = message
            user_id = "default_user"
            user_name = "default_user"

        # 2. Get system prompt BEFORE any API calls
        system_prompt = system_prompts.get('default_web_chat', 
            "You are TARS, an AI assistant. Your humor setting is {persona_intensity}%."
        ).replace('{persona_intensity}', str(DEFAULT_PERSONA_INTENSITY))

        # 3. Build context and get history
        conversation_history = cache_manager.get_conversation_history(user_id)
        is_first_interaction = not bool(conversation_history)
        
        context = f"User {user_name} says: {content}\n\n"
        
        if conversation_history:
            context += "**Previous Conversation:**\n"
            for msg in conversation_history[-5:]:
                context += f"User: {msg.get('user_message', '')}\n"
                context += f"Assistant: {msg.get('ai_response', '')}\n"
            context += "\n"

        # 4. Get relevant memories
        relevant_memories = memory_index.search_memories(content, user_id=user_id)
        if relevant_memories:
            context += "**Relevant memories:**\n"
            for memory, memory_id, score in relevant_memories:
                context += f"[Relevance: {score:.2f}] {memory}\n"
            context += "\n"

        # 5. Select and format prompt
        prompt_key = 'introduction_web' if is_first_interaction else 'chat_with_memory'
        if prompt_key not in prompt_formats:
            logger.warning(f"Missing prompt format key: {prompt_key}, falling back to default")
            prompt = f"Context:\n{context}\n\nUser Message: {content}\n\nPlease provide a helpful response."
        else:
            try:
                prompt = prompt_formats[prompt_key].format(
                    context=context,
                    user_message=content,
                    user_name=user_name
                )
            except KeyError as e:
                logger.error(f"Error formatting prompt: {str(e)}")
                prompt = f"Context:\n{context}\n\nUser Message: {content}\n\nPlease provide a helpful response."

        # 6. Generate response
        try:
            response = await call_api(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=DEFAULT_TEMPERATURE
            )
        except Exception as e:
            logger.error(f"Error calling API: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"

        # 7. Save to conversation history
        cache_manager.append_to_conversation(user_id, {
            'user_name': user_name,
            'user_message': content,
            'ai_response': response
        })

        # 8. Generate and save thought
        memory_text = f"User {user_name}: {content}\nAssistant: {response}"
        await generate_and_save_thought(
            memory_index=memory_index,
            user_id=user_id,
            user_name=user_name,
            memory_text=memory_text,
            prompt_formats=prompt_formats,
            system_prompts=system_prompts
        )

        # 9. Log interaction
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
        logger.error(f"Error processing message: {str(e)}")
        traceback.print_exc()
        return f"I apologize, but I encountered an error: {str(e)}"
    
class WebBot:
    """Main web bot class integrating all components"""
    def __init__(self):
        # Initialize API client
        args = Namespace(
            api='anthropic',
            model='claude-3-sonnet-20240229'
        )
        initialize_api_client(args)
        
        # Initialize memory manager with correct path
        memory_cache_path = os.path.join(project_root, "cache", "discord_bot", "user_memory_index", "memory_cache.pkl")
        logger.info(f"Initializing memory manager with path: {memory_cache_path}")
        self.memory_manager = MemoryManager(memory_cache_path)
        
        # Initialize cache manager
        self.cache_manager = CacheManager('web_conversation_history')
        
        # Load prompts from correct paths
        prompts_path = os.path.join(project_root, 'agent', 'prompts')
        try:
            with open(os.path.join(prompts_path, 'prompt_formats.yaml'), 'r', encoding='utf-8') as f:
                prompt_formats = yaml.safe_load(f)
            with open(os.path.join(prompts_path, 'system_prompts.yaml'), 'r', encoding='utf-8') as f:
                system_prompts = yaml.safe_load(f)
                
            logger.info("Successfully loaded prompt files")
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            prompt_formats = {}
            system_prompts = {}

        # Initialize prompt formats with fallbacks
        self.prompt_formats = {
            'chat_with_memory': prompt_formats.get('chat_with_memory', 
                "Context:\n{context}\n\nUser Message: {user_message}\n\nPlease provide a helpful response."),
            'introduction_web': prompt_formats.get('introduction_web',
                "This is your first interaction with {user_name}. Please provide a friendly greeting."),
            'generate_thought': prompt_formats.get('generate_thought',
                "Based on this interaction with {user_name}, generate a concise thought or insight:\n{memory_text}\nTimestamp: {timestamp}"),
            'analyze_code': prompt_formats.get('analyze_code', 
                "Please analyze this code:\n{code}"),
            'analyze_file': prompt_formats.get('analyze_file',
                "Please analyze this file content:\n{content}")
        }
        
        # Initialize system prompts with fallbacks
        self.system_prompts = {
            'default_web_chat': system_prompts.get('default_web_chat', 
                "You are TARS, an AI assistant. Your humor setting is {persona_intensity}%."),
            'thought_generation': system_prompts.get('thought_generation',
                "Generate an insightful thought about this interaction. Humor setting: {persona_intensity}%."),
            'file_analysis': system_prompts.get('file_analysis',
                "Analyze this file content thoroughly.")
        }

        # Set persona intensity
        self.persona_intensity = DEFAULT_PERSONA_INTENSITY
        
        # Initialize FastAPI app
        self.app = FastAPI(title="AI Chat Bot")
        
        # Setup components
        self.setup_cors()
        self.ws_manager = WebSocketManager()
        self.rate_limiter = RateLimiter()
        
        # Initialize cache managers
        self.cache_managers = {
            'conversation': CacheManager('web_conversation_history'),
            'file': CacheManager('web_file_cache'),
            'prompt': CacheManager('web_prompt_cache')
        }
        
        # Setup routes last
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

    async def process_websocket_message(self, message_data: dict) -> str:
        """Process WebSocket messages and return response"""
        try:
            # Get message components
            message = message_data.get('message', '')
            user_id = message_data.get('user_id', '')
            user_name = message_data.get('user_name', '')

            # Process the message using shared process_message function
            response = await process_message(
                message=message_data,  # Pass the entire message data
                memory_index=self.memory_manager,
                prompt_formats=self.prompt_formats,
                system_prompts=self.system_prompts,
                cache_manager=self.cache_manager
            )

            # Generate and save thought/memory
            memory_text = f"Chat interaction - User: {message} | Agent: {response}"
            await generate_and_save_thought(
                memory_index=self.memory_manager,
                user_id=user_id,
                user_name=user_name,
                memory_text=memory_text,
                prompt_formats=self.prompt_formats,
                system_prompts=self.system_prompts
            )

            return response

        except Exception as e:
            logger.error(f"Error processing WebSocket message: {str(e)}")
            traceback.print_exc()
            return f"Error processing message: {str(e)}"

    def setup_routes(self):
        """Setup API routes with proper error handling"""
        
        @self.app.get("/")
        async def root():
            """Serve the main interface"""
            index_path = os.path.join(project_root, "gui", "static", "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"error": "index.html not found"}
        
        @self.app.post("/chat")
        async def chat_endpoint(request: ChatRequest):
            """Handle chat messages"""
            try:
                # Rate limiting check
                await self.rate_limiter.check_rate_limit(request.user_id)
                
                # Process message
                response = await process_message(
                    message={
                        'user_id': request.user_id,
                        'user_name': request.user_name,
                        'message': request.message
                    },
                    memory_index=self.memory_manager,
                    prompt_formats=self.prompt_formats,
                    system_prompts=self.system_prompts,
                    cache_manager=self.cache_manager
                )
                
                return response
            except HTTPException as he:
                raise he
            except Exception as e:
                logger.error(f"Error in chat endpoint: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error: {str(e)}"
                )

        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """Handle WebSocket connections with proper error handling"""
            await self.ws_manager.connect(websocket, client_id)
            try:
                while True:
                    try:
                        # Receive and parse message
                        data = await websocket.receive_json()
                        logger.info(f"Received WebSocket message from {client_id}")
                        
                        # Process message using dedicated method
                        response = await self.process_websocket_message(data)
                        
                        # Send response
                        await self.ws_manager.send_personal(client_id, response)
                        
                    except WebSocketDisconnect:
                        logger.info(f"WebSocket client {client_id} disconnected")
                        break
                    except json.JSONDecodeError:
                        error_msg = "Error: Invalid message format"
                        logger.error(error_msg)
                        await self.ws_manager.send_personal(client_id, error_msg)
                    except Exception as e:
                        error_msg = f"Error processing message: {str(e)}"
                        logger.error(error_msg)
                        traceback.print_exc()
                        await self.ws_manager.send_personal(client_id, error_msg)
            finally:
                await self.ws_manager.disconnect(client_id)

        @self.app.get("/memories/{user_id}")
        async def get_memories(user_id: str, query: Optional[str] = None):
            """Get user memories with optional search"""
            await self.rate_limiter.check_rate_limit(user_id)
            try:
                if query:
                    results = self.memory_manager.search_memories(query, user_id=user_id)
                    return [
                        {
                            "content": memory,
                            "memory_id": memory_id,
                            "relevance_score": score
                        }
                        for memory, memory_id, score in results
                    ]
                else:
                    memories = self.memory_manager.get_user_memories(user_id)
                    return [
                        {
                            "content": memory,
                            "memory_id": memory_id
                        }
                        for memory, memory_id in memories
                    ]
            except Exception as e:
                logger.error(f"Error getting memories: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error retrieving memories: {str(e)}"
                )

        @self.app.post("/memories/{user_id}")
        async def add_memory(user_id: str, memory: MemoryCreate):
            """Add new memory"""
            await self.rate_limiter.check_rate_limit(user_id)
            try:
                memory_id = self.memory_manager.add_memory(
                    user_id,
                    memory.content
                )
                return {"status": "success", "memory_id": memory_id}
            except Exception as e:
                logger.error(f"Error adding memory: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error adding memory: {str(e)}"
                )

        @self.app.put("/memories/{memory_id}")
        async def update_memory(memory_id: int, update: MemoryUpdate):
            """Update existing memory"""
            try:
                result = self.memory_manager.update_memory(memory_id, update)
                if result:
                    return {"status": "success", "memory": result}
                raise HTTPException(status_code=404, detail="Memory not found")
            except Exception as e:
                logger.error(f"Error updating memory: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error updating memory: {str(e)}"
                )

        @self.app.delete("/memories/{memory_id}")
        async def delete_memory(memory_id: int):
            """Delete memory"""
            try:
                success = self.memory_manager.delete_memory(memory_id)
                if success:
                    return {"status": "success"}
                raise HTTPException(status_code=404, detail="Memory not found")
            except Exception as e:
                logger.error(f"Error deleting memory: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error deleting memory: {str(e)}"
                )
            
def setup_web_bot() -> WebBot:
    """Setup and return web bot instance with error handling"""
    try:
        return WebBot()
    except Exception as e:
        logger.error(f"Error setting up web bot: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the web bot')
    parser.add_argument(
        '--api',
        choices=['ollama', 'openai', 'anthropic', 'vllm'],
        default='anthropic',
        help='Choose the API to use'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='claude-3-sonnet-20240229',
        help='Specify the model to use'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=FASTAPI_HOST,
        help='Host to run the server on'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=FASTAPI_PORT,
        help='Port to run the server on'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    
    args = parser.parse_args()
    
    # Configure logging based on arguments
    logging.getLogger().setLevel(args.log_level)
    
    # Initialize components
    try:
        # Initialize API client
        initialize_api_client(args)
        logger.info(f"Initialized API client with {args.api} using model {args.model}")
        
        # Setup bot
        bot = setup_web_bot()
        logger.info("Web bot setup completed successfully")
        
        # Run server
        import uvicorn
        logger.info(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(
            bot.app,
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower()
        )
        
    except Exception as e:
        logger.error(f"Fatal error starting web bot: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
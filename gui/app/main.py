from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import json
import traceback
import sys
import os
from argparse import Namespace
from agent.api_client import initialize_api_client

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from gui.app.api.endpoints import router as api_router, get_memory_manager
from gui.app.web_bot import WebBot
from gui.app.memory_manager import MemoryManager

app = FastAPI(title="Memory Editor API")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API routes first
app.include_router(api_router, prefix="/api")

# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                response = await web_bot.process_websocket_message(message_data)
                await websocket.send_text(response)
            except json.JSONDecodeError:
                await websocket.send_text("Error: Invalid message format")
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        traceback.print_exc()

# Initialize API client with default settings
args = Namespace(
    api='anthropic',  # or your preferred API
    model='claude-3-sonnet-20240229'  # or your preferred model
)
initialize_api_client(args)

# Initialize the web bot
web_bot = WebBot()

# Initialize memory manager
memory_manager = MemoryManager('web_memory.pkl')

def get_memory_manager():
    return memory_manager

# Root endpoint to serve index.html
@app.get("/")
async def read_root():
    return FileResponse("gui/static/index.html")

# Mount static files last
app.mount("/static", StaticFiles(directory="gui/static"), name="static")
app.mount("/css", StaticFiles(directory="gui/static/css"), name="css")
app.mount("/js", StaticFiles(directory="gui/static/js"), name="js")
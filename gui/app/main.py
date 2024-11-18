from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging

from gui.app.api.endpoints import router as api_router, get_memory_manager

app = FastAPI(title="Memory Editor API")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - this should be AFTER API routes to prevent conflicts
app.include_router(api_router, prefix="/api")

# Serve static files from gui/static directory
app.mount("/", StaticFiles(directory="gui/static", html=True), name="static")

# Remove the root endpoint since static files will be served at /
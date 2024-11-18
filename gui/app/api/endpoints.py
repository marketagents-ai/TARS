from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import os
from pathlib import Path
import logging

from ..models import MemoryResponse, MemoryUpdate, MemorySearchParams, MemoryCreate
from ..memory_manager import MemoryManager

router = APIRouter()
memory_manager = None

def get_memory_manager():
    global memory_manager
    if memory_manager is None:
        # Get the root directory (where the script is running)
        root_dir = Path.cwd()
        
        # Construct cache path with correct filename
        cache_dir = root_dir /"cache" / "discord_bot" / "user_memory_index"
        cache_path = cache_dir / "memory_cache.pkl"
        
        # Debug logging
        logging.info(f"Project root directory: {root_dir.absolute()}")
        logging.info(f"Looking for cache at: {cache_path.absolute()}")
        logging.info(f"Cache directory exists: {cache_dir.exists()}")
        logging.info(f"Cache file exists: {cache_path.exists()}")
        
        if cache_path.exists():
            logging.info(f"Cache file size: {cache_path.stat().st_size} bytes")
        else:
            logging.warning("Cache file does not exist - will create new empty cache")
        
        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory manager with absolute path
        memory_manager = MemoryManager(str(cache_path.absolute()))
    
    return memory_manager

@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: int, mm: MemoryManager = Depends(get_memory_manager)):
    """Get a single memory by ID"""
    memory = mm.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse(content=memory, memory_id=memory_id)

@router.get("/memories/user/{user_id}", response_model=List[MemoryResponse])
async def get_user_memories(user_id: str, mm: MemoryManager = Depends(get_memory_manager)):
    """Get all memories for a user"""
    memories = mm.get_user_memories(user_id)
    return [MemoryResponse(content=content, memory_id=mid) for content, mid in memories]

@router.post("/memories", response_model=MemoryResponse)
async def create_memory(memory: MemoryCreate, mm: MemoryManager = Depends(get_memory_manager)):
    """Create a new memory"""
    memory_id = mm.add_memory(memory.user_id, memory.content)
    return MemoryResponse(content=memory.content, memory_id=memory_id)

@router.put("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: int,
    update_data: MemoryUpdate,
    mm: MemoryManager = Depends(get_memory_manager)
):
    """Update an existing memory"""
    updated_memory = mm.update_memory(memory_id, update_data)
    if not updated_memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse(content=updated_memory, memory_id=memory_id)

@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int, mm: MemoryManager = Depends(get_memory_manager)):
    """Delete a memory"""
    if not mm.delete_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "success"}

@router.post("/memories/search", response_model=List[MemoryResponse])
async def search_memories(
    params: MemorySearchParams,
    mm: MemoryManager = Depends(get_memory_manager)
):
    """Search memories"""
    results = mm.search_memories(params.query, params.user_id, params.limit)
    return [
        MemoryResponse(
            content=memory, 
            memory_id=memory_id, 
            relevance_score=score
        ) 
        for memory, memory_id, score in results
    ]

@router.get("/users", response_model=List[str])
async def get_users(mm: MemoryManager = Depends(get_memory_manager)):
    """Get list of all users"""
    return mm.get_all_users()

@router.get("/validate")
async def validate_db(mm: MemoryManager = Depends(get_memory_manager)):
    """Validate database integrity"""
    return mm.validate_db()
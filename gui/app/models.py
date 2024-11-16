from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class MemoryResponse(BaseModel):
    """Response model for memory data"""
    content: str
    memory_id: int
    relevance_score: Optional[float] = None

class MemoryUpdate(BaseModel):
    """Model for memory update operations"""
    content: str = Field(..., description="The new content for the memory")

class MemorySearchParams(BaseModel):
    """Parameters for memory search operations"""
    query: str = Field(..., description="Search query string")
    user_id: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)

class MemoryCreate(BaseModel):
    """Model for creating new memories"""
    user_id: str = Field(..., description="ID of the user this memory belongs to")
    content: str = Field(..., description="The memory content")
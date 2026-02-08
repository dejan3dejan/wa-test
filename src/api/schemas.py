"""Pydantic schemas for API request and response validation."""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Request schema for hybrid search query."""
    query: str = Field(..., description="The search query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for hybrid search (1.0=Dense, 0.0=Sparse)")
    namespace: str = Field(default="", description="Pinecone namespace to search in")
    with_audit: bool = Field(default=False, description="Whether to audit the top result with Gemini")

class AuditResult(BaseModel):
    """Schema for the LLM audit judge result."""
    label: Literal["factual", "hallucinated"]
    reasoning: str
    confidence_score: float

class SearchResult(BaseModel):
    """Single search result item."""
    id: str
    score: float
    metadata: Dict
    audit: Optional[AuditResult] = None

class QueryResponse(BaseModel):
    """Response schema for hybrid search."""
    query: str
    results: List[SearchResult]
    namespace: str

class UpsertItem(BaseModel):
    """Single item to be upserted."""
    id: str
    text: str
    metadata: Dict = {}

class UpsertRequest(BaseModel):
    """Request schema for batch vector upsert."""
    items: List[UpsertItem]
    namespace: str = ""

class StatsResponse(BaseModel):
    """Response schema for index statistics."""
    total_vector_count: int
    namespaces: Dict[str, Dict]
    index_fullness: float

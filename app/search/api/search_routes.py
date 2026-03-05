"""
Search API Routes
FastAPI routes for semantic search functionality
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field

from app.search.search_engine.semantic_search import SemanticSearchEngine
from app.utils.logger import logger

# Initialize router
router = APIRouter()

# Initialize search engine (singleton)
search_engine = None


def get_search_engine() -> SemanticSearchEngine:
    """Get or initialize search engine"""
    global search_engine
    
    if search_engine is None:
        search_engine = SemanticSearchEngine()
    
    return search_engine


# Request/Response Models
class SearchRequest(BaseModel):
    """Search request schema"""
    query: str = Field(..., description="Search query text", min_length=3)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    court: Optional[str] = Field(None, description="Filter by court name")
    case_type: Optional[str] = Field(None, description="Filter by case type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "land dispute ownership evidence",
                "top_k": 5,
                "court": "Delhi High Court",
                "case_type": "Civil"
            }
        }


class SearchResult(BaseModel):
    """Search result schema"""
    case_id: str
    chunk_id: str
    court: Optional[str]
    judge: Optional[str]
    date: Optional[str]
    case_type: Optional[str]
    score: float
    excerpt: str
    chunk_index: int


class SearchResponse(BaseModel):
    """Search response schema"""
    query: str
    total_results: int
    results: List[SearchResult]
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "land dispute ownership evidence",
                "total_results": 5,
                "results": [
                    {
                        "case_id": "case_123",
                        "chunk_id": "case_123_chunk_0",
                        "court": "Delhi High Court",
                        "judge": "Justice Sharma",
                        "date": "2023-05-15",
                        "case_type": "Civil",
                        "score": 0.89,
                        "excerpt": "In the matter of land ownership dispute...",
                        "chunk_index": 0
                    }
                ]
            }
        }


class SimilarCasesRequest(BaseModel):
    """Similar cases request schema"""
    case_id: str = Field(..., description="Case ID to find similar cases for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")


class IndexStatsResponse(BaseModel):
    """Index statistics response"""
    total_vectors: int
    unique_cases: int
    embedding_dim: int
    index_type: str
    is_loaded: bool


# API Endpoints
@router.post("/search", response_model=SearchResponse)
async def search_judgments(request: SearchRequest):
    """
    Search for semantically similar legal judgments
    
    Args:
        request: Search request with query and parameters
    
    Returns:
        Search results with similar cases
    """
    try:
        engine = get_search_engine()
        
        if not engine.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Search index not loaded. Please build the index first."
            )
        
        # Build filters
        filters = {}
        if request.court:
            filters['court'] = request.court
        if request.case_type:
            filters['case_type'] = request.case_type
        
        # Perform search
        results = engine.search(
            query=request.query,
            top_k=request.top_k,
            filters=filters if filters else None
        )
        
        # Convert to response format
        search_results = [
            SearchResult(
                case_id=r.get('case_id', 'unknown'),
                chunk_id=r.get('chunk_id', 'unknown'),
                court=r.get('court'),
                judge=r.get('judge'),
                date=r.get('date'),
                case_type=r.get('case_type'),
                score=r.get('score', 0.0),
                excerpt=r.get('excerpt', ''),
                chunk_index=r.get('chunk_index', 0)
            )
            for r in results
        ]
        
        logger.info(f"Search completed: '{request.query}' - {len(search_results)} results")
        
        return SearchResponse(
            query=request.query,
            total_results=len(search_results),
            results=search_results
        )
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/search/similar", response_model=SearchResponse)
async def find_similar_cases(request: SimilarCasesRequest):
    """
    Find cases similar to a given case
    
    Args:
        request: Request with case ID
    
    Returns:
        Similar cases
    """
    try:
        engine = get_search_engine()
        
        if not engine.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Search index not loaded"
            )
        
        # Search for similar cases
        results = engine.search_by_case_id(
            case_id=request.case_id,
            top_k=request.top_k
        )
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Case {request.case_id} not found in index"
            )
        
        # Convert to response format
        search_results = [
            SearchResult(
                case_id=r.get('case_id', 'unknown'),
                chunk_id=r.get('chunk_id', 'unknown'),
                court=r.get('court'),
                judge=r.get('judge'),
                date=r.get('date'),
                case_type=r.get('case_type'),
                score=r.get('score', 0.0),
                excerpt=r.get('excerpt', ''),
                chunk_index=r.get('chunk_index', 0)
            )
            for r in results
        ]
        
        return SearchResponse(
            query=f"Similar to case {request.case_id}",
            total_results=len(search_results),
            results=search_results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar cases error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/stats", response_model=IndexStatsResponse)
async def get_index_stats():
    """
    Get search index statistics
    
    Returns:
        Index statistics
    """
    try:
        engine = get_search_engine()
        stats = engine.get_index_stats()
        
        return IndexStatsResponse(
            total_vectors=stats.get('total_vectors', 0),
            unique_cases=stats.get('unique_cases', 0),
            embedding_dim=stats.get('embedding_dim', 384),
            index_type=stats.get('index_type', 'IP'),
            is_loaded=stats.get('is_loaded', False)
        )
    
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/reload")
async def reload_index():
    """
    Reload the search index from disk
    
    Returns:
        Reload status
    """
    try:
        engine = get_search_engine()
        success = engine.reload_index()
        
        if success:
            stats = engine.get_index_stats()
            return {
                "status": "success",
                "message": "Index reloaded successfully",
                "total_vectors": stats.get('total_vectors', 0),
                "unique_cases": stats.get('unique_cases', 0)
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Index file not found. Please build the index first."
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

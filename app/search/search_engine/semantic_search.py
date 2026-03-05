"""
Semantic Search Engine Module
Provides semantic search functionality for legal judgments
"""
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from app.search.embedding.embedding_generator import EmbeddingGenerator
from app.search.vector_store.faiss_index import FAISSVectorStore
from app.utils.logger import logger


class SemanticSearchEngine:
    """Semantic search engine for legal judgments"""
    
    def __init__(
        self,
        index_file: str = "data/embeddings/index.faiss",
        metadata_file: str = "data/embeddings/metadata.pkl",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize semantic search engine
        
        Args:
            index_file: Path to FAISS index file
            metadata_file: Path to metadata file
            model_name: Sentence transformer model name
        """
        self.index_file = index_file
        self.metadata_file = metadata_file
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(model_name=model_name)
        
        # Initialize vector store
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedding_generator.embedding_dim,
            index_type="IP"
        )
        
        # Load index if exists
        self.is_loaded = self._load_index()
        
        logger.info(f"Initialized semantic search engine. Index loaded: {self.is_loaded}")
    
    def _load_index(self) -> bool:
        """
        Load FAISS index
        
        Returns:
            True if loaded successfully
        """
        if Path(self.index_file).exists():
            return self.vector_store.load_index(self.index_file, self.metadata_file)
        else:
            logger.warning(f"Index file not found: {self.index_file}")
            return False
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar judgments
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional filters (court, case_type, etc.)
        
        Returns:
            List of search results with metadata
        """
        if not self.is_loaded:
            logger.error("Index not loaded. Please build index first.")
            return []
        
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search in vector store
        # Request more results if we need to filter
        search_k = top_k * 3 if filters else top_k
        results = self.vector_store.search_with_metadata(query_embedding, top_k=search_k)
        
        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)
        
        # Limit to top_k
        results = results[:top_k]
        
        # Add excerpts
        results = self._add_excerpts(results, query)
        
        logger.info(f"Found {len(results)} results")
        
        return results
    
    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """
        Apply filters to search results
        
        Args:
            results: Search results
            filters: Filter dictionary
        
        Returns:
            Filtered results
        """
        filtered = []
        
        for result in results:
            match = True
            
            # Check each filter
            for key, value in filters.items():
                if key in result and result[key] != value:
                    match = False
                    break
            
            if match:
                filtered.append(result)
        
        return filtered
    
    def _add_excerpts(self, results: List[Dict], query: str, excerpt_length: int = 200) -> List[Dict]:
        """
        Add text excerpts to results
        
        Args:
            results: Search results
            query: Original query
            excerpt_length: Maximum excerpt length
        
        Returns:
            Results with excerpts
        """
        for result in results:
            text = result.get('text', '')
            
            if text:
                # Create excerpt (simple approach - take first N chars)
                if len(text) > excerpt_length:
                    excerpt = text[:excerpt_length] + "..."
                else:
                    excerpt = text
                
                result['excerpt'] = excerpt
            else:
                result['excerpt'] = "No text available"
        
        return results
    
    def search_by_case_id(self, case_id: str, top_k: int = 5) -> List[Dict]:
        """
        Find similar cases to a given case
        
        Args:
            case_id: Case ID to find similar cases for
            top_k: Number of results to return
        
        Returns:
            List of similar cases
        """
        if not self.is_loaded:
            logger.error("Index not loaded")
            return []
        
        # Find the case in metadata
        case_chunks = [
            (i, meta) for i, meta in enumerate(self.vector_store.metadata)
            if meta.get('case_id') == case_id
        ]
        
        if not case_chunks:
            logger.warning(f"Case {case_id} not found in index")
            return []
        
        # Use the first chunk as query
        chunk_idx, chunk_meta = case_chunks[0]
        
        # Get the embedding for this chunk
        query_embedding = self.vector_store.index.reconstruct(chunk_idx)
        
        # Search for similar cases
        results = self.vector_store.search_with_metadata(query_embedding, top_k=top_k * 2)
        
        # Filter out chunks from the same case
        filtered_results = [
            r for r in results
            if r.get('case_id') != case_id
        ]
        
        # Group by case_id and take best match per case
        case_results = {}
        for result in filtered_results:
            cid = result.get('case_id')
            if cid and (cid not in case_results or result['score'] > case_results[cid]['score']):
                case_results[cid] = result
        
        # Sort by score and limit
        final_results = sorted(case_results.values(), key=lambda x: x['score'], reverse=True)[:top_k]
        
        return final_results
    
    def get_index_stats(self) -> Dict:
        """
        Get index statistics
        
        Returns:
            Statistics dictionary
        """
        stats = self.vector_store.get_stats()
        stats['is_loaded'] = self.is_loaded
        stats['index_file'] = self.index_file
        
        # Count unique cases
        if self.is_loaded:
            unique_cases = set(
                meta.get('case_id') for meta in self.vector_store.metadata
                if meta.get('case_id')
            )
            stats['unique_cases'] = len(unique_cases)
        
        return stats
    
    def reload_index(self) -> bool:
        """
        Reload the index from disk
        
        Returns:
            True if successful
        """
        logger.info("Reloading index...")
        self.is_loaded = self._load_index()
        return self.is_loaded
    
    def add_judgment(
        self,
        text: str,
        case_id: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add a new judgment to the index
        
        Args:
            text: Judgment text
            case_id: Case identifier
            metadata: Additional metadata
        """
        if not self.is_loaded:
            logger.error("Index not loaded")
            return
        
        # Process judgment
        chunks = self.embedding_generator.process_judgment(text, case_id, metadata)
        
        # Extract embeddings and metadata
        embeddings = np.array([chunk['embedding'] for chunk in chunks])
        chunk_metadata = [{k: v for k, v in chunk.items() if k != 'embedding'} for chunk in chunks]
        
        # Add to vector store
        self.vector_store.add_vectors(embeddings, chunk_metadata)
        
        # Save updated index
        self.vector_store.save_index(self.index_file, self.metadata_file)
        
        logger.info(f"Added case {case_id} to index ({len(chunks)} chunks)")
    
    def remove_judgment(self, case_id: str):
        """
        Remove a judgment from the index
        
        Args:
            case_id: Case identifier
        """
        if not self.is_loaded:
            logger.error("Index not loaded")
            return
        
        # Delete from vector store
        deleted_count = self.vector_store.delete_by_case_id(case_id)
        
        if deleted_count > 0:
            # Save updated index
            self.vector_store.save_index(self.index_file, self.metadata_file)
            logger.info(f"Removed case {case_id} from index")
        else:
            logger.warning(f"Case {case_id} not found in index")


def search_judgments(
    query: str,
    top_k: int = 5,
    index_file: str = "data/embeddings/index.faiss",
    metadata_file: str = "data/embeddings/metadata.pkl"
) -> List[Dict]:
    """
    Convenience function to search judgments
    
    Args:
        query: Search query
        top_k: Number of results
        index_file: Path to index file
        metadata_file: Path to metadata file
    
    Returns:
        List of search results
    """
    engine = SemanticSearchEngine(index_file, metadata_file)
    return engine.search(query, top_k)

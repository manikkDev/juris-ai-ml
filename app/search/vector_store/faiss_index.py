"""
FAISS Vector Store Module
Manages FAISS index for semantic search
"""
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from app.utils.logger import logger


class FAISSVectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(
        self,
        embedding_dim: int = 384,
        index_type: str = "L2"
    ):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ("L2" or "IP" for inner product)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata = []
        
        logger.info(f"Initialized FAISS vector store (dim={embedding_dim}, type={index_type})")
    
    def create_index(self, use_gpu: bool = False):
        """
        Create a new FAISS index
        
        Args:
            use_gpu: Use GPU for indexing (if available)
        """
        if self.index_type == "L2":
            # L2 distance (Euclidean)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IP":
            # Inner product (cosine similarity after normalization)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"Created FAISS index: {self.index_type}")
    
    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ):
        """
        Add vectors to the index
        
        Args:
            embeddings: Array of embeddings (n_vectors, embedding_dim)
            metadata: List of metadata dictionaries
        """
        if self.index is None:
            self.create_index()
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity if using IP
        if self.index_type == "IP":
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return np.array([]), np.array([])
        
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype('float32')
        
        # Normalize for cosine similarity if using IP
        if self.index_type == "IP":
            faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        return distances[0], indices[0]
    
    def search_with_metadata(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search and return results with metadata
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of result dictionaries with metadata and scores
        """
        distances, indices = self.search(query_embedding, top_k)
        
        results = []
        
        for dist, idx in zip(distances, indices):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                
                # Convert distance to similarity score (0-1)
                if self.index_type == "L2":
                    # For L2, smaller distance = more similar
                    # Convert to similarity score
                    score = 1 / (1 + dist)
                else:
                    # For IP (cosine), higher = more similar
                    score = float(dist)
                
                result['score'] = float(score)
                result['distance'] = float(dist)
                result['index'] = int(idx)
                
                results.append(result)
        
        return results
    
    def save_index(
        self,
        index_file: str = "data/embeddings/index.faiss",
        metadata_file: str = "data/embeddings/metadata.pkl"
    ):
        """
        Save index and metadata to disk
        
        Args:
            index_file: Path to save FAISS index
            metadata_file: Path to save metadata
        """
        if self.index is None:
            logger.warning("No index to save")
            return
        
        # Create directory
        Path(index_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_file)
        logger.info(f"Saved FAISS index to {index_file}")
        
        # Save metadata
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved metadata to {metadata_file}")
    
    def load_index(
        self,
        index_file: str = "data/embeddings/index.faiss",
        metadata_file: str = "data/embeddings/metadata.pkl"
    ):
        """
        Load index and metadata from disk
        
        Args:
            index_file: Path to FAISS index file
            metadata_file: Path to metadata file
        """
        if not Path(index_file).exists():
            logger.error(f"Index file not found: {index_file}")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        logger.info(f"Loaded FAISS index from {index_file}. Total vectors: {self.index.ntotal}")
        
        # Load metadata
        if Path(metadata_file).exists():
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded {len(self.metadata)} metadata records")
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
            self.metadata = []
        
        return True
    
    def get_stats(self) -> Dict:
        """
        Get index statistics
        
        Returns:
            Dictionary with index statistics
        """
        stats = {
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metadata_count': len(self.metadata),
            'is_trained': self.index.is_trained if self.index else False,
        }
        
        return stats
    
    def delete_by_case_id(self, case_id: str) -> int:
        """
        Delete all vectors for a specific case
        
        Args:
            case_id: Case ID to delete
        
        Returns:
            Number of vectors deleted
        """
        # Find indices to remove
        indices_to_remove = [
            i for i, meta in enumerate(self.metadata)
            if meta.get('case_id') == case_id
        ]
        
        if not indices_to_remove:
            logger.info(f"No vectors found for case {case_id}")
            return 0
        
        # FAISS doesn't support deletion, so we need to rebuild
        logger.info(f"Rebuilding index to remove {len(indices_to_remove)} vectors")
        
        # Get all vectors except those to remove
        all_vectors = []
        new_metadata = []
        
        for i in range(self.index.ntotal):
            if i not in indices_to_remove:
                # Reconstruct vector (this is slow for large indices)
                vector = self.index.reconstruct(i)
                all_vectors.append(vector)
                new_metadata.append(self.metadata[i])
        
        # Rebuild index
        self.create_index()
        
        if all_vectors:
            all_vectors = np.array(all_vectors)
            self.add_vectors(all_vectors, new_metadata)
        
        logger.info(f"Deleted {len(indices_to_remove)} vectors for case {case_id}")
        
        return len(indices_to_remove)
    
    def update_vectors(
        self,
        case_id: str,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ):
        """
        Update vectors for a specific case
        
        Args:
            case_id: Case ID to update
            embeddings: New embeddings
            metadata: New metadata
        """
        # Delete existing vectors
        self.delete_by_case_id(case_id)
        
        # Add new vectors
        self.add_vectors(embeddings, metadata)
        
        logger.info(f"Updated {len(embeddings)} vectors for case {case_id}")


def create_faiss_index(
    embeddings: np.ndarray,
    metadata: List[Dict],
    index_file: str = "data/embeddings/index.faiss",
    metadata_file: str = "data/embeddings/metadata.pkl"
) -> FAISSVectorStore:
    """
    Convenience function to create and save FAISS index
    
    Args:
        embeddings: Embeddings array
        metadata: Metadata list
        index_file: Path to save index
        metadata_file: Path to save metadata
    
    Returns:
        FAISSVectorStore instance
    """
    # Get embedding dimension
    embedding_dim = embeddings.shape[1]
    
    # Create vector store
    store = FAISSVectorStore(embedding_dim=embedding_dim, index_type="IP")
    
    # Add vectors
    store.add_vectors(embeddings, metadata)
    
    # Save to disk
    store.save_index(index_file, metadata_file)
    
    return store

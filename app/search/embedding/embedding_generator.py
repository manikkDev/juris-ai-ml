"""
Embedding Generator Module
Generates vector embeddings for legal judgment texts using Sentence Transformers
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm

from app.utils.logger import logger


class EmbeddingGenerator:
    """Generate embeddings for legal judgment texts"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize embedding generator
        
        Args:
            model_name: Sentence transformer model name
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
        
        Returns:
            List of text chunks
        """
        # Simple word-based chunking
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            
            if len(chunk.strip()) > 50:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks if chunks else [text]  # Return full text if no chunks
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
        
        Returns:
            Array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings
    
    def process_judgment(
        self,
        text: str,
        case_id: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Process a judgment text into chunks with embeddings
        
        Args:
            text: Judgment text
            case_id: Case identifier
            metadata: Additional metadata
        
        Returns:
            List of chunk dictionaries with embeddings
        """
        # Chunk the text
        chunks = self.chunk_text(text)
        
        logger.info(f"Processing case {case_id}: {len(chunks)} chunks")
        
        # Generate embeddings for all chunks
        embeddings = self.generate_embeddings_batch(chunks)
        
        # Create chunk records
        chunk_records = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            record = {
                'case_id': case_id,
                'chunk_id': f"{case_id}_chunk_{i}",
                'chunk_index': i,
                'text': chunk,
                'embedding': embedding,
                'court': metadata.get('court') if metadata else None,
                'judge': metadata.get('judge') if metadata else None,
                'date': metadata.get('date') if metadata else None,
                'case_type': metadata.get('case_type') if metadata else None,
            }
            
            chunk_records.append(record)
        
        return chunk_records
    
    def process_judgment_file(
        self,
        text_file: Path,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Process a judgment text file
        
        Args:
            text_file: Path to text file
            metadata: Additional metadata
        
        Returns:
            List of chunk records
        """
        # Read text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Use filename as case_id if not provided
        case_id = metadata.get('case_id') if metadata else text_file.stem
        
        return self.process_judgment(text, case_id, metadata)
    
    def process_dataset(
        self,
        text_files: List[Path],
        metadata_list: Optional[List[Dict]] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process multiple judgment files
        
        Args:
            text_files: List of text file paths
            metadata_list: List of metadata dictionaries
        
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        logger.info(f"Processing {len(text_files)} judgment files")
        
        all_chunks = []
        
        for i, text_file in enumerate(tqdm(text_files, desc="Processing judgments")):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            
            try:
                chunks = self.process_judgment_file(text_file, metadata)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {text_file.name}: {str(e)}")
                continue
        
        if not all_chunks:
            logger.error("No chunks generated from any file")
            return np.array([]), []
        
        # Extract embeddings and metadata
        embeddings = np.array([chunk['embedding'] for chunk in all_chunks])
        
        # Create metadata without embeddings (for storage)
        metadata_records = []
        for chunk in all_chunks:
            meta = {k: v for k, v in chunk.items() if k != 'embedding'}
            metadata_records.append(meta)
        
        logger.info(f"Generated {len(embeddings)} embeddings from {len(text_files)} files")
        
        return embeddings, metadata_records
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        embeddings_file: str = "data/embeddings/embeddings.npy",
        metadata_file: str = "data/embeddings/metadata.pkl"
    ):
        """
        Save embeddings and metadata to disk
        
        Args:
            embeddings: Embeddings array
            metadata: Metadata list
            embeddings_file: Path to save embeddings
            metadata_file: Path to save metadata
        """
        # Create directory
        Path(embeddings_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(embeddings_file, embeddings)
        logger.info(f"Saved {len(embeddings)} embeddings to {embeddings_file}")
        
        # Save metadata
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_file}")
    
    def load_embeddings(
        self,
        embeddings_file: str = "data/embeddings/embeddings.npy",
        metadata_file: str = "data/embeddings/metadata.pkl"
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Load embeddings and metadata from disk
        
        Args:
            embeddings_file: Path to embeddings file
            metadata_file: Path to metadata file
        
        Returns:
            Tuple of (embeddings, metadata)
        """
        embeddings = np.load(embeddings_file)
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {embeddings_file}")
        
        return embeddings, metadata


def generate_embeddings_from_texts(
    texts: List[str],
    case_ids: List[str],
    metadata_list: Optional[List[Dict]] = None
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Convenience function to generate embeddings from text list
    
    Args:
        texts: List of judgment texts
        case_ids: List of case IDs
        metadata_list: List of metadata dictionaries
    
    Returns:
        Tuple of (embeddings, metadata)
    """
    generator = EmbeddingGenerator()
    
    all_chunks = []
    
    for i, (text, case_id) in enumerate(zip(texts, case_ids)):
        metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
        chunks = generator.process_judgment(text, case_id, metadata)
        all_chunks.extend(chunks)
    
    embeddings = np.array([chunk['embedding'] for chunk in all_chunks])
    metadata_records = [{k: v for k, v in chunk.items() if k != 'embedding'} for chunk in all_chunks]
    
    return embeddings, metadata_records

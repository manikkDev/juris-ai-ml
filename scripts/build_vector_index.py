"""
Build Vector Index Script
Builds FAISS vector index from processed judgment texts
"""
import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.search.embedding.embedding_generator import EmbeddingGenerator
from app.search.vector_store.faiss_index import FAISSVectorStore
from app.pipeline.parsers.metadata_parser import MetadataParser
from app.utils.logger import logger


def build_index_from_texts(
    text_dir: str = "data/intermediate/text",
    index_file: str = "data/embeddings/index.faiss",
    metadata_file: str = "data/embeddings/metadata.pkl",
    max_files: int = None
):
    """
    Build vector index from text files
    
    Args:
        text_dir: Directory containing extracted text files
        index_file: Output path for FAISS index
        metadata_file: Output path for metadata
        max_files: Maximum number of files to process (None for all)
    """
    logger.info("=" * 60)
    logger.info("BUILDING VECTOR INDEX")
    logger.info("=" * 60)
    
    # Get text files
    text_path = Path(text_dir)
    
    if not text_path.exists():
        logger.error(f"Text directory not found: {text_dir}")
        return False
    
    text_files = list(text_path.glob("*.txt"))
    
    if not text_files:
        logger.error(f"No text files found in {text_dir}")
        return False
    
    if max_files:
        text_files = text_files[:max_files]
    
    logger.info(f"Found {len(text_files)} text files")
    
    # Initialize components
    embedding_generator = EmbeddingGenerator()
    metadata_parser = MetadataParser()
    
    # Process files and extract metadata
    logger.info("Processing text files and extracting metadata...")
    
    metadata_list = []
    
    for text_file in text_files:
        try:
            # Parse metadata from text
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            parsed_metadata = metadata_parser.parse_complete_metadata(text, text_file.name)
            
            # Create metadata dict for embedding
            meta = {
                'case_id': parsed_metadata.get('case_number') or text_file.stem,
                'court': parsed_metadata.get('court'),
                'judge': ', '.join(parsed_metadata.get('judges', [])) if parsed_metadata.get('judges') else None,
                'date': parsed_metadata.get('judgment_date'),
                'case_type': parsed_metadata.get('case_type'),
            }
            
            metadata_list.append(meta)
        
        except Exception as e:
            logger.error(f"Error processing {text_file.name}: {str(e)}")
            # Add minimal metadata
            metadata_list.append({
                'case_id': text_file.stem,
                'court': None,
                'judge': None,
                'date': None,
                'case_type': None,
            })
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings, chunk_metadata = embedding_generator.process_dataset(
        text_files,
        metadata_list
    )
    
    if len(embeddings) == 0:
        logger.error("No embeddings generated")
        return False
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Create FAISS index
    logger.info("Creating FAISS index...")
    
    embedding_dim = embeddings.shape[1]
    vector_store = FAISSVectorStore(embedding_dim=embedding_dim, index_type="IP")
    vector_store.add_vectors(embeddings, chunk_metadata)
    
    # Save index
    logger.info("Saving index to disk...")
    vector_store.save_index(index_file, metadata_file)
    
    # Print statistics
    stats = vector_store.get_stats()
    
    logger.info("=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total vectors: {stats['total_vectors']}")
    logger.info(f"Embedding dimension: {stats['embedding_dim']}")
    logger.info(f"Index type: {stats['index_type']}")
    logger.info(f"Index file: {index_file}")
    logger.info(f"Metadata file: {metadata_file}")
    logger.info("=" * 60)
    
    return True


def build_index_from_dataset(
    dataset_file: str = "data/processed/dataset.csv",
    text_dir: str = "data/intermediate/text",
    index_file: str = "data/embeddings/index.faiss",
    metadata_file: str = "data/embeddings/metadata.pkl"
):
    """
    Build index using dataset metadata
    
    Args:
        dataset_file: Path to processed dataset CSV
        text_dir: Directory with text files
        index_file: Output index path
        metadata_file: Output metadata path
    """
    import pandas as pd
    
    logger.info("Building index from dataset...")
    
    # Load dataset
    if not Path(dataset_file).exists():
        logger.warning(f"Dataset file not found: {dataset_file}")
        logger.info("Falling back to text-only indexing")
        return build_index_from_texts(text_dir, index_file, metadata_file)
    
    df = pd.read_csv(dataset_file)
    logger.info(f"Loaded dataset with {len(df)} cases")
    
    # Get text files
    text_path = Path(text_dir)
    text_files = list(text_path.glob("*.txt"))
    
    # Match text files with dataset
    # This is a simple approach - in production, you'd have better linking
    text_files = text_files[:len(df)]
    
    # Extract metadata from dataset
    metadata_list = []
    
    for _, row in df.iterrows():
        meta = {
            'case_id': row.get('source_file', '').replace('.txt', ''),
            'court': row.get('court'),
            'judge': None,  # Not in dataset
            'date': None,   # Not in dataset
            'case_type': row.get('case_type'),
        }
        metadata_list.append(meta)
    
    # Generate embeddings
    embedding_generator = EmbeddingGenerator()
    
    embeddings, chunk_metadata = embedding_generator.process_dataset(
        text_files[:len(metadata_list)],
        metadata_list
    )
    
    if len(embeddings) == 0:
        logger.error("No embeddings generated")
        return False
    
    # Create and save index
    embedding_dim = embeddings.shape[1]
    vector_store = FAISSVectorStore(embedding_dim=embedding_dim, index_type="IP")
    vector_store.add_vectors(embeddings, chunk_metadata)
    vector_store.save_index(index_file, metadata_file)
    
    logger.info(f"Index built successfully with {len(embeddings)} vectors")
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Build FAISS vector index for semantic search")
    
    parser.add_argument('--text-dir', type=str, default='data/intermediate/text',
                       help='Directory containing text files')
    parser.add_argument('--dataset', type=str, default='data/processed/dataset.csv',
                       help='Path to dataset CSV (optional)')
    parser.add_argument('--index-file', type=str, default='data/embeddings/index.faiss',
                       help='Output path for FAISS index')
    parser.add_argument('--metadata-file', type=str, default='data/embeddings/metadata.pkl',
                       help='Output path for metadata')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process')
    parser.add_argument('--use-dataset', action='store_true',
                       help='Use dataset metadata if available')
    
    args = parser.parse_args()
    
    # Build index
    if args.use_dataset:
        success = build_index_from_dataset(
            dataset_file=args.dataset,
            text_dir=args.text_dir,
            index_file=args.index_file,
            metadata_file=args.metadata_file
        )
    else:
        success = build_index_from_texts(
            text_dir=args.text_dir,
            index_file=args.index_file,
            metadata_file=args.metadata_file,
            max_files=args.max_files
        )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

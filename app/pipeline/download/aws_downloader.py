"""
AWS Downloader for Indian High Court Judgments
Downloads PDFs from AWS Open Data registry
"""
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm
import json
from datetime import datetime

from app.utils.logger import logger


class IndianHighCourtDownloader:
    """
    Download Indian High Court judgments from AWS Open Data
    
    Dataset: Indian High Court Judgments
    Registry: AWS Open Data
    """
    
    def __init__(self, output_dir: str = "data/raw/pdfs"):
        """
        Initialize downloader
        
        Args:
            output_dir: Directory to save downloaded PDFs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure boto3 for anonymous access to public bucket
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED)
        )
        
        # Indian High Court dataset bucket (example - adjust based on actual bucket)
        # Note: This is a placeholder - actual bucket name may vary
        self.bucket_name = "indian-judiciary-data"
        
        # Metadata tracking
        self.metadata_file = self.output_dir / "download_metadata.json"
        self.downloaded_files = self._load_metadata()
        
        logger.info(f"Initialized AWS downloader. Output: {self.output_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load download metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"files": [], "last_updated": None}
    
    def _save_metadata(self):
        """Save download metadata"""
        self.downloaded_files["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.downloaded_files, f, indent=2)
    
    def list_available_files(
        self,
        prefix: str = "",
        year: Optional[int] = None,
        court: Optional[str] = None,
        max_files: int = 100
    ) -> List[str]:
        """
        List available files in the S3 bucket
        
        Args:
            prefix: S3 prefix to filter files
            year: Filter by year
            court: Filter by court name
            max_files: Maximum number of files to list
        
        Returns:
            List of S3 keys
        """
        try:
            logger.info(f"Listing files from bucket: {self.bucket_name}")
            
            # Build prefix based on filters
            if year:
                prefix = f"{prefix}{year}/" if prefix else f"{year}/"
            if court:
                prefix = f"{prefix}{court}/" if prefix else f"{court}/"
            
            # List objects
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Filter for PDF files
                        if key.endswith('.pdf'):
                            files.append(key)
                            if len(files) >= max_files:
                                break
                
                if len(files) >= max_files:
                    break
            
            logger.info(f"Found {len(files)} PDF files")
            return files
        
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            # Return empty list if bucket doesn't exist (for demo purposes)
            logger.warning("Using demo mode - no actual AWS download")
            return []
    
    def download_file(self, s3_key: str, local_path: Optional[Path] = None) -> Optional[Path]:
        """
        Download a single file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local path to save file (optional)
        
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            if local_path is None:
                # Create local path from S3 key
                filename = Path(s3_key).name
                local_path = self.output_dir / filename
            
            # Skip if already downloaded
            if str(local_path) in self.downloaded_files.get("files", []):
                logger.info(f"File already downloaded: {local_path.name}")
                return local_path
            
            # Download file
            logger.info(f"Downloading: {s3_key} -> {local_path}")
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                str(local_path)
            )
            
            # Update metadata
            self.downloaded_files.setdefault("files", []).append(str(local_path))
            self._save_metadata()
            
            return local_path
        
        except Exception as e:
            logger.error(f"Error downloading {s3_key}: {str(e)}")
            return None
    
    def download_batch(
        self,
        year: Optional[int] = None,
        court: Optional[str] = None,
        case_type: Optional[str] = None,
        max_files: int = 100,
        resume: bool = True
    ) -> List[Path]:
        """
        Download a batch of files with filters
        
        Args:
            year: Filter by year
            court: Filter by court name
            case_type: Filter by case type
            max_files: Maximum number of files to download
            resume: Resume interrupted downloads
        
        Returns:
            List of downloaded file paths
        """
        logger.info(f"Starting batch download (max: {max_files} files)")
        logger.info(f"Filters - Year: {year}, Court: {court}, Case Type: {case_type}")
        
        # List available files
        s3_keys = self.list_available_files(
            year=year,
            court=court,
            max_files=max_files
        )
        
        if not s3_keys:
            logger.warning("No files found matching criteria")
            return []
        
        # Download files with progress bar
        downloaded_paths = []
        
        with tqdm(total=len(s3_keys), desc="Downloading PDFs") as pbar:
            for s3_key in s3_keys:
                # Apply case type filter if specified
                if case_type and case_type.lower() not in s3_key.lower():
                    pbar.update(1)
                    continue
                
                # Download file
                local_path = self.download_file(s3_key)
                
                if local_path:
                    downloaded_paths.append(local_path)
                
                pbar.update(1)
        
        logger.info(f"Downloaded {len(downloaded_paths)} files successfully")
        return downloaded_paths
    
    def get_download_stats(self) -> Dict:
        """
        Get download statistics
        
        Returns:
            Dictionary with download stats
        """
        return {
            "total_files": len(self.downloaded_files.get("files", [])),
            "last_updated": self.downloaded_files.get("last_updated"),
            "output_directory": str(self.output_dir)
        }


def download_sample_data(output_dir: str = "data/raw/pdfs", num_files: int = 10) -> List[Path]:
    """
    Download sample data for testing
    
    Args:
        output_dir: Output directory
        num_files: Number of sample files
    
    Returns:
        List of downloaded paths
    """
    downloader = IndianHighCourtDownloader(output_dir)
    
    # Try to download from AWS, fallback to demo mode
    files = downloader.download_batch(max_files=num_files)
    
    # If no files downloaded (demo mode), create sample PDFs
    if not files:
        logger.info("Creating sample PDF files for demo purposes")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder files
        for i in range(min(num_files, 5)):
            sample_file = output_path / f"sample_judgment_{i+1}.pdf"
            if not sample_file.exists():
                sample_file.write_text(f"Sample judgment document {i+1}")
                files.append(sample_file)
                logger.info(f"Created sample file: {sample_file.name}")
    
    return files

"""
PDF Text Extraction Module
Extracts text from legal judgment PDFs using pdfminer.six
"""
from pathlib import Path
from typing import Optional, Dict
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
import io

from app.utils.logger import logger


class PDFTextExtractor:
    """Extract text from PDF files"""
    
    def __init__(self, output_dir: str = "data/intermediate/text"):
        """
        Initialize PDF text extractor
        
        Args:
            output_dir: Directory to save extracted text
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized PDF text extractor. Output: {self.output_dir}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Extracted text or None if extraction failed
        """
        try:
            logger.info(f"Extracting text from: {pdf_path.name}")
            
            # Extract text using pdfminer.six
            text = extract_text(str(pdf_path))
            
            if not text or len(text.strip()) < 100:
                logger.warning(f"Extracted text too short or empty: {pdf_path.name}")
                return None
            
            logger.info(f"Successfully extracted {len(text)} characters from {pdf_path.name}")
            return text
        
        except PDFSyntaxError as e:
            logger.error(f"PDF syntax error in {pdf_path.name}: {str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path.name}: {str(e)}")
            return None
    
    def save_extracted_text(self, text: str, output_filename: str) -> Path:
        """
        Save extracted text to file
        
        Args:
            text: Extracted text
            output_filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Saved extracted text to: {output_path.name}")
        return output_path
    
    def extract_and_save(self, pdf_path: Path) -> Optional[Dict]:
        """
        Extract text from PDF and save to file
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with extraction results
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if text is None:
            return None
        
        # Generate output filename
        output_filename = pdf_path.stem + ".txt"
        
        # Save text
        text_path = self.save_extracted_text(text, output_filename)
        
        return {
            "pdf_path": str(pdf_path),
            "text_path": str(text_path),
            "text_length": len(text),
            "success": True
        }
    
    def extract_batch(self, pdf_paths: list) -> Dict:
        """
        Extract text from multiple PDFs
        
        Args:
            pdf_paths: List of PDF file paths
        
        Returns:
            Dictionary with batch extraction results
        """
        logger.info(f"Starting batch extraction for {len(pdf_paths)} files")
        
        results = {
            "successful": [],
            "failed": [],
            "total": len(pdf_paths)
        }
        
        for pdf_path in pdf_paths:
            result = self.extract_and_save(pdf_path)
            
            if result:
                results["successful"].append(result)
            else:
                results["failed"].append(str(pdf_path))
        
        logger.info(f"Batch extraction complete. Success: {len(results['successful'])}, Failed: {len(results['failed'])}")
        
        return results
    
    def get_text_preview(self, text: str, max_chars: int = 500) -> str:
        """
        Get preview of extracted text
        
        Args:
            text: Full text
            max_chars: Maximum characters to return
        
        Returns:
            Text preview
        """
        if len(text) <= max_chars:
            return text
        
        return text[:max_chars] + "..."


def extract_text_from_file(pdf_path: str) -> Optional[str]:
    """
    Convenience function to extract text from a single PDF
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text or None
    """
    extractor = PDFTextExtractor()
    return extractor.extract_text_from_pdf(Path(pdf_path))

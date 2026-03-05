"""
OCR Fallback Module
Uses Tesseract OCR when PDF text extraction fails
"""
from pathlib import Path
from typing import Optional, List
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import io

from app.utils.logger import logger


class OCRExtractor:
    """OCR-based text extraction for scanned PDFs"""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize OCR extractor
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        logger.info("Initialized OCR extractor")
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from an image using OCR
        
        Args:
            image: PIL Image object
        
        Returns:
            Extracted text
        """
        try:
            # Use Tesseract to extract text
            text = pytesseract.image_to_string(image, lang='eng')
            return text
        
        except Exception as e:
            logger.error(f"OCR extraction error: {str(e)}")
            return ""
    
    def pdf_to_images(self, pdf_path: Path, dpi: int = 200) -> List[Image.Image]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for image conversion (higher = better quality but slower)
        
        Returns:
            List of PIL Image objects
        """
        try:
            logger.info(f"Converting PDF to images: {pdf_path.name} (DPI: {dpi})")
            
            # Convert PDF to images
            images = convert_from_path(str(pdf_path), dpi=dpi)
            
            logger.info(f"Converted {len(images)} pages to images")
            return images
        
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            return []
    
    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        max_pages: Optional[int] = None,
        dpi: int = 200
    ) -> Optional[str]:
        """
        Extract text from PDF using OCR
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (None for all)
            dpi: DPI for image conversion
        
        Returns:
            Extracted text or None if failed
        """
        try:
            logger.info(f"Starting OCR extraction for: {pdf_path.name}")
            
            # Convert PDF to images
            images = self.pdf_to_images(pdf_path, dpi=dpi)
            
            if not images:
                logger.error(f"No images extracted from PDF: {pdf_path.name}")
                return None
            
            # Limit pages if specified
            if max_pages:
                images = images[:max_pages]
                logger.info(f"Processing first {max_pages} pages")
            
            # Extract text from each page
            all_text = []
            
            for i, image in enumerate(images, 1):
                logger.info(f"Processing page {i}/{len(images)}")
                
                page_text = self.extract_text_from_image(image)
                
                if page_text:
                    all_text.append(f"--- Page {i} ---\n{page_text}")
            
            # Combine all text
            full_text = "\n\n".join(all_text)
            
            if not full_text or len(full_text.strip()) < 100:
                logger.warning(f"OCR extracted text too short: {pdf_path.name}")
                return None
            
            logger.info(f"OCR extraction successful. Extracted {len(full_text)} characters")
            return full_text
        
        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path.name}: {str(e)}")
            return None
    
    def extract_with_confidence(self, pdf_path: Path) -> Optional[dict]:
        """
        Extract text with confidence scores
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with text and confidence data
        """
        try:
            images = self.pdf_to_images(pdf_path, dpi=200)
            
            if not images:
                return None
            
            results = []
            
            for i, image in enumerate(images[:5], 1):  # Limit to first 5 pages
                # Get detailed OCR data
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                # Extract text with confidence
                page_text = []
                confidences = []
                
                for j, text in enumerate(data['text']):
                    if text.strip():
                        page_text.append(text)
                        confidences.append(data['conf'][j])
                
                results.append({
                    'page': i,
                    'text': ' '.join(page_text),
                    'avg_confidence': sum(confidences) / len(confidences) if confidences else 0
                })
            
            # Combine all pages
            full_text = '\n\n'.join([r['text'] for r in results])
            avg_confidence = sum([r['avg_confidence'] for r in results]) / len(results)
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'pages_processed': len(results)
            }
        
        except Exception as e:
            logger.error(f"Error in confidence extraction: {str(e)}")
            return None


def extract_with_ocr_fallback(pdf_path: Path, use_ocr: bool = False) -> Optional[str]:
    """
    Extract text with OCR fallback
    
    Args:
        pdf_path: Path to PDF file
        use_ocr: Force OCR extraction
    
    Returns:
        Extracted text or None
    """
    if use_ocr:
        logger.info("Using OCR extraction")
        ocr = OCRExtractor()
        return ocr.extract_text_from_pdf(pdf_path, max_pages=10)
    
    # Try regular extraction first
    from app.pipeline.extract.pdf_text_extractor import extract_text_from_file
    
    text = extract_text_from_file(str(pdf_path))
    
    if text and len(text.strip()) >= 100:
        return text
    
    # Fallback to OCR
    logger.info("Regular extraction failed, falling back to OCR")
    ocr = OCRExtractor()
    return ocr.extract_text_from_pdf(pdf_path, max_pages=10)

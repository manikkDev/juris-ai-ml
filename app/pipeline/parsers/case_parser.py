"""
Case Parser Module
Extracts case metadata from judgment text using regex patterns
"""
import re
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path

from app.utils.logger import logger


class CaseParser:
    """Parse case metadata from judgment text"""
    
    def __init__(self):
        """Initialize case parser with regex patterns"""
        
        # Case number patterns
        self.case_number_patterns = [
            r'(?:Case|Petition|Appeal|Writ|Criminal|Civil)\s+(?:No\.|Number|#)\s*[:.]?\s*(\d+(?:/\d+)?(?:/\d+)?)',
            r'(?:W\.P\.|C\.A\.|Crl\.A\.|C\.P\.)\s*(?:No\.)?\s*(\d+(?:/\d+)?)',
            r'(?:Case|Matter)\s+(?:bearing\s+)?(?:No\.)?\s*[:.]?\s*(\d+/\d+)',
        ]
        
        # Judge name patterns
        self.judge_patterns = [
            r'(?:Hon\'?ble\s+)?(?:Mr\.|Ms\.|Mrs\.)?\s*Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:CORAM|Before)\s*[:.]?\s*(?:Hon\'?ble\s+)?(?:Mr\.|Ms\.)?\s*Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:Presided\s+by|Delivered\s+by)\s+(?:Hon\'?ble\s+)?Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        # Court name patterns
        self.court_patterns = [
            r'(?:IN\s+THE\s+)?([A-Z][A-Z\s]+HIGH\s+COURT)',
            r'(HIGH\s+COURT\s+OF\s+[A-Z][A-Z\s]+)',
            r'(?:SUPREME\s+COURT\s+OF\s+INDIA)',
        ]
        
        # Date patterns
        self.date_patterns = [
            r'(?:Date|Dated|Judgment\s+dated)\s*[:.]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:Date|Dated)\s*[:.]?\s*(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+,?\s+\d{4})',
            r'(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',
        ]
        
        # Case type patterns
        self.case_type_patterns = [
            r'(Criminal\s+Appeal)',
            r'(Civil\s+Appeal)',
            r'(Writ\s+Petition)',
            r'(Special\s+Leave\s+Petition)',
            r'(Constitutional\s+Matter)',
            r'(Family\s+Court)',
            r'(Commercial\s+Suit)',
            r'(Tax\s+Appeal)',
        ]
        
        logger.info("Initialized case parser")
    
    def extract_case_number(self, text: str) -> Optional[str]:
        """
        Extract case number from text
        
        Args:
            text: Judgment text
        
        Returns:
            Case number or None
        """
        for pattern in self.case_number_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_judge_names(self, text: str) -> List[str]:
        """
        Extract judge names from text
        
        Args:
            text: Judgment text
        
        Returns:
            List of judge names
        """
        judges = []
        
        for pattern in self.judge_patterns:
            matches = re.finditer(pattern, text[:2000])  # Search in first 2000 chars
            for match in matches:
                judge_name = match.group(1).strip()
                if judge_name and judge_name not in judges:
                    judges.append(judge_name)
        
        return judges
    
    def extract_court_name(self, text: str) -> Optional[str]:
        """
        Extract court name from text
        
        Args:
            text: Judgment text
        
        Returns:
            Court name or None
        """
        for pattern in self.court_patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return None
    
    def extract_judgment_date(self, text: str) -> Optional[str]:
        """
        Extract judgment date from text
        
        Args:
            text: Judgment text
        
        Returns:
            Date string or None
        """
        for pattern in self.date_patterns:
            match = re.search(pattern, text[:2000])
            if match:
                date_str = match.group(1).strip()
                # Try to parse and standardize date
                parsed_date = self._parse_date(date_str)
                if parsed_date:
                    return parsed_date
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """
        Parse and standardize date string
        
        Args:
            date_str: Raw date string
        
        Returns:
            Standardized date (YYYY-MM-DD) or None
        """
        # Common date formats
        formats = [
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%d-%m-%y',
            '%d/%m/%y',
            '%d %B %Y',
            '%d %b %Y',
            '%d %B, %Y',
        ]
        
        # Remove ordinal suffixes
        date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def extract_case_type(self, text: str) -> Optional[str]:
        """
        Extract case type from text
        
        Args:
            text: Judgment text
        
        Returns:
            Case type or None
        """
        for pattern in self.case_type_patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Default classification based on keywords
        text_lower = text[:2000].lower()
        
        if 'criminal' in text_lower or 'accused' in text_lower:
            return 'Criminal'
        elif 'civil' in text_lower or 'plaintiff' in text_lower:
            return 'Civil'
        elif 'family' in text_lower or 'divorce' in text_lower:
            return 'Family'
        elif 'commercial' in text_lower or 'contract' in text_lower:
            return 'Commercial'
        elif 'tax' in text_lower or 'revenue' in text_lower:
            return 'Tax'
        elif 'constitutional' in text_lower or 'fundamental right' in text_lower:
            return 'Constitutional'
        
        return 'Other'
    
    def parse_case(self, text: str, filename: Optional[str] = None) -> Dict:
        """
        Parse all case metadata from text
        
        Args:
            text: Judgment text
            filename: Source filename (optional)
        
        Returns:
            Dictionary with parsed metadata
        """
        logger.info(f"Parsing case metadata{' from ' + filename if filename else ''}")
        
        metadata = {
            'case_number': self.extract_case_number(text),
            'judges': self.extract_judge_names(text),
            'court': self.extract_court_name(text),
            'judgment_date': self.extract_judgment_date(text),
            'case_type': self.extract_case_type(text),
            'source_file': filename,
            'text_length': len(text)
        }
        
        # Log extracted metadata
        logger.info(f"Extracted metadata: Case #{metadata['case_number']}, "
                   f"Court: {metadata['court']}, Type: {metadata['case_type']}")
        
        return metadata
    
    def parse_from_file(self, text_file: Path) -> Dict:
        """
        Parse case metadata from text file
        
        Args:
            text_file: Path to text file
        
        Returns:
            Dictionary with parsed metadata
        """
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.parse_case(text, filename=text_file.name)


def parse_case_metadata(text: str) -> Dict:
    """
    Convenience function to parse case metadata
    
    Args:
        text: Judgment text
    
    Returns:
        Dictionary with metadata
    """
    parser = CaseParser()
    return parser.parse_case(text)

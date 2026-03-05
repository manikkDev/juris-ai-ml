"""
Metadata Parser Module
Combines case parser and hearing event parser for complete metadata extraction
"""
from typing import Dict, Optional
from pathlib import Path

from app.pipeline.parsers.case_parser import CaseParser
from app.pipeline.parsers.hearing_event_parser import HearingEventParser
from app.utils.logger import logger


class MetadataParser:
    """Complete metadata parser combining case and hearing information"""
    
    def __init__(self):
        """Initialize metadata parser"""
        self.case_parser = CaseParser()
        self.hearing_parser = HearingEventParser()
        
        logger.info("Initialized metadata parser")
    
    def parse_complete_metadata(self, text: str, source_file: Optional[str] = None) -> Dict:
        """
        Parse complete metadata from judgment text
        
        Args:
            text: Judgment text
            source_file: Source filename (optional)
        
        Returns:
            Dictionary with complete metadata
        """
        logger.info(f"Parsing complete metadata{' from ' + source_file if source_file else ''}")
        
        # Parse case metadata
        case_metadata = self.case_parser.parse_case(text, filename=source_file)
        
        # Parse timeline
        timeline = self.hearing_parser.parse_timeline(
            text,
            case_id=case_metadata.get('case_number')
        )
        
        # Combine metadata
        complete_metadata = {
            # Case information
            'case_number': case_metadata.get('case_number'),
            'case_type': case_metadata.get('case_type'),
            'court': case_metadata.get('court'),
            'judges': case_metadata.get('judges', []),
            'judgment_date': case_metadata.get('judgment_date'),
            
            # Timeline information
            'events': timeline.get('events', []),
            'total_events': timeline.get('total_events', 0),
            'adjournment_count': timeline.get('adjournment_count', 0),
            'hearing_count': timeline.get('hearing_count', 0),
            'case_age_days': timeline.get('case_age_days'),
            'avg_hearing_gap': timeline.get('avg_hearing_gap'),
            
            # Source information
            'source_file': source_file,
            'text_length': len(text),
            
            # Data quality indicators
            'has_case_number': case_metadata.get('case_number') is not None,
            'has_dates': timeline.get('total_events', 0) > 0,
            'has_judge': len(case_metadata.get('judges', [])) > 0,
        }
        
        # Calculate data completeness score
        completeness_score = self._calculate_completeness(complete_metadata)
        complete_metadata['completeness_score'] = completeness_score
        
        logger.info(f"Metadata parsing complete. Completeness: {completeness_score:.2f}")
        
        return complete_metadata
    
    def _calculate_completeness(self, metadata: Dict) -> float:
        """
        Calculate metadata completeness score (0-1)
        
        Args:
            metadata: Metadata dictionary
        
        Returns:
            Completeness score
        """
        checks = [
            metadata.get('case_number') is not None,
            metadata.get('case_type') is not None,
            metadata.get('court') is not None,
            len(metadata.get('judges', [])) > 0,
            metadata.get('judgment_date') is not None,
            metadata.get('total_events', 0) > 0,
            metadata.get('adjournment_count', 0) >= 0,
            metadata.get('hearing_count', 0) > 0,
            metadata.get('case_age_days') is not None,
        ]
        
        return sum(checks) / len(checks)
    
    def parse_from_file(self, text_file: Path) -> Dict:
        """
        Parse metadata from text file
        
        Args:
            text_file: Path to text file
        
        Returns:
            Complete metadata dictionary
        """
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.parse_complete_metadata(text, source_file=text_file.name)
    
    def validate_metadata(self, metadata: Dict) -> Dict:
        """
        Validate and clean metadata
        
        Args:
            metadata: Raw metadata dictionary
        
        Returns:
            Validated metadata with quality flags
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check required fields
        if not metadata.get('case_number'):
            validation['warnings'].append("Missing case number")
        
        if not metadata.get('case_type'):
            validation['warnings'].append("Missing case type")
        
        if metadata.get('completeness_score', 0) < 0.5:
            validation['warnings'].append("Low completeness score")
        
        if metadata.get('text_length', 0) < 500:
            validation['errors'].append("Text too short")
            validation['is_valid'] = False
        
        if metadata.get('total_events', 0) == 0:
            validation['warnings'].append("No timeline events found")
        
        metadata['validation'] = validation
        
        return metadata


def parse_judgment_metadata(text: str, source_file: Optional[str] = None) -> Dict:
    """
    Convenience function to parse judgment metadata
    
    Args:
        text: Judgment text
        source_file: Source filename
    
    Returns:
        Complete metadata dictionary
    """
    parser = MetadataParser()
    metadata = parser.parse_complete_metadata(text, source_file)
    return parser.validate_metadata(metadata)

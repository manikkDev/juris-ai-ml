"""
Hearing Event Parser Module
Extracts hearing events and timeline from judgment text
"""
import re
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

from app.utils.logger import logger


class HearingEventParser:
    """Parse hearing events and case timeline from judgment text"""
    
    def __init__(self):
        """Initialize hearing event parser with patterns"""
        
        # Event type patterns
        self.event_patterns = {
            'filed': [
                r'(?:petition|case|matter|appeal)\s+(?:was\s+)?filed\s+on\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'filed\s+on\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',
            ],
            'hearing': [
                r'hearing\s+(?:held\s+)?on\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'heard\s+on\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',
                r'matter\s+(?:was\s+)?heard\s+on\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            ],
            'adjourned': [
                r'adjourned\s+(?:to|till|until)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'adjourned\s+(?:to|till)\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',
                r'matter\s+(?:is\s+)?adjourned\s+to\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'case\s+adjourned',
            ],
            'listed': [
                r'(?:matter|case)\s+(?:is\s+)?listed\s+(?:for|on)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'list\s+(?:the\s+matter\s+)?on\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',
            ],
            'next_hearing': [
                r'next\s+(?:date\s+of\s+)?hearing\s+(?:on|:)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'next\s+hearing\s+(?:on|:)\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',
            ],
            'judgment': [
                r'judgment\s+(?:delivered|pronounced)\s+on\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'judgment\s+dated\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',
            ],
            'order': [
                r'order\s+(?:passed|dated)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'ordered\s+on\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',
            ]
        }
        
        # Adjournment reason patterns
        self.adjournment_reasons = [
            r'adjourned\s+(?:due\s+to|on\s+account\s+of|for)\s+([^.]+)',
            r'reason\s+for\s+adjournment[:\s]+([^.]+)',
        ]
        
        logger.info("Initialized hearing event parser")
    
    def extract_events(self, text: str) -> List[Dict]:
        """
        Extract all hearing events from text
        
        Args:
            text: Judgment text
        
        Returns:
            List of event dictionaries
        """
        events = []
        
        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Extract date if present in match
                    date_str = None
                    if match.groups():
                        date_str = match.group(1)
                    
                    # Parse date
                    parsed_date = self._parse_date(date_str) if date_str else None
                    
                    event = {
                        'type': event_type,
                        'date': parsed_date,
                        'raw_date': date_str,
                        'context': self._get_context(text, match.start(), match.end())
                    }
                    
                    # Add adjournment reason if applicable
                    if event_type == 'adjourned':
                        reason = self._extract_adjournment_reason(
                            text[max(0, match.start()-100):min(len(text), match.end()+200)]
                        )
                        if reason:
                            event['reason'] = reason
                    
                    events.append(event)
        
        # Sort events by date
        events_with_dates = [e for e in events if e['date']]
        events_with_dates.sort(key=lambda x: x['date'])
        
        logger.info(f"Extracted {len(events_with_dates)} events with dates")
        
        return events_with_dates
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """
        Parse date string to standard format
        
        Args:
            date_str: Raw date string
        
        Returns:
            Standardized date (YYYY-MM-DD) or None
        """
        if not date_str:
            return None
        
        formats = [
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%d-%m-%y',
            '%d/%m/%y',
            '%d %B %Y',
            '%d %b %Y',
            '%d %B, %Y',
        ]
        
        # Clean date string
        date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """
        Get context around a match
        
        Args:
            text: Full text
            start: Match start position
            end: Match end position
            window: Context window size
        
        Returns:
            Context string
        """
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        return text[context_start:context_end].strip()
    
    def _extract_adjournment_reason(self, context: str) -> Optional[str]:
        """
        Extract reason for adjournment from context
        
        Args:
            context: Text context around adjournment mention
        
        Returns:
            Adjournment reason or None
        """
        for pattern in self.adjournment_reasons:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                reason = match.group(1).strip()
                # Clean up reason
                reason = re.sub(r'\s+', ' ', reason)
                return reason[:200]  # Limit length
        
        return None
    
    def count_adjournments(self, events: List[Dict]) -> int:
        """
        Count total adjournments
        
        Args:
            events: List of events
        
        Returns:
            Number of adjournments
        """
        return sum(1 for e in events if e['type'] == 'adjourned')
    
    def count_hearings(self, events: List[Dict]) -> int:
        """
        Count total hearings
        
        Args:
            events: List of events
        
        Returns:
            Number of hearings
        """
        hearing_types = {'hearing', 'listed', 'next_hearing'}
        return sum(1 for e in events if e['type'] in hearing_types)
    
    def calculate_case_age(self, events: List[Dict]) -> Optional[int]:
        """
        Calculate case age in days
        
        Args:
            events: List of events
        
        Returns:
            Case age in days or None
        """
        # Find filing date
        filing_events = [e for e in events if e['type'] == 'filed' and e['date']]
        
        if not filing_events:
            return None
        
        filing_date = datetime.strptime(filing_events[0]['date'], '%Y-%m-%d')
        
        # Find last event date
        dated_events = [e for e in events if e['date']]
        
        if not dated_events:
            return None
        
        last_date = datetime.strptime(dated_events[-1]['date'], '%Y-%m-%d')
        
        # Calculate age
        age_days = (last_date - filing_date).days
        
        return max(0, age_days)
    
    def calculate_days_between_hearings(self, events: List[Dict]) -> List[int]:
        """
        Calculate days between consecutive hearings
        
        Args:
            events: List of events
        
        Returns:
            List of days between hearings
        """
        hearing_types = {'hearing', 'listed', 'next_hearing'}
        hearing_events = [e for e in events if e['type'] in hearing_types and e['date']]
        
        if len(hearing_events) < 2:
            return []
        
        # Sort by date
        hearing_events.sort(key=lambda x: x['date'])
        
        gaps = []
        for i in range(1, len(hearing_events)):
            date1 = datetime.strptime(hearing_events[i-1]['date'], '%Y-%m-%d')
            date2 = datetime.strptime(hearing_events[i]['date'], '%Y-%m-%d')
            
            gap = (date2 - date1).days
            gaps.append(gap)
        
        return gaps
    
    def parse_timeline(self, text: str, case_id: Optional[str] = None) -> Dict:
        """
        Parse complete case timeline
        
        Args:
            text: Judgment text
            case_id: Case identifier (optional)
        
        Returns:
            Dictionary with timeline data
        """
        logger.info(f"Parsing timeline{' for case ' + case_id if case_id else ''}")
        
        # Extract events
        events = self.extract_events(text)
        
        # Calculate metrics
        timeline = {
            'case_id': case_id,
            'events': events,
            'total_events': len(events),
            'adjournment_count': self.count_adjournments(events),
            'hearing_count': self.count_hearings(events),
            'case_age_days': self.calculate_case_age(events),
            'hearing_gaps': self.calculate_days_between_hearings(events),
        }
        
        # Calculate average hearing gap
        if timeline['hearing_gaps']:
            timeline['avg_hearing_gap'] = sum(timeline['hearing_gaps']) / len(timeline['hearing_gaps'])
        else:
            timeline['avg_hearing_gap'] = None
        
        logger.info(f"Timeline parsed: {timeline['total_events']} events, "
                   f"{timeline['adjournment_count']} adjournments, "
                   f"{timeline['hearing_count']} hearings")
        
        return timeline


def parse_hearing_events(text: str) -> Dict:
    """
    Convenience function to parse hearing events
    
    Args:
        text: Judgment text
    
    Returns:
        Timeline dictionary
    """
    parser = HearingEventParser()
    return parser.parse_timeline(text)

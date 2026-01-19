"""
Working Memory for agent's current task.

Stores intermediate results, tool outputs, and scratchpad.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Single item in working memory."""
    key: str
    value: Any
    source: str  # Tool name or 'user' or 'agent'
    timestamp: datetime = field(default_factory=datetime.now)
    relevance: float = 1.0  # Decays over time


class WorkingMemory:
    """
    Working memory for current agent task.
    
    Stores:
    - Tool results
    - Intermediate computations
    - Important facts extracted
    
    SKELETON - Core structure defined.
    """
    
    def __init__(self, max_items: int = 10):
        """
        Initialize working memory.
        
        Args:
            max_items: Maximum items to keep
        """
        self.max_items = max_items
        self.items: Dict[str, MemoryItem] = {}
        self.scratchpad: str = ""
    
    def store(
        self,
        key: str,
        value: Any,
        source: str,
        relevance: float = 1.0
    ) -> None:
        """
        Store an item in working memory.
        
        Args:
            key: Unique identifier
            value: Data to store
            source: Where it came from
            relevance: Importance score
        """
        self.items[key] = MemoryItem(
            key=key,
            value=value,
            source=source,
            relevance=relevance
        )
        self._trim()
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve an item by key."""
        item = self.items.get(key)
        return item.value if item else None
    
    def get_by_source(self, source: str) -> List[MemoryItem]:
        """Get all items from a specific source (tool)."""
        return [
            item for item in self.items.values()
            if item.source == source
        ]
    
    def _trim(self) -> None:
        """Keep only top items by relevance."""
        if len(self.items) <= self.max_items:
            return
        
        # Sort by relevance, keep top
        sorted_items = sorted(
            self.items.items(),
            key=lambda x: x[1].relevance,
            reverse=True
        )
        
        self.items = dict(sorted_items[:self.max_items])
    
    def decay_relevance(self, factor: float = 0.9) -> None:
        """Decay relevance of all items."""
        for item in self.items.values():
            item.relevance *= factor
    
    def add_to_scratchpad(self, text: str) -> None:
        """Add text to scratchpad."""
        self.scratchpad += text + "\n"
    
    def get_scratchpad(self) -> str:
        """Get current scratchpad."""
        return self.scratchpad
    
    def clear_scratchpad(self) -> None:
        """Clear scratchpad."""
        self.scratchpad = ""
    
    def get_context(self) -> str:
        """
        Get working memory as context string.
        
        SKELETON - Formats memory for LLM context.
        """
        lines = ["Current working memory:"]
        
        for key, item in self.items.items():
            value_str = str(item.value)[:200]  # Truncate
            lines.append(f"- [{item.source}] {key}: {value_str}")
        
        if self.scratchpad:
            lines.append(f"\nScratchpad:\n{self.scratchpad}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all working memory."""
        self.items = {}
        self.scratchpad = ""
    
    def has_key(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.items


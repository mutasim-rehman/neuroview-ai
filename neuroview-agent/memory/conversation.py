"""
Conversation Memory for multi-turn interactions.

Stores and manages conversation history.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Single conversation message."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """
    Manages conversation history.
    
    Features:
    - Store messages with roles
    - Retrieve recent context
    - Trim old messages
    - Export/import history
    
    SKELETON - Core structure defined.
    """
    
    def __init__(self, max_turns: int = 20):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum conversation turns to keep
        """
        self.max_turns = max_turns
        self.messages: List[Message] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_user_message(self, content: str, **metadata) -> None:
        """Add a user message."""
        self.messages.append(Message(
            role="user",
            content=content,
            metadata=metadata
        ))
        self._trim()
    
    def add_assistant_message(self, content: str, **metadata) -> None:
        """Add an assistant message."""
        self.messages.append(Message(
            role="assistant",
            content=content,
            metadata=metadata
        ))
        self._trim()
    
    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        self.messages.append(Message(
            role="system",
            content=content
        ))
    
    def _trim(self) -> None:
        """Trim to max_turns (user+assistant pairs)."""
        # Keep system messages + last max_turns pairs
        system_msgs = [m for m in self.messages if m.role == "system"]
        other_msgs = [m for m in self.messages if m.role != "system"]
        
        if len(other_msgs) > self.max_turns * 2:
            other_msgs = other_msgs[-(self.max_turns * 2):]
        
        self.messages = system_msgs + other_msgs
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages in chat format."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]
    
    def get_recent(self, n: int = 5) -> List[Message]:
        """Get last n messages."""
        return self.messages[-n:]
    
    def get_context_string(self, max_tokens: int = 2000) -> str:
        """
        Get conversation as string for context.
        
        SKELETON - Implements basic concatenation.
        """
        lines = []
        char_count = 0
        
        for msg in reversed(self.messages):
            line = f"{msg.role.upper()}: {msg.content}"
            if char_count + len(line) > max_tokens * 4:  # Rough token estimate
                break
            lines.insert(0, line)
            char_count += len(line)
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
        self.metadata = {}
    
    def set_disease_context(self, disease: str) -> None:
        """Set current disease being discussed."""
        self.metadata["current_disease"] = disease
    
    def get_disease_context(self) -> Optional[str]:
        """Get current disease context."""
        return self.metadata.get("current_disease")
    
    def export(self) -> Dict[str, Any]:
        """Export conversation for saving."""
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in self.messages
            ],
            "metadata": self.metadata
        }
    
    def import_history(self, data: Dict[str, Any]) -> None:
        """Import conversation from saved data."""
        self.messages = [
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=datetime.fromisoformat(m.get("timestamp", datetime.now().isoformat()))
            )
            for m in data.get("messages", [])
        ]
        self.metadata = data.get("metadata", {})


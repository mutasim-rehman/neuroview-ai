"""
Base tool class for NeuroView Agent.

All tools inherit from BaseTool and implement the execute method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    source: Optional[str] = None  # URL or reference
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_observation(self) -> str:
        """Convert to observation string for agent."""
        if self.success:
            if isinstance(self.data, str):
                return self.data
            elif isinstance(self.data, list):
                return "\n".join(str(item) for item in self.data)
            else:
                return str(self.data)
        else:
            return f"Error: {self.error}"


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.
    
    Each tool must define:
    - name: Unique identifier
    - description: What the tool does (shown to LLM)
    - parameters: Expected input parameters
    - execute(): The actual tool logic
    """
    
    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: Dict[str, Any] = {}
    
    def __init__(self):
        """Initialize the tool."""
        self.logger = logging.getLogger(f"tool.{self.name}")
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with success/failure and data
        """
        pass
    
    def get_definition(self) -> Dict[str, Any]:
        """
        Get tool definition for prompt/function calling.
        
        Returns:
            Dict with name, description, parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def validate_params(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Returns:
            True if valid
        """
        required = self.parameters.get("required", [])
        for param in required:
            if param not in kwargs:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        return True
    
    def __call__(self, **kwargs) -> ToolResult:
        """Allow calling tool as function."""
        return self.execute(**kwargs)
    
    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"


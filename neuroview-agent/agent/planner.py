"""
Agent Planner for deciding tool usage.

Helps the agent decide:
- Which tool to use
- What arguments to provide
- When to stop and give final answer
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    """A planned action."""
    tool: str
    arguments: Dict[str, Any]
    reasoning: str
    confidence: float = 0.0


class AgentPlanner:
    """
    Plans tool usage for the agent.
    
    SKELETON - Core structure for planning logic.
    """
    
    # Keywords that suggest specific tools
    TOOL_KEYWORDS = {
        "web_search": ["search", "find", "look up", "google", "current", "recent", "news"],
        "pubmed_search": ["research", "study", "clinical", "evidence", "paper", "journal", "pubmed"],
        "wikipedia": ["what is", "definition", "overview", "explain", "describe"],
        "drug_lookup": ["drug", "medication", "medicine", "treatment", "side effect"],
        "vision_model": ["scan", "mri", "prediction", "image", "brain scan"],
        "medical_calc": ["calculate", "bmi", "score", "convert"],
    }
    
    # Disease-specific keywords
    DISEASE_KEYWORDS = [
        "glioma", "meningioma", "pituitary", "metastasis", "alzheimer",
        "tumor", "tumour", "cancer", "lesion"
    ]
    
    def __init__(self, tools: List[Dict[str, Any]]):
        """
        Initialize the planner.
        
        Args:
            tools: List of tool definitions
        """
        self.tools = {t["name"]: t for t in tools}
    
    def suggest_tools(self, query: str) -> List[Plan]:
        """
        Suggest tools based on query analysis.
        
        SKELETON - Implementation:
        1. Analyze query keywords
        2. Match to tool capabilities
        3. Return ranked suggestions
        
        Args:
            query: User query
            
        Returns:
            List of suggested Plans
        """
        query_lower = query.lower()
        suggestions = []
        
        # Check for disease keywords
        has_disease = any(kw in query_lower for kw in self.DISEASE_KEYWORDS)
        
        # Match tools by keywords
        for tool_name, keywords in self.TOOL_KEYWORDS.items():
            if tool_name not in self.tools:
                continue
                
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > 0:
                confidence = min(matches / len(keywords), 1.0)
                suggestions.append(Plan(
                    tool=tool_name,
                    arguments=self._suggest_arguments(tool_name, query),
                    reasoning=f"Query contains {matches} matching keywords",
                    confidence=confidence
                ))
        
        # Add wikipedia for disease queries
        if has_disease and "wikipedia" not in [s.tool for s in suggestions]:
            for disease in self.DISEASE_KEYWORDS:
                if disease in query_lower:
                    suggestions.append(Plan(
                        tool="wikipedia",
                        arguments={"topic": disease},
                        reasoning=f"Query mentions {disease}",
                        confidence=0.7
                    ))
                    break
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return suggestions[:3]  # Top 3 suggestions
    
    def _suggest_arguments(self, tool_name: str, query: str) -> Dict[str, Any]:
        """
        Suggest arguments for a tool.
        
        SKELETON - Basic argument suggestion.
        """
        if tool_name in ["web_search", "pubmed_search"]:
            return {"query": query}
        elif tool_name == "wikipedia":
            # Extract topic from query
            return {"topic": query}
        elif tool_name == "vision_model":
            return {"action": "get_prediction"}
        else:
            return {}
    
    def should_stop(
        self,
        steps: List[Dict[str, Any]],
        max_iterations: int
    ) -> bool:
        """
        Decide if agent should stop.
        
        Conditions:
        - Has enough information
        - Max iterations reached
        - Repeated failures
        
        Returns:
            True if should stop
        """
        if len(steps) >= max_iterations:
            return True
        
        # Check for repeated failures
        if len(steps) >= 3:
            recent = steps[-3:]
            failures = sum(1 for s in recent if not s.get("success", True))
            if failures >= 2:
                return True
        
        return False
    
    def extract_search_terms(self, query: str) -> List[str]:
        """
        Extract key search terms from query.
        
        SKELETON - Basic extraction.
        """
        # Remove common words
        stop_words = {"what", "is", "the", "a", "an", "how", "does", "can", "you", "tell", "me", "about"}
        words = query.lower().split()
        terms = [w for w in words if w not in stop_words and len(w) > 2]
        return terms


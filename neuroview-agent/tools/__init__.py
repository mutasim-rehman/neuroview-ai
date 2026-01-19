"""
Tools module for NeuroView Agent.

Available tools:
- WebSearchTool: DuckDuckGo search
- PubMedTool: Search medical literature
- WikipediaTool: Get Wikipedia summaries
- MedlinePlusTool: Patient-friendly medical info
- DrugLookupTool: OpenFDA drug information
- MedicalCalcTool: Medical calculations
- VisionModelTool: Brain scan analysis
"""

from .base import BaseTool, ToolResult
from .web_search import WebSearchTool
from .pubmed import PubMedTool
from .wikipedia import WikipediaTool
from .medlineplus import MedlinePlusTool
from .drug_lookup import DrugLookupTool
from .medical_calc import MedicalCalcTool
from .vision_model import VisionModelTool

# Registry of all available tools
TOOL_REGISTRY = {
    "web_search": WebSearchTool,
    "pubmed_search": PubMedTool,
    "wikipedia": WikipediaTool,
    "medlineplus": MedlinePlusTool,
    "drug_lookup": DrugLookupTool,
    "medical_calc": MedicalCalcTool,
    "vision_model": VisionModelTool,
}


def get_all_tools() -> list:
    """Get instances of all available tools."""
    return [tool_class() for tool_class in TOOL_REGISTRY.values()]


def get_tool(name: str) -> BaseTool:
    """Get a specific tool by name."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")
    return TOOL_REGISTRY[name]()


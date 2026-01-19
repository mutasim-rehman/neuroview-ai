"""
Tool prompt formatting for the agent.

Formats tool definitions for inclusion in prompts.
"""

from typing import List, Dict, Any


def format_tools_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Format tool definitions for the system prompt.
    
    Args:
        tools: List of tool definitions
        
    Returns:
        Formatted tools string
    """
    lines = []
    
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "No description")
        params = tool.get("parameters", {})
        
        lines.append(f"### {name}")
        lines.append(f"Description: {desc}")
        
        if params.get("properties"):
            lines.append("Parameters:")
            for param_name, param_info in params["properties"].items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                required = param_name in params.get("required", [])
                req_str = " (required)" if required else ""
                lines.append(f"  - {param_name}: {param_type}{req_str} - {param_desc}")
        
        lines.append("")
    
    return "\n".join(lines)


def format_tool_call_example(tool_name: str, args: Dict[str, Any]) -> str:
    """
    Format an example tool call.
    
    Args:
        tool_name: Name of the tool
        args: Example arguments
        
    Returns:
        Formatted example
    """
    import json
    return f"Action: {tool_name}\nAction Input: {json.dumps(args)}"


# Pre-formatted tool descriptions for common tools
TOOL_DESCRIPTIONS = {
    "web_search": """### web_search
Search the web using DuckDuckGo.
Use for: General queries, current information, recent news
Parameters:
  - query (required): Search query string
  - max_results: Number of results (default: 5)
Example: Action: web_search
Action Input: {"query": "glioma treatment 2024"}""",

    "pubmed_search": """### pubmed_search
Search PubMed medical literature database.
Use for: Peer-reviewed research, clinical studies, evidence-based information
Parameters:
  - query (required): Medical search terms
  - max_results: Number of results (default: 5)
Example: Action: pubmed_search
Action Input: {"query": "glioblastoma immunotherapy"}""",

    "wikipedia": """### wikipedia
Get Wikipedia article summaries.
Use for: Disease overviews, definitions, general knowledge
Parameters:
  - topic (required): Topic to look up
  - sentences: Number of sentences (default: 5)
Example: Action: wikipedia
Action Input: {"topic": "Meningioma"}""",

    "vision_model": """### vision_model
Get brain scan analysis from NeuroView vision model.
Use for: Scan predictions, disease detection results
Parameters:
  - action (required): 'get_prediction', 'get_details'
  - scan_id: Scan identifier
Example: Action: vision_model
Action Input: {"action": "get_prediction"}""",

    "drug_lookup": """### drug_lookup
Look up drug information from OpenFDA.
Use for: Drug uses, side effects, interactions
Parameters:
  - drug_name (required): Name of medication
Example: Action: drug_lookup
Action Input: {"drug_name": "temozolomide"}"""
}


def get_compact_tools_description() -> str:
    """Get a compact version of all tool descriptions."""
    return "\n\n".join(TOOL_DESCRIPTIONS.values())


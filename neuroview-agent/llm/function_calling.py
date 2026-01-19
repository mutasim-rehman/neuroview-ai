"""
Function calling / tool use parsing for the agent.

Handles:
- Parsing tool calls from LLM output
- Formatting tool definitions for prompts
- Validating tool arguments
"""

import json
import re
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a parsed tool call."""
    name: str
    arguments: Dict[str, Any]
    raw_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments
        }


class FunctionCallParser:
    """
    Parses function/tool calls from LLM output.
    
    Supports multiple formats:
    - JSON format: {"tool": "name", "args": {...}}
    - ReAct format: Action: tool_name\nAction Input: {...}
    - XML format: <tool_call><name>...</name><args>...</args></tool_call>
    
    SKELETON - Core structure defined.
    """
    
    # Regex patterns for different formats
    REACT_PATTERN = r"Action:\s*(\w+)\s*\nAction Input:\s*(.+?)(?=\n(?:Observation|Thought|$))"
    JSON_PATTERN = r'\{[^{}]*"(?:tool|function|name)"[^{}]*\}'
    
    def __init__(self, format: str = "react"):
        """
        Initialize the parser.
        
        Args:
            format: Output format ('react', 'json', 'xml')
        """
        self.format = format
    
    def parse(self, text: str) -> Optional[ToolCall]:
        """
        Parse a tool call from LLM output.
        
        SKELETON - Implementation:
        1. Try to match format pattern
        2. Extract tool name and arguments
        3. Validate and return ToolCall
        
        Args:
            text: LLM output text
            
        Returns:
            ToolCall if found, None otherwise
        """
        if self.format == "react":
            return self._parse_react(text)
        elif self.format == "json":
            return self._parse_json(text)
        else:
            return None
    
    def _parse_react(self, text: str) -> Optional[ToolCall]:
        """
        Parse ReAct format tool call.
        
        Format:
        Action: tool_name
        Action Input: {"arg1": "value1"}
        
        SKELETON - Implementation pending.
        """
        # TODO: Implement ReAct parsing
        # match = re.search(self.REACT_PATTERN, text, re.DOTALL)
        # if match:
        #     tool_name = match.group(1).strip()
        #     args_str = match.group(2).strip()
        #     try:
        #         arguments = json.loads(args_str)
        #     except json.JSONDecodeError:
        #         arguments = {"input": args_str}
        #     return ToolCall(name=tool_name, arguments=arguments, raw_text=match.group(0))
        
        logger.debug("SKELETON: Would parse ReAct format")
        return None
    
    def _parse_json(self, text: str) -> Optional[ToolCall]:
        """
        Parse JSON format tool call.
        
        SKELETON - Implementation pending.
        """
        # TODO: Implement JSON parsing
        logger.debug("SKELETON: Would parse JSON format")
        return None
    
    def format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Format tool definitions for inclusion in prompt.
        
        SKELETON - Implementation:
        Create clear tool descriptions for the LLM.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            Formatted string for prompt
        """
        # TODO: Implement tool formatting
        lines = ["Available tools:"]
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "No description")
            params = tool.get("parameters", {})
            lines.append(f"\n- {name}: {desc}")
            if params:
                lines.append(f"  Parameters: {json.dumps(params)}")
        
        return "\n".join(lines)
    
    def validate_tool_call(
        self,
        tool_call: ToolCall,
        tool_definitions: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate a tool call against definitions.
        
        SKELETON - Implementation:
        1. Check tool name exists
        2. Validate required arguments
        3. Check argument types
        
        Returns:
            True if valid
        """
        # TODO: Implement validation
        logger.debug("SKELETON: Would validate tool call")
        return True


class ReActPromptBuilder:
    """
    Builds prompts in ReAct format for the agent.
    
    ReAct = Reasoning + Acting
    Format:
        Thought: I need to...
        Action: tool_name
        Action Input: {...}
        Observation: [tool result]
        ... repeat ...
        Thought: I now have enough information
        Final Answer: ...
    """
    
    REACT_TEMPLATE = """You are a medical AI assistant. Answer questions by using the available tools.

{tools}

Use this format:

Thought: Think about what you need to do
Action: The tool to use (one of: {tool_names})
Action Input: The input for the tool (as JSON)
Observation: The result from the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to answer
Final Answer: Your final response to the user

IMPORTANT: Always include medical disclaimers in your final answer.

Begin!

Question: {question}
{agent_scratchpad}"""

    def build_prompt(
        self,
        question: str,
        tools: List[Dict[str, Any]],
        scratchpad: str = ""
    ) -> str:
        """
        Build a ReAct prompt.
        
        Args:
            question: User question
            tools: Available tools
            scratchpad: Previous thoughts/actions/observations
            
        Returns:
            Formatted prompt
        """
        parser = FunctionCallParser()
        tools_str = parser.format_tools_for_prompt(tools)
        tool_names = ", ".join(t.get("name", "") for t in tools)
        
        return self.REACT_TEMPLATE.format(
            tools=tools_str,
            tool_names=tool_names,
            question=question,
            agent_scratchpad=scratchpad
        )


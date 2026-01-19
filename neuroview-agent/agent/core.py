"""
Core Agent implementation using ReAct pattern.

ReAct = Reasoning + Acting
The agent thinks, acts (uses tools), observes results, and repeats.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Current state of the agent."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class AgentStep:
    """Single step in agent execution."""
    thought: str = ""
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: str = ""
    

@dataclass
class AgentResponse:
    """Final response from the agent."""
    answer: str
    steps: List[AgentStep]
    sources: List[str]
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NeuroViewAgent:
    """
    Main agent class implementing ReAct pattern.
    
    Flow:
    1. Receive user query
    2. THINK: Decide what to do
    3. ACT: Execute a tool
    4. OBSERVE: Process tool result
    5. Repeat 2-4 until answer is ready
    6. Return final answer with sources
    
    SKELETON - Core structure defined, implementation pending.
    """
    
    def __init__(
        self,
        llm,  # LocalLLM instance
        tools: List,  # List of BaseTool instances
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the agent.
        
        Args:
            llm: Language model instance
            tools: Available tools
            max_iterations: Max ReAct loops
            verbose: Log detailed steps
        """
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        self.state = AgentState.IDLE
        self.steps: List[AgentStep] = []
        self.scratchpad = ""
    
    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Run the agent on a query.
        
        SKELETON - Implementation:
        1. Build initial prompt with tools
        2. Loop: think -> act -> observe
        3. Parse final answer
        4. Return with sources
        
        Args:
            query: User question
            context: Optional context (e.g., prediction data)
            
        Returns:
            AgentResponse with answer and steps
        """
        self.state = AgentState.THINKING
        self.steps = []
        self.scratchpad = ""
        sources = []
        
        # Add context if provided (e.g., from vision model)
        if context:
            self._add_context(context)
        
        logger.info(f"Agent starting for query: {query[:50]}...")
        
        for iteration in range(self.max_iterations):
            logger.debug(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Step 1: Think and decide action
            step = AgentStep()
            thought_result = self._think(query)
            step.thought = thought_result.get("thought", "")
            
            # Check if we have final answer
            if thought_result.get("final_answer"):
                self.state = AgentState.FINISHED
                return AgentResponse(
                    answer=thought_result["final_answer"],
                    steps=self.steps,
                    sources=list(set(sources)),
                    success=True,
                    metadata={"iterations": iteration + 1}
                )
            
            # Step 2: Execute action
            action = thought_result.get("action")
            action_input = thought_result.get("action_input", {})
            
            if action:
                step.action = action
                step.action_input = action_input
                
                self.state = AgentState.ACTING
                observation = self._act(action, action_input)
                step.observation = observation.get("result", "")
                
                if observation.get("source"):
                    sources.append(observation["source"])
                
                self._update_scratchpad(step)
            
            self.steps.append(step)
            self.state = AgentState.THINKING
        
        # Max iterations reached
        self.state = AgentState.ERROR
        return AgentResponse(
            answer="I was unable to fully answer your question. Please try rephrasing.",
            steps=self.steps,
            sources=sources,
            success=False,
            error="Max iterations reached"
        )
    
    def _think(self, query: str) -> Dict[str, Any]:
        """
        Think step - decide what to do next.
        
        SKELETON - Implementation:
        1. Build prompt with query + scratchpad
        2. Call LLM
        3. Parse response for action or final answer
        
        Returns:
            Dict with thought, action, action_input, or final_answer
        """
        # TODO: Implement thinking
        # prompt = self._build_prompt(query)
        # response = self.llm.generate(prompt)
        # parsed = self._parse_response(response.text)
        # return parsed
        
        logger.info("SKELETON: Would think about query")
        
        # Return mock response for skeleton
        return {
            "thought": "[SKELETON] Thinking about the query...",
            "action": "wikipedia",
            "action_input": {"topic": "glioma"}
        }
    
    def _act(self, action: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Act step - execute a tool.
        
        SKELETON - Implementation:
        1. Get tool by name
        2. Execute with inputs
        3. Return observation
        
        Returns:
            Dict with result and source
        """
        if action not in self.tools:
            return {
                "result": f"Error: Unknown tool '{action}'",
                "source": None
            }
        
        tool = self.tools[action]
        
        try:
            result = tool.execute(**action_input)
            return {
                "result": result.to_observation(),
                "source": result.source
            }
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "result": f"Error executing {action}: {str(e)}",
                "source": None
            }
    
    def _add_context(self, context: Dict[str, Any]) -> None:
        """Add context to the scratchpad."""
        if "prediction" in context:
            pred = context["prediction"]
            self.scratchpad += f"\nContext: Vision model prediction - Disease: {pred.get('disease')}, "
            self.scratchpad += f"Confidence: {pred.get('confidence')}, Location: {pred.get('location')}\n"
    
    def _update_scratchpad(self, step: AgentStep) -> None:
        """Update scratchpad with latest step."""
        self.scratchpad += f"\nThought: {step.thought}"
        if step.action:
            self.scratchpad += f"\nAction: {step.action}"
            self.scratchpad += f"\nAction Input: {step.action_input}"
            self.scratchpad += f"\nObservation: {step.observation}"
    
    def _build_prompt(self, query: str) -> str:
        """
        Build the full prompt for the LLM.
        
        SKELETON - Would use ReActPromptBuilder.
        """
        # TODO: Implement prompt building
        return f"Query: {query}\n{self.scratchpad}"
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get definitions of all available tools."""
        return [tool.get_definition() for tool in self.tools.values()]
    
    def reset(self) -> None:
        """Reset agent state."""
        self.state = AgentState.IDLE
        self.steps = []
        self.scratchpad = ""


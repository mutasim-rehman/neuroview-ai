"""
Configuration for NeuroView Agent.

Hardware Target:
- GPU: RTX 4060 6GB VRAM
- RAM: 16GB
- CPU: Ryzen 5
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


# Base paths
BASE_DIR = Path(__file__).parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"


@dataclass
class LLMConfig:
    """Configuration for the local LLM."""
    
    # Model selection - optimized for 6GB VRAM
    # Option 1: Mistral (recommended for agents)
    model_name: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_file: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    # Option 2: LLaMA 3
    # model_name: str = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
    # model_file: str = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    
    # Model parameters
    context_length: int = 4096
    max_tokens: int = 1024
    temperature: float = 0.1  # Low for consistent tool use
    top_p: float = 0.9
    
    # Hardware optimization
    n_gpu_layers: int = 35
    n_batch: int = 512
    n_threads: int = 8
    
    # Paths
    model_cache_dir: str = str(CHECKPOINTS_DIR / "models")


@dataclass
class ToolsConfig:
    """Configuration for agent tools."""
    
    # Tool timeouts (seconds)
    web_search_timeout: int = 10
    pubmed_timeout: int = 15
    wikipedia_timeout: int = 10
    
    # API settings
    pubmed_max_results: int = 5
    web_search_max_results: int = 5
    
    # Vision model integration
    vision_model_url: str = "http://localhost:8000"
    
    # Rate limiting
    requests_per_minute: int = 30


@dataclass
class MemoryConfig:
    """Configuration for agent memory."""
    
    max_conversation_turns: int = 20
    max_working_memory_items: int = 10
    summarize_after_turns: int = 10
    max_context_tokens: int = 3000


@dataclass
class AgentConfig:
    """Main agent configuration."""
    
    # Agent behavior
    max_iterations: int = 10  # Max ReAct loops
    max_tool_retries: int = 2
    
    # Safety
    always_include_disclaimer: bool = True
    require_source_citation: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_tool_calls: bool = True
    
    # Sub-configs
    llm: LLMConfig = field(default_factory=LLMConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "llm": {
                "model_name": self.llm.model_name,
                "context_length": self.llm.context_length,
            },
            "tools": {
                "vision_model_url": self.tools.vision_model_url,
            }
        }


# Global config instance
config = AgentConfig()


def ensure_directories():
    """Create necessary directories."""
    directories = [
        CHECKPOINTS_DIR,
        CHECKPOINTS_DIR / "models",
        LOGS_DIR,
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)


ensure_directories()


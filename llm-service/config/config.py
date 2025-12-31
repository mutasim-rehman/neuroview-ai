"""
Configuration settings for NeuroView LLM Service.

Hardware Target:
- GPU: RTX 4060 6GB VRAM
- RAM: 16GB
- CPU: Ryzen 5

Model Selection:
- Primary: LLaMA 3.2 3B Instruct (4-bit quantized) - ~2-3GB VRAM
- Alternative: LLaMA 2 7B Chat (4-bit quantized) - ~4-5GB VRAM
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


# Base paths
BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "rag" / "knowledge_base"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"


@dataclass
class LLMConfig:
    """Configuration for the LLM model."""
    
    # Model selection - optimized for 6GB VRAM
    model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF"  # HuggingFace repo
    model_file: str = "llama-2-7b-chat.Q4_K_M.gguf"    # 4-bit quantized
    
    # Alternative: Smaller model for lower VRAM usage
    # model_name: str = "lmstudio-community/Llama-3.2-3B-Instruct-GGUF"
    # model_file: str = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    
    # Model parameters
    context_length: int = 4096
    max_tokens: int = 1024
    temperature: float = 0.3  # Lower for medical accuracy
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    
    # Hardware optimization
    n_gpu_layers: int = 35  # Layers to offload to GPU (-1 for all)
    n_batch: int = 512      # Batch size for prompt processing
    n_threads: int = 8      # CPU threads (Ryzen 5 has 6-12 threads)
    use_mmap: bool = True   # Memory-mapped model loading
    use_mlock: bool = False # Lock model in RAM
    
    # Paths
    model_cache_dir: str = str(CHECKPOINTS_DIR / "llm_models")
    
    # Generation settings for medical context
    stop_sequences: List[str] = field(default_factory=lambda: [
        "\n\nUser:",
        "\n\nHuman:",
        "\n\nPatient:",
        "</s>",
        "[/INST]"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_file": self.model_file,
            "context_length": self.context_length,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n_gpu_layers": self.n_gpu_layers,
            "n_batch": self.n_batch,
            "n_threads": self.n_threads
        }


@dataclass
class RAGConfig:
    """Configuration for Retrieval-Augmented Generation."""
    
    # Embedding model - runs efficiently on CPU/GPU
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Alternative: Medical-specific embeddings
    # embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    # embedding_dimension: int = 768
    
    # Vector store settings
    vector_store_type: str = "chroma"  # Options: chroma, faiss, qdrant
    collection_name: str = "neuroview_medical_knowledge"
    persist_directory: str = str(KNOWLEDGE_BASE_DIR / "vector_store")
    
    # Retrieval settings
    top_k: int = 5              # Number of documents to retrieve
    similarity_threshold: float = 0.5  # Minimum similarity score
    chunk_size: int = 512       # Document chunk size
    chunk_overlap: int = 50     # Overlap between chunks
    
    # Knowledge base paths
    knowledge_base_dir: str = str(KNOWLEDGE_BASE_DIR / "documents")
    processed_docs_dir: str = str(KNOWLEDGE_BASE_DIR / "processed")
    
    # Supported diseases for knowledge retrieval
    target_diseases: List[str] = field(default_factory=lambda: [
        "glioma",
        "meningioma",
        "pituitary_tumor",
        "brain_metastases",
        "alzheimer",
        "healthy_brain"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "vector_store_type": self.vector_store_type,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size
        }


@dataclass
class FineTuningConfig:
    """Configuration for Supervised Fine-Tuning (SFT)."""
    
    # Base model for fine-tuning
    base_model: str = "meta-llama/Llama-2-7b-chat-hf"
    
    # LoRA/QLoRA settings (for efficient fine-tuning on limited VRAM)
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training settings
    learning_rate: float = 2e-4
    batch_size: int = 1  # Small batch for 6GB VRAM
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    
    # 4-bit quantization for training
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Paths
    output_dir: str = str(CHECKPOINTS_DIR / "fine_tuned")
    dataset_dir: str = str(BASE_DIR / "fine_tuning" / "datasets")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "use_qlora": self.use_qlora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs
        }


@dataclass
class APIConfig:
    """Configuration for the API server."""
    
    host: str = "0.0.0.0"
    port: int = 8001  # Different from ai-training API (8000)
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    max_requests_per_minute: int = 30
    max_concurrent_requests: int = 2  # Limited by VRAM
    
    # Timeouts
    request_timeout: int = 120  # seconds
    generation_timeout: int = 90  # seconds


# Global configuration instances
llm_config = LLMConfig()
rag_config = RAGConfig()
fine_tuning_config = FineTuningConfig()
api_config = APIConfig()


def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        KNOWLEDGE_BASE_DIR,
        KNOWLEDGE_BASE_DIR / "documents",
        KNOWLEDGE_BASE_DIR / "processed",
        KNOWLEDGE_BASE_DIR / "vector_store",
        CHECKPOINTS_DIR,
        CHECKPOINTS_DIR / "llm_models",
        CHECKPOINTS_DIR / "fine_tuned",
        LOGS_DIR,
        BASE_DIR / "fine_tuning" / "datasets"
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Create directories on import
ensure_directories()


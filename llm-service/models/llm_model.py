"""
NeuroView LLM Model Wrapper.

Provides a unified interface for loading and running LLaMA models
with optimization for limited VRAM (6GB RTX 4060).

Supports:
- GGUF quantized models via llama-cpp-python
- HuggingFace transformers (for fine-tuning)
- Streaming generation
"""

import logging
import time
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ModelBackend(Enum):
    """Supported model backends."""
    LLAMA_CPP = "llama_cpp"
    TRANSFORMERS = "transformers"


@dataclass
class LLMResponse:
    """Response from the LLM."""
    
    text: str
    tokens_generated: int
    generation_time: float
    prompt_tokens: int = 0
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tokens_per_second(self) -> float:
        if self.generation_time > 0:
            return self.tokens_generated / self.generation_time
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "generation_time": self.generation_time,
            "tokens_per_second": self.tokens_per_second,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata
        }


class NeuroViewLLM:
    """
    NeuroView LLM wrapper for medical conversation.
    
    Optimized for RTX 4060 6GB VRAM:
    - Uses 4-bit quantized GGUF models
    - Efficient GPU offloading
    - Streaming generation support
    
    Example usage:
        llm = NeuroViewLLM()
        llm.load_model()
        response = llm.generate("Explain glioma")
    """
    
    # Medical safety prefix
    MEDICAL_DISCLAIMER = (
        "IMPORTANT: This information is for educational purposes only and "
        "should not be used as a substitute for professional medical advice, "
        "diagnosis, or treatment. Always consult a qualified healthcare provider."
    )
    
    def __init__(
        self,
        model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF",
        model_file: str = "llama-2-7b-chat.Q4_K_M.gguf",
        backend: ModelBackend = ModelBackend.LLAMA_CPP,
        context_length: int = 4096,
        n_gpu_layers: int = 35,
        n_batch: int = 512,
        n_threads: int = 8,
        model_cache_dir: Optional[str] = None
    ):
        """
        Initialize the LLM wrapper.
        
        Args:
            model_name: HuggingFace repo name
            model_file: Model filename (for GGUF)
            backend: Model backend to use
            context_length: Context window size
            n_gpu_layers: Layers to offload to GPU
            n_batch: Batch size for prompt processing
            n_threads: CPU threads to use
            model_cache_dir: Directory for model cache
        """
        self.model_name = model_name
        self.model_file = model_file
        self.backend = backend
        self.context_length = context_length
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.n_threads = n_threads
        
        # Set cache directory
        if model_cache_dir:
            self.model_cache_dir = Path(model_cache_dir)
        else:
            self.model_cache_dir = Path(__file__).parent.parent / "checkpoints" / "llm_models"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self._is_loaded = False
    
    def load_model(self) -> bool:
        """
        Load the LLM model.
        
        Returns:
            True if successful, False otherwise
        """
        if self._is_loaded:
            logger.info("Model already loaded")
            return True
        
        if self.backend == ModelBackend.LLAMA_CPP:
            return self._load_llama_cpp()
        elif self.backend == ModelBackend.TRANSFORMERS:
            return self._load_transformers()
        
        return False
    
    def _load_llama_cpp(self) -> bool:
        """Load model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download
            
            logger.info(f"Downloading/loading model: {self.model_name}/{self.model_file}")
            
            # Download model from HuggingFace
            model_path = hf_hub_download(
                repo_id=self.model_name,
                filename=self.model_file,
                cache_dir=str(self.model_cache_dir),
                resume_download=True
            )
            
            logger.info(f"Loading model from: {model_path}")
            logger.info(f"GPU layers: {self.n_gpu_layers}, Batch: {self.n_batch}")
            
            # Load model with GPU acceleration
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.context_length,
                n_gpu_layers=self.n_gpu_layers,
                n_batch=self.n_batch,
                n_threads=self.n_threads,
                use_mmap=True,
                verbose=False
            )
            
            self._is_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except ImportError:
            logger.error(
                "llama-cpp-python not installed. Install with:\n"
                "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _load_transformers(self) -> bool:
        """Load model using HuggingFace transformers."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            logger.info(f"Loading model with transformers: {self.model_name}")
            
            # 4-bit quantization config for limited VRAM
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.model_cache_dir)
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                cache_dir=str(self.model_cache_dir)
            )
            
            self._is_loaded = True
            logger.info("Model loaded successfully with transformers")
            return True
            
        except ImportError:
            logger.error(
                "transformers or bitsandbytes not installed. Install with:\n"
                "pip install transformers bitsandbytes accelerate"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ) -> LLMResponse:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more focused)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
            stop_sequences: Sequences that stop generation
            stream: Whether to stream output
            
        Returns:
            LLMResponse object
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.backend == ModelBackend.LLAMA_CPP:
            return self._generate_llama_cpp(
                prompt, max_tokens, temperature, top_p, top_k,
                repeat_penalty, stop_sequences, stream
            )
        elif self.backend == ModelBackend.TRANSFORMERS:
            return self._generate_transformers(
                prompt, max_tokens, temperature, top_p, top_k,
                repeat_penalty, stop_sequences
            )
        
        raise RuntimeError(f"Unsupported backend: {self.backend}")
    
    def _generate_llama_cpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop_sequences: Optional[List[str]],
        stream: bool
    ) -> LLMResponse:
        """Generate using llama-cpp-python."""
        start_time = time.time()
        
        # Default stop sequences for chat models
        if stop_sequences is None:
            stop_sequences = ["</s>", "[/INST]", "\n\nUser:", "\n\nHuman:"]
        
        if stream:
            return self._stream_llama_cpp(
                prompt, max_tokens, temperature, top_p, top_k,
                repeat_penalty, stop_sequences
            )
        
        # Non-streaming generation
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop_sequences,
            echo=False
        )
        
        generation_time = time.time() - start_time
        text = output["choices"][0]["text"]
        tokens_generated = output["usage"]["completion_tokens"]
        prompt_tokens = output["usage"]["prompt_tokens"]
        finish_reason = output["choices"][0].get("finish_reason", "stop")
        
        return LLMResponse(
            text=text,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            prompt_tokens=prompt_tokens,
            finish_reason=finish_reason,
            metadata={"backend": "llama_cpp"}
        )
    
    def _stream_llama_cpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop_sequences: List[str]
    ) -> Generator[str, None, LLMResponse]:
        """Stream generation using llama-cpp-python."""
        start_time = time.time()
        tokens_generated = 0
        full_text = ""
        
        stream = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop_sequences,
            echo=False,
            stream=True
        )
        
        for output in stream:
            text_chunk = output["choices"][0]["text"]
            full_text += text_chunk
            tokens_generated += 1
            yield text_chunk
        
        generation_time = time.time() - start_time
        
        return LLMResponse(
            text=full_text,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            finish_reason="stop",
            metadata={"backend": "llama_cpp", "streamed": True}
        )
    
    def _generate_transformers(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop_sequences: Optional[List[str]]
    ) -> LLMResponse:
        """Generate using HuggingFace transformers."""
        import torch
        
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repeat_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output, excluding prompt
        generated = outputs[0][prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        tokens_generated = len(generated)
        
        generation_time = time.time() - start_time
        
        return LLMResponse(
            text=text,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            prompt_tokens=prompt_tokens,
            finish_reason="stop",
            metadata={"backend": "transformers"}
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Chat-style generation with message history.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse object
        """
        prompt = self._format_chat_prompt(messages, system_prompt)
        return self.generate(prompt, **kwargs)
    
    def _format_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format messages into LLaMA chat format.
        
        LLaMA 2 Chat format:
        <s>[INST] <<SYS>>
        {system_prompt}
        <</SYS>>
        
        {user_message} [/INST] {assistant_message} </s><s>[INST] {user_message} [/INST]
        """
        parts = []
        
        # Start with system prompt if provided
        if system_prompt:
            parts.append(f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n")
        else:
            parts.append("<s>[INST] ")
        
        # Add conversation history
        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                if i == 0 and system_prompt:
                    parts.append(f"{content} [/INST] ")
                elif i == 0:
                    parts[-1] = f"<s>[INST] {content} [/INST] "
                else:
                    parts.append(f"<s>[INST] {content} [/INST] ")
            elif role == "assistant":
                parts.append(f"{content} </s>")
        
        return "".join(parts)
    
    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_loaded = False
            
            # Clear CUDA cache if using GPU
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_file": self.model_file,
            "backend": self.backend.value,
            "context_length": self.context_length,
            "n_gpu_layers": self.n_gpu_layers,
            "is_loaded": self._is_loaded
        }


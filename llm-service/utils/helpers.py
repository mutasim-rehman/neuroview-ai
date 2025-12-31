"""
Helper utilities for NeuroView LLM Service.

Provides common functionality for:
- Logging setup
- Device detection
- Response formatting
- Medical text processing
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional specific log file name
        log_dir: Optional log directory
        
    Returns:
        Configured logger
    """
    # Determine log file path
    if log_dir:
        log_path = Path(log_dir)
    else:
        log_path = Path(__file__).parent.parent / "logs"
    
    log_path.mkdir(parents=True, exist_ok=True)
    
    if log_file:
        log_file_path = log_path / log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_path / f"llm_service_{timestamp}.log"
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("neuroview_llm")
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    
    return logger


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "cuda_memory_total": None,
        "cuda_memory_free": None,
        "cpu_count": None,
        "recommended_backend": "cpu"
    }
    
    # Check CUDA availability
    try:
        import torch
        
        info["cuda_available"] = torch.cuda.is_available()
        
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            
            # Get memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory = total_memory - torch.cuda.memory_allocated(0)
            
            info["cuda_memory_total"] = f"{total_memory / 1024**3:.2f} GB"
            info["cuda_memory_free"] = f"{free_memory / 1024**3:.2f} GB"
            info["recommended_backend"] = "cuda"
            
    except ImportError:
        pass
    
    # Get CPU info
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except:
        pass
    
    return info


def format_medical_response(
    response: str,
    include_disclaimer: bool = True,
    disclaimer_position: str = "end"
) -> str:
    """
    Format a medical response with appropriate disclaimers.
    
    Args:
        response: Raw response text
        include_disclaimer: Whether to include medical disclaimer
        disclaimer_position: 'start', 'end', or 'both'
        
    Returns:
        Formatted response
    """
    DISCLAIMER = (
        "\n\n---\n"
        "**Medical Disclaimer:** This information is provided for educational purposes only "
        "and should not be considered medical advice. Please consult with a qualified "
        "healthcare professional for any medical concerns."
    )
    
    if not include_disclaimer:
        return response
    
    if disclaimer_position == "start":
        return DISCLAIMER + "\n\n" + response
    elif disclaimer_position == "both":
        return DISCLAIMER + "\n\n" + response + DISCLAIMER
    else:  # end
        return response + DISCLAIMER


def clean_medical_text(text: str) -> str:
    """
    Clean and normalize medical text.
    
    Args:
        text: Raw medical text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Normalize common medical abbreviations
    # (This is a simple example - could be expanded)
    
    return text.strip()


def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities from text.
    
    This is a placeholder for NER functionality.
    Could be enhanced with a medical NER model.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of entity types and values
    """
    # Placeholder - would use medical NER in production
    entities = {
        "diseases": [],
        "symptoms": [],
        "medications": [],
        "procedures": [],
        "anatomy": []
    }
    
    # Simple keyword matching (placeholder)
    disease_keywords = [
        "glioma", "meningioma", "pituitary", "metastasis", "metastases",
        "alzheimer", "tumor", "tumour", "cancer"
    ]
    
    anatomy_keywords = [
        "brain", "frontal", "temporal", "parietal", "occipital",
        "hippocampus", "pituitary", "hypothalamus", "cerebellum",
        "brainstem", "ventricle"
    ]
    
    text_lower = text.lower()
    
    for keyword in disease_keywords:
        if keyword in text_lower:
            entities["diseases"].append(keyword)
    
    for keyword in anatomy_keywords:
        if keyword in text_lower:
            entities["anatomy"].append(keyword)
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities


def calculate_response_metrics(
    response: str,
    generation_time: float,
    tokens_generated: int
) -> Dict[str, Any]:
    """
    Calculate metrics for a generated response.
    
    Args:
        response: Generated text
        generation_time: Time taken to generate
        tokens_generated: Number of tokens generated
        
    Returns:
        Dictionary of metrics
    """
    return {
        "response_length": len(response),
        "word_count": len(response.split()),
        "sentence_count": response.count('.') + response.count('!') + response.count('?'),
        "tokens_generated": tokens_generated,
        "generation_time_seconds": round(generation_time, 3),
        "tokens_per_second": round(tokens_generated / generation_time, 2) if generation_time > 0 else 0
    }


def validate_disease_name(disease: str) -> Optional[str]:
    """
    Validate and normalize a disease name.
    
    Args:
        disease: Input disease name
        
    Returns:
        Normalized disease name or None if invalid
    """
    VALID_DISEASES = {
        "glioma": "glioma",
        "meningioma": "meningioma",
        "pituitary": "pituitary_tumor",
        "pituitary_tumor": "pituitary_tumor",
        "pituitary tumor": "pituitary_tumor",
        "pituitary adenoma": "pituitary_tumor",
        "metastasis": "brain_metastases",
        "metastases": "brain_metastases",
        "brain_metastases": "brain_metastases",
        "brain metastases": "brain_metastases",
        "alzheimer": "alzheimer",
        "alzheimers": "alzheimer",
        "alzheimer's": "alzheimer",
        "healthy": "healthy_brain",
        "healthy_brain": "healthy_brain",
        "normal": "healthy_brain",
        "no_tumor": "healthy_brain"
    }
    
    normalized = disease.lower().strip()
    return VALID_DISEASES.get(normalized)


def create_conversation_id() -> str:
    """Generate a unique conversation ID."""
    import uuid
    return f"conv_{uuid.uuid4().hex[:12]}"


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.
    
    This is a rough estimate - actual tokenization depends on the model.
    
    Args:
        text: Text to estimate
        
    Returns:
        Estimated token count
    """
    # Rough estimate: ~4 characters per token for English
    return len(text) // 4


def truncate_context(
    context: str,
    max_tokens: int = 2000,
    preserve_start: bool = True
) -> str:
    """
    Truncate context to fit within token limit.
    
    Args:
        context: Context text
        max_tokens: Maximum tokens allowed
        preserve_start: If True, preserve start; if False, preserve end
        
    Returns:
        Truncated context
    """
    estimated_tokens = estimate_tokens(context)
    
    if estimated_tokens <= max_tokens:
        return context
    
    # Calculate approximate character limit
    char_limit = max_tokens * 4
    
    if preserve_start:
        return context[:char_limit] + "... [truncated]"
    else:
        return "[truncated] ..." + context[-char_limit:]


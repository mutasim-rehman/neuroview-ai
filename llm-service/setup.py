"""Setup script for NeuroView LLM Service."""

from setuptools import setup, find_packages

setup(
    name="neuroview-llm",
    version="0.1.0",
    description="Clinical Decision-Support and Educational Conversational Model for Brain MRI Analysis",
    author="NeuroView Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "llama-cpp-python>=0.2.56",
        "huggingface-hub>=0.20.0",
        "transformers>=4.36.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.22",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.5.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=24.1.0",
            "isort>=5.13.0",
        ],
        "fine-tuning": [
            "torch>=2.1.0",
            "bitsandbytes>=0.42.0",
            "peft>=0.8.0",
            "datasets>=2.16.0",
            "accelerate>=0.26.0",
            "trl>=0.7.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuroview-llm=llm_service.main:main",
        ],
    },
)


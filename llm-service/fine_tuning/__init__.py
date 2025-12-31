"""
Fine-tuning module for NeuroView LLM.

Provides supervised fine-tuning (SFT) capabilities using:
- QLoRA for efficient training on limited VRAM
- Custom medical conversation datasets
- Safety-aware training procedures
"""

from .dataset_preparation import DatasetPreparer, MedicalConversation
# from .trainer import FineTuningTrainer  # TODO: Implement


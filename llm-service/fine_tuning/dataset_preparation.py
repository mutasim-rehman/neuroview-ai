"""
Dataset Preparation for Supervised Fine-Tuning.

Prepares medical conversation datasets for training NeuroView LLM.

Data sources to consider:
- Synthetic medical conversations (generated with safety constraints)
- Medical Q&A datasets (PubMedQA, MedQA - if licensed appropriately)
- Custom neurological disease conversations
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import random

logger = logging.getLogger(__name__)


@dataclass
class MedicalConversation:
    """
    Represents a medical conversation for training.
    
    Structure follows chat format for LLaMA fine-tuning.
    """
    
    id: str
    disease: str
    conversation: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_training_format(self, system_prompt: str = "") -> str:
        """
        Convert to LLaMA chat training format.
        
        Format:
        <s>[INST] <<SYS>>
        {system_prompt}
        <</SYS>>
        
        {user_msg} [/INST] {assistant_msg} </s>
        """
        parts = []
        
        if system_prompt:
            parts.append(f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n")
        
        for i, turn in enumerate(self.conversation):
            role = turn["role"]
            content = turn["content"]
            
            if role == "user":
                if i == 0 and system_prompt:
                    parts.append(f"{content} [/INST] ")
                elif i == 0:
                    parts.append(f"<s>[INST] {content} [/INST] ")
                else:
                    parts.append(f"<s>[INST] {content} [/INST] ")
            elif role == "assistant":
                parts.append(f"{content} </s>")
        
        return "".join(parts)


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    
    # Dataset paths
    raw_data_dir: str = ""
    processed_data_dir: str = ""
    output_path: str = ""
    
    # Processing settings
    max_conversations: int = 10000
    train_split: float = 0.9
    val_split: float = 0.1
    max_tokens_per_conversation: int = 2048
    
    # Filtering
    min_turns: int = 2
    max_turns: int = 10
    
    # Augmentation
    include_system_prompt: bool = True
    shuffle: bool = True
    random_seed: int = 42


class DatasetPreparer:
    """
    Prepares datasets for supervised fine-tuning.
    
    Workflow:
    1. Load raw conversation data
    2. Filter and validate conversations
    3. Format for LLaMA training
    4. Split into train/validation sets
    5. Save processed datasets
    """
    
    # Target diseases
    DISEASES = [
        "glioma",
        "meningioma", 
        "pituitary_tumor",
        "brain_metastases",
        "alzheimer",
        "healthy_brain"
    ]
    
    # Medical disclaimer for training
    SYSTEM_PROMPT = """You are NeuroView Medical Assistant, specialized in providing educational information about neurological conditions. You are NOT a licensed medical professional. Always include appropriate disclaimers and recommend professional consultation. Your responses should be accurate, empathetic, and educational."""
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize dataset preparer.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.conversations: List[MedicalConversation] = []
        
        # Set random seed
        random.seed(config.random_seed)
    
    def load_conversations(self, file_path: Path) -> List[MedicalConversation]:
        """
        Load conversations from a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of MedicalConversation objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversations = []
            for item in data:
                conv = MedicalConversation(
                    id=item.get("id", ""),
                    disease=item.get("disease", ""),
                    conversation=item.get("conversation", []),
                    metadata=item.get("metadata", {})
                )
                conversations.append(conv)
            
            logger.info(f"Loaded {len(conversations)} conversations from {file_path}")
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
            return []
    
    def validate_conversation(self, conv: MedicalConversation) -> bool:
        """
        Validate a conversation for training.
        
        Args:
            conv: Conversation to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check minimum turns
        if len(conv.conversation) < self.config.min_turns * 2:
            return False
        
        # Check maximum turns
        if len(conv.conversation) > self.config.max_turns * 2:
            return False
        
        # Verify alternating user/assistant roles
        expected_role = "user"
        for turn in conv.conversation:
            if turn.get("role") != expected_role:
                return False
            expected_role = "assistant" if expected_role == "user" else "user"
        
        # Check for empty content
        for turn in conv.conversation:
            if not turn.get("content", "").strip():
                return False
        
        # Check disease is valid
        if conv.disease.lower() not in [d.lower() for d in self.DISEASES]:
            logger.warning(f"Unknown disease: {conv.disease}")
        
        return True
    
    def filter_conversations(
        self,
        conversations: List[MedicalConversation]
    ) -> List[MedicalConversation]:
        """
        Filter conversations for quality.
        
        Args:
            conversations: List of conversations
            
        Returns:
            Filtered list of valid conversations
        """
        valid = []
        for conv in conversations:
            if self.validate_conversation(conv):
                valid.append(conv)
        
        logger.info(f"Filtered {len(conversations)} -> {len(valid)} conversations")
        return valid
    
    def format_for_training(
        self,
        conversations: List[MedicalConversation]
    ) -> List[Dict[str, str]]:
        """
        Format conversations for training.
        
        Args:
            conversations: List of conversations
            
        Returns:
            List of formatted training examples
        """
        formatted = []
        system_prompt = self.SYSTEM_PROMPT if self.config.include_system_prompt else ""
        
        for conv in conversations:
            text = conv.to_training_format(system_prompt)
            
            formatted.append({
                "id": conv.id,
                "disease": conv.disease,
                "text": text,
                "num_turns": len(conv.conversation) // 2
            })
        
        return formatted
    
    def split_dataset(
        self,
        data: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Split dataset into train and validation sets.
        
        Args:
            data: Full dataset
            
        Returns:
            Tuple of (train_data, val_data)
        """
        if self.config.shuffle:
            random.shuffle(data)
        
        split_idx = int(len(data) * self.config.train_split)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data
    
    def save_dataset(
        self,
        data: List[Dict[str, str]],
        output_path: Path,
        format: str = "jsonl"
    ) -> None:
        """
        Save processed dataset.
        
        Args:
            data: Dataset to save
            output_path: Output file path
            format: Output format ('jsonl' or 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(data)} examples to {output_path}")
    
    def prepare(self) -> Tuple[Path, Path]:
        """
        Run full dataset preparation pipeline.
        
        Returns:
            Tuple of (train_path, val_path)
        """
        # Load all conversations from raw data directory
        raw_dir = Path(self.config.raw_data_dir)
        
        if raw_dir.exists():
            for json_file in raw_dir.glob("*.json"):
                convs = self.load_conversations(json_file)
                self.conversations.extend(convs)
        
        logger.info(f"Total conversations loaded: {len(self.conversations)}")
        
        # Filter
        self.conversations = self.filter_conversations(self.conversations)
        
        # Limit if needed
        if len(self.conversations) > self.config.max_conversations:
            self.conversations = self.conversations[:self.config.max_conversations]
        
        # Format
        formatted = self.format_for_training(self.conversations)
        
        # Split
        train_data, val_data = self.split_dataset(formatted)
        
        # Save
        output_dir = Path(self.config.processed_data_dir)
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"
        
        self.save_dataset(train_data, train_path, format="jsonl")
        self.save_dataset(val_data, val_path, format="jsonl")
        
        return train_path, val_path
    
    def create_synthetic_example(
        self,
        disease: str,
        example_type: str = "explanation"
    ) -> MedicalConversation:
        """
        Create a synthetic training example.
        
        This is a placeholder for synthetic data generation.
        In production, this would use GPT-4 or similar for high-quality generation.
        
        Args:
            disease: Disease to create example for
            example_type: Type of example ('explanation', 'qa', 'history')
            
        Returns:
            Synthetic conversation
        """
        # Placeholder - actual implementation would generate diverse examples
        conversation = [
            {
                "role": "user",
                "content": f"Can you explain what {disease.replace('_', ' ')} is?"
            },
            {
                "role": "assistant",
                "content": f"I'd be happy to provide educational information about {disease.replace('_', ' ')}. [Placeholder - actual training data would have detailed, medically accurate responses with appropriate disclaimers.]"
            }
        ]
        
        return MedicalConversation(
            id=f"synthetic_{disease}_{example_type}_{random.randint(1000, 9999)}",
            disease=disease,
            conversation=conversation,
            metadata={"synthetic": True, "type": example_type}
        )


class ConversationTemplateGenerator:
    """
    Generates conversation templates for various medical scenarios.
    
    Used to create diverse training data covering:
    - Disease explanations
    - Q&A sessions
    - Clinical discussions
    - Patient education
    """
    
    # Question templates by category
    QUESTION_TEMPLATES = {
        "definition": [
            "What is {disease}?",
            "Can you explain {disease} to me?",
            "What exactly is {disease}?",
            "I was told about {disease}. What does that mean?"
        ],
        "imaging": [
            "How does {disease} appear on MRI?",
            "What are the MRI characteristics of {disease}?",
            "What would a radiologist look for when diagnosing {disease}?",
            "Can you describe the imaging features of {disease}?"
        ],
        "causes": [
            "What causes {disease}?",
            "What are the risk factors for {disease}?",
            "Why do people get {disease}?",
            "What leads to developing {disease}?"
        ],
        "symptoms": [
            "What are the symptoms of {disease}?",
            "How would someone know if they have {disease}?",
            "What signs should I look for with {disease}?",
            "What does {disease} feel like?"
        ],
        "treatment": [
            "How is {disease} treated?",
            "What are the treatment options for {disease}?",
            "What can be done about {disease}?",
            "How do doctors manage {disease}?"
        ],
        "prognosis": [
            "What is the prognosis for {disease}?",
            "What are the outcomes for {disease}?",
            "Is {disease} serious?",
            "What can someone with {disease} expect?"
        ]
    }
    
    @classmethod
    def generate_question(cls, disease: str, category: str) -> str:
        """Generate a question for a disease and category."""
        templates = cls.QUESTION_TEMPLATES.get(category, [])
        if not templates:
            return f"Tell me about {disease}"
        
        template = random.choice(templates)
        return template.format(disease=disease.replace("_", " "))
    
    @classmethod
    def generate_conversation_outline(
        cls,
        disease: str,
        categories: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate a conversation outline (questions only).
        
        Args:
            disease: Disease to discuss
            categories: Categories to cover (defaults to all)
            
        Returns:
            List of questions
        """
        if categories is None:
            categories = list(cls.QUESTION_TEMPLATES.keys())
        
        questions = []
        for category in categories:
            question = cls.generate_question(disease, category)
            questions.append(question)
        
        return questions


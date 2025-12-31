"""
Prompt Templates for NeuroView LLM.

Provides structured templates for various interaction patterns:
- Disease explanation
- Q&A with retrieved context
- Medical history simulation
- Prediction explanation
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Message:
    """Represents a conversation message."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class PromptTemplates:
    """
    Collection of prompt templates for different interaction types.
    """
    
    # Template for RAG-augmented Q&A
    RAG_QA_TEMPLATE = """Based on the following medical reference information, please answer the question.

--- Reference Information ---
{context}
--- End Reference Information ---

User Question: {question}

Please provide an accurate, educational response based on the reference information above. If the reference information doesn't fully address the question, acknowledge what can and cannot be answered from the available information.

Remember to include appropriate medical disclaimers and recommend professional consultation where relevant."""

    # Template for disease explanation
    DISEASE_EXPLANATION_TEMPLATE = """Please provide a comprehensive educational overview of {disease}.

Structure your response as follows:

1. **Definition & Classification**
   - What is {disease}?
   - How is it classified?

2. **MRI Imaging Characteristics**
   - What does {disease} typically look like on MRI?
   - Key imaging features to note

3. **Causes & Risk Factors**
   - Known or suspected causes
   - Risk factors associated with this condition

4. **Clinical Presentation**
   - Common symptoms
   - How it typically presents

5. **General Treatment Approaches**
   - Overview of treatment modalities (educational only)
   - General management principles

6. **Prognosis Considerations**
   - General outcomes information
   - Factors that may influence prognosis

Please include appropriate disclaimers that this information is for educational purposes only."""

    # Template for prediction explanation
    PREDICTION_EXPLANATION_TEMPLATE = """An AI model has analyzed a brain MRI scan and produced the following prediction:

**Predicted Finding:** {disease}
**Model Confidence:** {confidence}
**Location (if available):** {location}

{context}

Please explain:
1. What this prediction means (in accessible terms)
2. Typical imaging features of {disease} that might support this finding
3. Important context about {disease}
4. What steps the person should consider (emphasizing professional consultation)

Remember:
- This is an AI prediction, NOT a medical diagnosis
- Only qualified healthcare providers can make diagnoses
- Frame all information educationally
- Strongly recommend professional medical evaluation"""

    # Template for follow-up questions
    FOLLOW_UP_TEMPLATE = """Based on our discussion about {disease}, I'd like to ask some follow-up questions to provide more relevant educational information.

Previous context:
{previous_context}

Please ask 2-3 relevant follow-up questions that would help provide more tailored educational information. Questions should be:
- Clear and specific
- Answerable with yes/no or brief responses
- Relevant to understanding the user's educational needs

Frame questions as educational exploration, not medical assessment."""

    # Template for medical history simulation (educational)
    HISTORY_SIMULATION_TEMPLATE = """For educational purposes, let's simulate a structured medical history discussion.

This is a DEMONSTRATION of how medical professionals gather clinical information. This is NOT medical advice or diagnosis.

The topic is: {disease}

I will ask structured questions similar to what might be asked in a clinical setting. This helps demonstrate:
- How medical information is gathered
- Why certain questions are clinically relevant
- The logical structure of medical assessment

Please respond to each question as you would in a learning scenario. Remember, this is purely educational.

Let's begin with the first question:
{first_question}"""

    # Template for differential discussion (educational)
    DIFFERENTIAL_TEMPLATE = """For educational purposes, let's discuss the differential considerations for:

**Presenting Information:**
{presentation}

In a typical clinical scenario, what conditions might be considered? Please discuss:

1. **Primary considerations** - Conditions that commonly present this way
2. **Distinguishing features** - What imaging or clinical features help differentiate
3. **Additional information needed** - What other data would help narrow possibilities

This is an educational discussion of general medical concepts, not a diagnosis of any specific case."""

    @classmethod
    def format_rag_qa(cls, question: str, context: str) -> str:
        """Format RAG Q&A template."""
        return cls.RAG_QA_TEMPLATE.format(
            question=question,
            context=context
        )
    
    @classmethod
    def format_disease_explanation(cls, disease: str) -> str:
        """Format disease explanation template."""
        return cls.DISEASE_EXPLANATION_TEMPLATE.format(disease=disease)
    
    @classmethod
    def format_prediction_explanation(
        cls,
        disease: str,
        confidence: float,
        location: Optional[str] = None,
        context: str = ""
    ) -> str:
        """Format prediction explanation template."""
        confidence_str = f"{confidence * 100:.1f}%"
        location_str = location or "Not specified"
        
        return cls.PREDICTION_EXPLANATION_TEMPLATE.format(
            disease=disease,
            confidence=confidence_str,
            location=location_str,
            context=context
        )
    
    @classmethod
    def format_follow_up(
        cls,
        disease: str,
        previous_context: str
    ) -> str:
        """Format follow-up questions template."""
        return cls.FOLLOW_UP_TEMPLATE.format(
            disease=disease,
            previous_context=previous_context
        )


class ConversationManager:
    """
    Manages conversation state and history for multi-turn interactions.
    """
    
    def __init__(
        self,
        system_prompt: str,
        max_history: int = 10
    ):
        """
        Initialize conversation manager.
        
        Args:
            system_prompt: System prompt for the conversation
            max_history: Maximum messages to keep in history
        """
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.messages: List[Message] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_user_message(self, content: str, **metadata) -> None:
        """Add a user message to the conversation."""
        self.messages.append(Message(
            role="user",
            content=content,
            metadata=metadata
        ))
        self._trim_history()
    
    def add_assistant_message(self, content: str, **metadata) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append(Message(
            role="assistant",
            content=content,
            metadata=metadata
        ))
        self._trim_history()
    
    def _trim_history(self) -> None:
        """Trim conversation history to max_history."""
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM."""
        return [msg.to_dict() for msg in self.messages]
    
    def get_full_prompt(self) -> str:
        """Get full prompt with system prompt and history."""
        parts = [f"System: {self.system_prompt}\n"]
        
        for msg in self.messages:
            role = msg.role.capitalize()
            parts.append(f"\n{role}: {msg.content}")
        
        # Add assistant prompt
        parts.append("\nAssistant: ")
        
        return "".join(parts)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.metadata = {}
    
    def set_disease_context(self, disease: str) -> None:
        """Set the current disease context."""
        self.metadata["current_disease"] = disease
    
    def get_disease_context(self) -> Optional[str]:
        """Get the current disease context."""
        return self.metadata.get("current_disease")
    
    def add_rag_context(self, context: str, sources: List[str]) -> None:
        """Add RAG retrieved context to conversation."""
        self.metadata["rag_context"] = context
        self.metadata["rag_sources"] = sources
    
    def get_rag_context(self) -> Optional[str]:
        """Get RAG context if available."""
        return self.metadata.get("rag_context")
    
    def create_contextual_prompt(
        self,
        user_query: str,
        rag_context: Optional[str] = None
    ) -> str:
        """
        Create a contextual prompt with RAG information.
        
        Args:
            user_query: User's question
            rag_context: Retrieved context from RAG
            
        Returns:
            Formatted prompt string
        """
        if rag_context:
            # Use RAG template
            prompt = PromptTemplates.format_rag_qa(user_query, rag_context)
        else:
            # Direct question
            prompt = user_query
        
        return prompt
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "system_prompt": self.system_prompt[:100] + "...",  # Truncate
            "num_messages": len(self.messages),
            "metadata": self.metadata,
            "messages": [msg.to_dict() for msg in self.messages]
        }


class MedicalConversationBuilder:
    """
    Builds structured medical conversations for specific use cases.
    """
    
    @staticmethod
    def create_disease_discussion(
        disease: str,
        system_prompt: str
    ) -> ConversationManager:
        """
        Create a conversation manager for disease discussion.
        
        Args:
            disease: Disease to discuss
            system_prompt: System prompt to use
            
        Returns:
            Configured ConversationManager
        """
        manager = ConversationManager(system_prompt)
        manager.set_disease_context(disease)
        
        # Add initial context-setting message
        initial_prompt = PromptTemplates.format_disease_explanation(disease)
        manager.add_user_message(initial_prompt, intent="disease_overview")
        
        return manager
    
    @staticmethod
    def create_prediction_discussion(
        disease: str,
        confidence: float,
        location: Optional[str],
        system_prompt: str,
        rag_context: str = ""
    ) -> ConversationManager:
        """
        Create a conversation for discussing a prediction.
        
        Args:
            disease: Predicted disease
            confidence: Model confidence
            location: Anatomical location
            system_prompt: System prompt to use
            rag_context: Retrieved medical context
            
        Returns:
            Configured ConversationManager
        """
        manager = ConversationManager(system_prompt)
        manager.set_disease_context(disease)
        
        # Add prediction context
        manager.metadata["prediction"] = {
            "disease": disease,
            "confidence": confidence,
            "location": location
        }
        
        # Add RAG context if available
        if rag_context:
            manager.add_rag_context(rag_context, [])
        
        # Create initial explanation request
        initial_prompt = PromptTemplates.format_prediction_explanation(
            disease=disease,
            confidence=confidence,
            location=location,
            context=rag_context
        )
        manager.add_user_message(initial_prompt, intent="prediction_explanation")
        
        return manager
    
    @staticmethod
    def create_qa_session(
        topic: str,
        system_prompt: str
    ) -> ConversationManager:
        """
        Create a general Q&A session about a topic.
        
        Args:
            topic: Medical topic
            system_prompt: System prompt to use
            
        Returns:
            Configured ConversationManager
        """
        manager = ConversationManager(system_prompt)
        manager.metadata["topic"] = topic
        return manager


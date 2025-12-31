"""
FastAPI Server for NeuroView LLM Service.

Provides REST API endpoints for:
- Medical conversation generation
- Disease explanation
- Prediction integration
- Health checks
"""

import logging
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global instances (initialized at startup)
llm_instance = None
retriever_instance = None


# ============== Request/Response Models ==============

class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat completion."""
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    disease_context: Optional[str] = Field(None, description="Disease context for focused response")
    use_rag: bool = Field(True, description="Whether to use RAG for context retrieval")
    max_tokens: int = Field(1024, description="Maximum tokens to generate")
    temperature: float = Field(0.3, description="Generation temperature")
    stream: bool = Field(False, description="Whether to stream response")


class ChatResponse(BaseModel):
    """Response from chat completion."""
    response: str = Field(..., description="Generated response")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Time taken to generate")
    rag_sources: Optional[List[str]] = Field(None, description="RAG sources used")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DiseaseExplanationRequest(BaseModel):
    """Request for disease explanation."""
    disease: str = Field(..., description="Disease name to explain")
    detail_level: str = Field("standard", description="Detail level: 'brief', 'standard', 'detailed'")


class DiseaseExplanationResponse(BaseModel):
    """Response with disease explanation."""
    disease: str
    explanation: str
    sections: Dict[str, str] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    disclaimer: str


class PredictionIntegrationRequest(BaseModel):
    """Request for integrating with brain scan prediction."""
    disease: str = Field(..., description="Predicted disease from vision model")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    location: Optional[str] = Field(None, description="Predicted anatomical location")
    user_query: Optional[str] = Field(None, description="Optional user question")


class PredictionIntegrationResponse(BaseModel):
    """Response for prediction integration."""
    disease: str
    confidence: float
    explanation: str
    follow_up_questions: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    disclaimer: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    rag_initialized: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_file: str
    backend: str
    context_length: int
    is_loaded: bool


# ============== Application Setup ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global llm_instance, retriever_instance
    
    logger.info("Starting NeuroView LLM Service...")
    
    # Initialize components (lazy loading)
    # Models will be loaded on first request to save startup time
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down NeuroView LLM Service...")
    if llm_instance is not None:
        llm_instance.unload_model()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="NeuroView LLM Service",
        description="Clinical Decision-Support and Educational Conversational Model for Brain MRI Analysis",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# ============== Helper Functions ==============

def get_llm():
    """Get or initialize the LLM instance."""
    global llm_instance
    
    if llm_instance is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.llm_model import NeuroViewLLM
        from config.config import llm_config
        
        llm_instance = NeuroViewLLM(
            model_name=llm_config.model_name,
            model_file=llm_config.model_file,
            context_length=llm_config.context_length,
            n_gpu_layers=llm_config.n_gpu_layers,
            n_batch=llm_config.n_batch,
            n_threads=llm_config.n_threads,
            model_cache_dir=llm_config.model_cache_dir
        )
    
    if not llm_instance.is_loaded:
        success = llm_instance.load_model()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load LLM model")
    
    return llm_instance


def get_retriever():
    """Get or initialize the RAG retriever."""
    global retriever_instance
    
    if retriever_instance is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from rag.embeddings import EmbeddingModel
        from rag.document_store import DocumentStore
        from rag.retriever import MedicalRetriever
        from config.config import rag_config
        
        # Initialize embedding model
        embedding_model = EmbeddingModel(
            model_name=rag_config.embedding_model
        )
        
        # Initialize document store
        doc_store = DocumentStore(
            persist_directory=rag_config.persist_directory,
            collection_name=rag_config.collection_name,
            embedding_model=embedding_model,
            vector_store_type=rag_config.vector_store_type
        )
        
        # Initialize retriever
        retriever_instance = MedicalRetriever(
            document_store=doc_store,
            embedding_model=embedding_model,
            top_k=rag_config.top_k,
            similarity_threshold=rag_config.similarity_threshold
        )
    
    return retriever_instance


# ============== API Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=llm_instance is not None and llm_instance.is_loaded if llm_instance else False,
        rag_initialized=retriever_instance is not None,
        version="0.1.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    try:
        llm = get_llm()
        info = llm.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Generate a chat response.
    
    Supports:
    - Multi-turn conversations
    - RAG-augmented responses
    - Disease-specific context
    """
    try:
        llm = get_llm()
        
        # Get system prompt
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from prompts.system_prompts import MedicalSystemPrompts
        
        if request.disease_context:
            system_prompt = MedicalSystemPrompts.get_disease_specific_prompt(
                request.disease_context
            )
        else:
            system_prompt = MedicalSystemPrompts.MEDICAL_ASSISTANT
        
        # Get RAG context if enabled
        rag_context = ""
        rag_sources = []
        
        if request.use_rag and request.messages:
            try:
                retriever = get_retriever()
                last_user_message = next(
                    (m.content for m in reversed(request.messages) if m.role == "user"),
                    None
                )
                
                if last_user_message:
                    result = retriever.retrieve(
                        query=last_user_message,
                        disease=request.disease_context
                    )
                    rag_context = result.context
                    rag_sources = [doc.metadata.get("source", "") for doc in result.documents]
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Format messages with RAG context
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        if rag_context and messages:
            # Inject RAG context into the last user message
            messages[-1]["content"] = f"""Based on the following medical information:

{rag_context}

User question: {messages[-1]['content']}

Please provide an accurate, educational response based on the reference information."""
        
        # Generate response
        response = llm.chat(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatResponse(
            response=response.text,
            tokens_generated=response.tokens_generated,
            generation_time=response.generation_time,
            rag_sources=rag_sources if rag_sources else None,
            metadata=response.metadata
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/disease", response_model=DiseaseExplanationResponse)
async def explain_disease(request: DiseaseExplanationRequest):
    """
    Get a structured explanation of a neurological disease.
    """
    try:
        llm = get_llm()
        
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from prompts.system_prompts import MedicalSystemPrompts
        from prompts.templates import PromptTemplates
        
        # Get disease-specific prompt
        system_prompt = MedicalSystemPrompts.get_disease_specific_prompt(request.disease)
        
        # Generate explanation request
        user_prompt = PromptTemplates.format_disease_explanation(request.disease)
        
        # Get RAG context
        sources = []
        try:
            retriever = get_retriever()
            result = retriever.get_disease_overview(request.disease)
            rag_context = result.context
            sources = [doc.metadata.get("source", "") for doc in result.documents]
            
            if rag_context:
                user_prompt = f"{user_prompt}\n\nRelevant medical information:\n{rag_context}"
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
        
        # Generate explanation
        response = llm.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.3
        )
        
        return DiseaseExplanationResponse(
            disease=request.disease,
            explanation=response.text,
            sections={},  # Could parse response into sections
            sources=sources,
            disclaimer=MedicalSystemPrompts.DISCLAIMER
        )
        
    except Exception as e:
        logger.error(f"Disease explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/integrate/prediction", response_model=PredictionIntegrationResponse)
async def integrate_prediction(request: PredictionIntegrationRequest):
    """
    Integrate with brain scan prediction from the vision model.
    
    This endpoint receives predictions from the ai-training model
    and generates human-readable explanations.
    """
    try:
        llm = get_llm()
        
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from prompts.system_prompts import MedicalSystemPrompts
        from prompts.templates import PromptTemplates
        
        # Get prediction-specific system prompt
        system_prompt = MedicalSystemPrompts.get_prediction_prompt(
            disease=request.disease,
            confidence=request.confidence,
            location=request.location
        )
        
        # Get RAG context for the predicted disease
        rag_context = ""
        try:
            retriever = get_retriever()
            result = retriever.retrieve_for_prediction(
                disease=request.disease,
                location=request.location,
                confidence=request.confidence
            )
            rag_context = result.context
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
        
        # Format the prompt
        user_prompt = PromptTemplates.format_prediction_explanation(
            disease=request.disease,
            confidence=request.confidence,
            location=request.location,
            context=rag_context
        )
        
        # Add user query if provided
        if request.user_query:
            user_prompt += f"\n\nThe user also asks: {request.user_query}"
        
        # Generate response
        response = llm.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            max_tokens=1500,
            temperature=0.3
        )
        
        # Standard follow-up questions
        follow_up_questions = [
            "Would you like more information about the typical symptoms?",
            "Do you have questions about the imaging characteristics?",
            "Would you like to discuss general treatment approaches?"
        ]
        
        # Standard recommended actions
        recommended_actions = [
            "Consult with a qualified healthcare provider",
            "Discuss these findings with a neurologist or neuro-oncologist",
            "Consider getting a second opinion from a specialist"
        ]
        
        return PredictionIntegrationResponse(
            disease=request.disease,
            confidence=request.confidence,
            explanation=response.text,
            follow_up_questions=follow_up_questions,
            recommended_actions=recommended_actions,
            disclaimer=MedicalSystemPrompts.DISCLAIMER
        )
        
    except Exception as e:
        logger.error(f"Prediction integration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/load")
async def load_model(background_tasks: BackgroundTasks):
    """
    Trigger model loading in background.
    
    Useful for pre-warming the model before first request.
    """
    def load_task():
        try:
            get_llm()
            logger.info("Model loaded successfully via API")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
    
    background_tasks.add_task(load_task)
    return {"message": "Model loading started in background"}


@app.post("/model/unload")
async def unload_model():
    """Unload the model from memory."""
    global llm_instance
    
    if llm_instance is not None:
        llm_instance.unload_model()
        llm_instance = None
        return {"message": "Model unloaded"}
    
    return {"message": "No model loaded"}


@app.get("/diseases")
async def list_diseases():
    """List supported diseases."""
    return {
        "diseases": [
            {"id": "glioma", "name": "Glioma", "description": "Brain tumor from glial cells"},
            {"id": "meningioma", "name": "Meningioma", "description": "Tumor from meninges"},
            {"id": "pituitary_tumor", "name": "Pituitary Tumor", "description": "Tumor of the pituitary gland"},
            {"id": "brain_metastases", "name": "Brain Metastases", "description": "Secondary brain tumors"},
            {"id": "alzheimer", "name": "Alzheimer's Disease", "description": "Neurodegenerative disease"},
            {"id": "healthy_brain", "name": "Healthy Brain", "description": "Normal brain without abnormalities"}
        ]
    }


# ============== Run Server ==============

def run_server(host: str = "0.0.0.0", port: int = 8001):
    """Run the API server."""
    import uvicorn
    
    logger.info(f"Starting NeuroView LLM Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()


"""
NeuroView LLM Service - Main Entry Point

A Clinical Decision-Support and Educational Conversational Model
for Brain MRI Analysis.

Usage:
    # Start API server
    python main.py serve
    
    # Interactive chat mode
    python main.py chat
    
    # Process prediction from vision model
    python main.py predict --disease glioma --confidence 0.85
    
    # Ingest documents for RAG
    python main.py ingest --path ./documents
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.config import llm_config, rag_config, api_config, ensure_directories
from utils.helpers import setup_logging, get_device_info


def serve(args):
    """Start the API server."""
    from api.server import run_server
    
    host = args.host or api_config.host
    port = args.port or api_config.port
    
    logging.info(f"Starting NeuroView LLM API server on {host}:{port}")
    run_server(host=host, port=port)


def chat(args):
    """Start interactive chat mode."""
    from models.llm_model import NeuroViewLLM
    from prompts.system_prompts import MedicalSystemPrompts
    from prompts.templates import ConversationManager
    
    print("\n" + "="*60)
    print("NeuroView LLM - Interactive Medical Assistant")
    print("="*60)
    print("\nLoading model... (this may take a moment)")
    
    # Initialize LLM
    llm = NeuroViewLLM(
        model_name=llm_config.model_name,
        model_file=llm_config.model_file,
        context_length=llm_config.context_length,
        n_gpu_layers=llm_config.n_gpu_layers,
        n_batch=llm_config.n_batch,
        n_threads=llm_config.n_threads,
        model_cache_dir=llm_config.model_cache_dir
    )
    
    if not llm.load_model():
        print("Failed to load model. Please check the logs.")
        return
    
    print("\nModel loaded successfully!")
    print("\nType your questions about neurological conditions.")
    print("Commands: /quit, /clear, /disease <name>, /help")
    print("-"*60 + "\n")
    
    # Initialize conversation
    system_prompt = MedicalSystemPrompts.MEDICAL_ASSISTANT
    conversation = ConversationManager(system_prompt)
    current_disease = None
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()
                
                if cmd[0] == "/quit":
                    print("\nGoodbye!")
                    break
                    
                elif cmd[0] == "/clear":
                    conversation.clear_history()
                    print("Conversation cleared.\n")
                    continue
                    
                elif cmd[0] == "/disease" and len(cmd) > 1:
                    current_disease = cmd[1]
                    system_prompt = MedicalSystemPrompts.get_disease_specific_prompt(current_disease)
                    conversation = ConversationManager(system_prompt)
                    print(f"Disease context set to: {current_disease}\n")
                    continue
                    
                elif cmd[0] == "/help":
                    print("\nCommands:")
                    print("  /quit     - Exit the chat")
                    print("  /clear    - Clear conversation history")
                    print("  /disease <name> - Set disease context")
                    print("  /help     - Show this help\n")
                    continue
            
            # Add user message
            conversation.add_user_message(user_input)
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            
            response = llm.chat(
                messages=conversation.get_messages(),
                system_prompt=system_prompt,
                max_tokens=1024,
                temperature=0.3
            )
            
            print(response.text)
            print(f"\n[{response.tokens_generated} tokens, {response.generation_time:.2f}s]\n")
            
            # Add assistant response to history
            conversation.add_assistant_message(response.text)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
    
    # Cleanup
    llm.unload_model()


def predict(args):
    """Process a prediction from the vision model."""
    from models.llm_model import NeuroViewLLM
    from prompts.system_prompts import MedicalSystemPrompts
    from prompts.templates import PromptTemplates
    
    print(f"\nProcessing prediction: {args.disease} (confidence: {args.confidence})")
    
    # Initialize LLM
    llm = NeuroViewLLM(
        model_name=llm_config.model_name,
        model_file=llm_config.model_file,
        n_gpu_layers=llm_config.n_gpu_layers,
        model_cache_dir=llm_config.model_cache_dir
    )
    
    if not llm.load_model():
        print("Failed to load model.")
        return
    
    # Get prediction-specific prompt
    system_prompt = MedicalSystemPrompts.get_prediction_prompt(
        disease=args.disease,
        confidence=args.confidence,
        location=args.location
    )
    
    # Generate explanation
    user_prompt = PromptTemplates.format_prediction_explanation(
        disease=args.disease,
        confidence=args.confidence,
        location=args.location
    )
    
    response = llm.chat(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=1500
    )
    
    print("\n" + "="*60)
    print("Prediction Explanation")
    print("="*60)
    print(response.text)
    print("\n" + "="*60)
    print(f"[{response.tokens_generated} tokens, {response.generation_time:.2f}s]")
    
    llm.unload_model()


def ingest(args):
    """Ingest documents into the RAG knowledge base."""
    from rag.embeddings import EmbeddingModel
    from rag.document_store import DocumentStore, DocumentProcessor
    
    doc_path = Path(args.path)
    
    if not doc_path.exists():
        print(f"Path not found: {doc_path}")
        return
    
    print(f"Ingesting documents from: {doc_path}")
    
    # Initialize components
    embedding_model = EmbeddingModel(model_name=rag_config.embedding_model)
    doc_store = DocumentStore(
        persist_directory=rag_config.persist_directory,
        collection_name=rag_config.collection_name,
        embedding_model=embedding_model
    )
    processor = DocumentProcessor(
        chunk_size=rag_config.chunk_size,
        chunk_overlap=rag_config.chunk_overlap
    )
    
    # Process files
    total_docs = 0
    
    if doc_path.is_file():
        files = [doc_path]
    else:
        files = list(doc_path.glob("**/*"))
        files = [f for f in files if f.is_file() and f.suffix in ['.txt', '.md', '.json', '.pdf']]
    
    print(f"Found {len(files)} files to process")
    
    for file_path in files:
        print(f"Processing: {file_path.name}")
        
        # Extract disease from path/filename if possible
        disease = None
        for d in rag_config.target_diseases:
            if d.lower() in str(file_path).lower():
                disease = d
                break
        
        documents = processor.process_file(
            file_path,
            metadata={"disease": disease} if disease else {}
        )
        
        if documents:
            doc_store.add_documents(documents)
            total_docs += len(documents)
    
    print(f"\nIngestion complete!")
    print(f"Total documents added: {total_docs}")
    print(f"Total documents in store: {doc_store.get_document_count()}")


def info(args):
    """Show system information."""
    print("\n" + "="*60)
    print("NeuroView LLM Service - System Information")
    print("="*60)
    
    # Device info
    device_info = get_device_info()
    print("\n[Device Information]")
    print(f"  CUDA Available: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        print(f"  GPU: {device_info['cuda_device_name']}")
        print(f"  VRAM Total: {device_info['cuda_memory_total']}")
        print(f"  VRAM Free: {device_info['cuda_memory_free']}")
    print(f"  CPU Cores: {device_info['cpu_count']}")
    print(f"  Recommended Backend: {device_info['recommended_backend']}")
    
    # Model config
    print("\n[Model Configuration]")
    print(f"  Model: {llm_config.model_name}")
    print(f"  Model File: {llm_config.model_file}")
    print(f"  Context Length: {llm_config.context_length}")
    print(f"  GPU Layers: {llm_config.n_gpu_layers}")
    print(f"  Batch Size: {llm_config.n_batch}")
    
    # RAG config
    print("\n[RAG Configuration]")
    print(f"  Embedding Model: {rag_config.embedding_model}")
    print(f"  Vector Store: {rag_config.vector_store_type}")
    print(f"  Top-K Retrieval: {rag_config.top_k}")
    
    # Supported diseases
    print("\n[Supported Diseases]")
    for disease in rag_config.target_diseases:
        print(f"  - {disease.replace('_', ' ').title()}")
    
    print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NeuroView LLM Service - Medical Conversational AI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", type=str, default=None, help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Process a prediction")
    predict_parser.add_argument("--disease", type=str, required=True, help="Predicted disease")
    predict_parser.add_argument("--confidence", type=float, required=True, help="Confidence score")
    predict_parser.add_argument("--location", type=str, default=None, help="Anatomical location")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents for RAG")
    ingest_parser.add_argument("--path", type=str, required=True, help="Path to documents")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Ensure directories exist
    ensure_directories()
    
    # Execute command
    if args.command == "serve":
        serve(args)
    elif args.command == "chat":
        chat(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "ingest":
        ingest(args)
    elif args.command == "info":
        info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


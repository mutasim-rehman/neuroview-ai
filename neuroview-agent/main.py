"""
NeuroView Agent - Main Entry Point

Medical AI Agent for Brain MRI Analysis

Usage:
    python main.py serve     # Start API server
    python main.py chat      # Interactive chat
    python main.py info      # Show system info
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import config, ensure_directories


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def serve(args):
    """Start the API server."""
    from api.server import run_server
    
    host = args.host or "0.0.0.0"
    port = args.port or 8002
    
    logging.info(f"Starting NeuroView Agent API on {host}:{port}")
    run_server(host=host, port=port)


def chat(args):
    """Interactive chat mode."""
    print("\n" + "="*60)
    print("NeuroView Agent - Interactive Mode")
    print("="*60)
    print("\nSKELETON MODE - Agent not fully implemented yet")
    print("\nThis will be an interactive chat with the medical agent.")
    print("Type /quit to exit, /help for commands")
    print("-"*60 + "\n")
    
    # TODO: Implement actual chat loop
    # from agent import NeuroViewAgent
    # from llm import LocalLLM
    # from tools import get_all_tools
    #
    # llm = LocalLLM(...)
    # llm.load()
    # agent = NeuroViewAgent(llm=llm, tools=get_all_tools())
    #
    # while True:
    #     user_input = input("You: ")
    #     if user_input == "/quit":
    #         break
    #     response = agent.run(user_input)
    #     print(f"\nAgent: {response.answer}\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "/quit":
                print("\nGoodbye!")
                break
            
            if user_input.lower() == "/help":
                print("\nCommands:")
                print("  /quit - Exit")
                print("  /help - Show this help")
                print("  /tools - List available tools")
                print()
                continue
            
            if user_input.lower() == "/tools":
                print("\nAvailable tools:")
                print("  - web_search: Search the web")
                print("  - pubmed_search: Search medical literature")
                print("  - wikipedia: Get Wikipedia summaries")
                print("  - medlineplus: Patient-friendly health info")
                print("  - drug_lookup: Drug information")
                print("  - vision_model: Brain scan analysis")
                print("  - medical_calc: Medical calculations")
                print()
                continue
            
            # SKELETON response
            print(f"\n[SKELETON] Agent would process: {user_input}")
            print("[SKELETON] Would use tools and provide response")
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break


def info(args):
    """Show system information."""
    print("\n" + "="*60)
    print("NeuroView Agent - System Information")
    print("="*60)
    
    print("\n[Configuration]")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Max tool retries: {config.max_tool_retries}")
    
    print("\n[LLM Configuration]")
    print(f"  Model: {config.llm.model_name}")
    print(f"  Model file: {config.llm.model_file}")
    print(f"  Context length: {config.llm.context_length}")
    print(f"  GPU layers: {config.llm.n_gpu_layers}")
    
    print("\n[Tools Configuration]")
    print(f"  Vision model URL: {config.tools.vision_model_url}")
    print(f"  PubMed max results: {config.tools.pubmed_max_results}")
    
    print("\n[Available Tools]")
    tools = [
        "web_search", "pubmed_search", "wikipedia",
        "medlineplus", "drug_lookup", "vision_model", "medical_calc"
    ]
    for tool in tools:
        print(f"  - {tool}")
    
    print("\n[Supported Diseases]")
    diseases = ["Glioma", "Meningioma", "Pituitary Tumor", 
                "Brain Metastases", "Alzheimer's", "Healthy Brain"]
    for d in diseases:
        print(f"  - {d}")
    
    print("\n" + "="*60)
    print("Status: SKELETON - Core structure ready, implementation pending")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NeuroView Agent - Medical AI for Brain MRI Analysis"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", type=str, default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="System information")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    ensure_directories()
    
    # Execute
    if args.command == "serve":
        serve(args)
    elif args.command == "chat":
        chat(args)
    elif args.command == "info":
        info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


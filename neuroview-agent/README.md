# NeuroView Agent

**Medical AI Agent for Brain MRI Analysis**

## Overview

NeuroView Agent is an AI agent that helps users understand neurological conditions and brain MRI scan predictions. Unlike RAG systems that rely on stored documents, this agent dynamically retrieves information from public medical APIs.

> ⚠️ **DISCLAIMER**: This system is for **educational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   NeuroView Agent                       │
├─────────────────────────────────────────────────────────┤
│  User Query                                             │
│      ↓                                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │              ReAct Loop                          │   │
│  │  THINK → ACT → OBSERVE → (repeat) → ANSWER      │   │
│  └─────────────────────────────────────────────────┘   │
│      ↓                                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │                    Tools                         │   │
│  │  • Web Search (DuckDuckGo)                      │   │
│  │  • PubMed Search (NCBI API)                     │   │
│  │  • Wikipedia                                     │   │
│  │  • MedlinePlus (NIH)                            │   │
│  │  • Drug Lookup (OpenFDA)                        │   │
│  │  • Vision Model (ai-training)                   │   │
│  │  • Medical Calculator                           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Key Features

- **ReAct Pattern**: Think → Act → Observe reasoning loop
- **Public APIs Only**: No copyrighted documents, legal and free
- **Local LLM**: Runs on your hardware (Mistral-7B / LLaMA)
- **Multi-Tool**: 7 specialized tools for medical information
- **Source Citations**: All information is cited
- **Safety First**: Medical disclaimers in all responses

## Hardware Requirements

Optimized for:
- **GPU**: RTX 4060 6GB VRAM (or similar)
- **RAM**: 16GB
- **CPU**: Ryzen 5

## Project Structure

```
neuroview-agent/
├── agent/              # Core agent logic
│   ├── core.py         # ReAct agent implementation
│   ├── executor.py     # Tool execution with retries
│   └── planner.py      # Tool selection logic
├── llm/                # LLM wrapper
│   ├── model.py        # Local LLM (llama-cpp)
│   └── function_calling.py  # Tool call parsing
├── tools/              # Agent tools
│   ├── base.py         # Base tool class
│   ├── web_search.py   # DuckDuckGo search
│   ├── pubmed.py       # PubMed API
│   ├── wikipedia.py    # Wikipedia API
│   ├── medlineplus.py  # MedlinePlus API
│   ├── drug_lookup.py  # OpenFDA API
│   ├── medical_calc.py # Medical calculators
│   └── vision_model.py # Brain scan integration
├── memory/             # Conversation memory
│   ├── conversation.py # Chat history
│   ├── working_memory.py # Tool results
│   └── summarizer.py   # Context compression
├── prompts/            # System prompts
│   ├── system.py       # Agent prompts
│   ├── tools.py        # Tool descriptions
│   └── safety.py       # Medical disclaimers
├── api/                # FastAPI server
│   └── server.py       # REST endpoints
├── config/             # Configuration
│   └── config.py       # All settings
├── main.py             # Entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
cd neuroview-agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install llama-cpp with CUDA
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

## Usage

### System Info
```bash
python main.py info
```

### Interactive Chat
```bash
python main.py chat
```

### API Server
```bash
python main.py serve --port 8002
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Chat with agent |
| `/explain/prediction` | POST | Explain scan prediction |
| `/tools` | GET | List available tools |
| `/diseases` | GET | List supported diseases |

### Example: Chat

```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is glioma?"}'
```

### Example: Explain Prediction

```bash
curl -X POST http://localhost:8002/explain/prediction \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "glioma",
    "confidence": 0.87,
    "location": "frontal lobe"
  }'
```

## Available Tools

| Tool | Source | Use Case |
|------|--------|----------|
| `web_search` | DuckDuckGo | General queries, current info |
| `pubmed_search` | NCBI | Research articles, clinical studies |
| `wikipedia` | Wikipedia | Disease overviews, definitions |
| `medlineplus` | NIH | Patient-friendly explanations |
| `drug_lookup` | OpenFDA | Drug information |
| `vision_model` | ai-training | Brain scan predictions |
| `medical_calc` | Local | BMI, GCS, unit conversions |

## Supported Diseases

- Glioma (glioblastoma, astrocytoma, etc.)
- Meningioma
- Pituitary Tumors
- Brain Metastases
- Alzheimer's Disease
- Normal/Healthy Brain

## Configuration

Edit `config/config.py`:

```python
# LLM Settings
model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
n_gpu_layers = 35  # Adjust for your VRAM

# Agent Settings
max_iterations = 10  # Max reasoning steps
```

## Current Status

**SKELETON** - Core structure complete, implementation pending:

- [x] Project structure
- [x] Configuration
- [x] Tool definitions
- [x] Agent core (ReAct)
- [x] Memory system
- [x] Prompts
- [x] API server
- [ ] LLM integration (needs model download)
- [ ] Tool implementations (needs API testing)
- [ ] Full agent loop

## Integration with NeuroView

```
MRI Upload → ai-training (Vision) → Prediction
                                        ↓
                               neuroview-agent
                                        ↓
                               Explanation + Discussion
                                        ↓
                                    Frontend
```

## Safety

- All responses include medical disclaimers
- Never provides diagnoses or treatment recommendations
- Always recommends professional consultation
- Uses low temperature for consistent responses
- Cites all sources

---

**Note**: This is educational software. Always consult qualified healthcare professionals for medical concerns.


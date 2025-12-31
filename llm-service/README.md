# NeuroView LLM Service

**A Clinical Decision-Support and Educational Conversational Model for Brain MRI Analysis**

## Overview

NeuroView LLM is a domain-specific large language model system designed to assist in the interpretation, explanation, and discussion of neurological diseases, particularly those identified through brain MRI scans.

The system integrates with the NeuroView brain scan prediction pipeline to provide:
- Educational information about neurological conditions
- Explanation of AI-generated predictions
- Structured medical discussions
- RAG-enhanced responses with medical knowledge

> ⚠️ **IMPORTANT**: This system is for **educational purposes only** and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.

## Hardware Requirements

This project is optimized for:
- **GPU**: NVIDIA RTX 4060 6GB VRAM (or similar)
- **RAM**: 16GB
- **CPU**: AMD Ryzen 5 (or Intel equivalent)

The system uses **4-bit quantized LLaMA models** to fit within 6GB VRAM constraints.

## Supported Diseases

Initial implementation covers:
- Glioma (including glioblastoma, astrocytoma)
- Meningioma
- Pituitary Tumors
- Brain Metastases
- Alzheimer's Disease
- Normal/Healthy Brain

## Project Structure

```
llm-service/
├── api/                    # FastAPI server
│   ├── __init__.py
│   └── server.py
├── config/                 # Configuration
│   ├── __init__.py
│   └── config.py
├── fine_tuning/           # Supervised Fine-Tuning (skeleton)
│   ├── __init__.py
│   ├── dataset_preparation.py
│   └── trainer.py
├── models/                # LLM model wrapper
│   ├── __init__.py
│   └── llm_model.py
├── prompts/               # System prompts & templates
│   ├── __init__.py
│   ├── system_prompts.py
│   └── templates.py
├── rag/                   # Retrieval-Augmented Generation
│   ├── __init__.py
│   ├── document_store.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── knowledge_base/    # Medical documents go here
├── utils/                 # Utilities
│   ├── __init__.py
│   └── helpers.py
├── checkpoints/           # Model checkpoints
├── logs/                  # Log files
├── main.py               # Main entry point
├── requirements.txt
└── README.md
```

## Installation

### 1. Create Virtual Environment

```bash
cd llm-service
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install base dependencies
pip install -r requirements.txt

# Install llama-cpp-python with CUDA support (for GPU acceleration)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### 3. Download Model

The model will be downloaded automatically on first run, but you can pre-download:

```bash
# The default model is LLaMA-2-7B-Chat (4-bit quantized)
# It will be downloaded from HuggingFace Hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('TheBloke/Llama-2-7B-Chat-GGUF', 'llama-2-7b-chat.Q4_K_M.gguf')"
```

## Usage

### System Information

```bash
python main.py info
```

### Interactive Chat Mode

```bash
python main.py chat
```

Commands in chat mode:
- `/quit` - Exit
- `/clear` - Clear conversation history
- `/disease <name>` - Set disease context
- `/help` - Show help

### API Server

```bash
# Start the server (default: http://localhost:8001)
python main.py serve

# With custom host/port
python main.py serve --host 0.0.0.0 --port 8001
```

### Process Prediction

Integrate with the brain scan prediction model:

```bash
python main.py predict --disease glioma --confidence 0.85 --location "frontal lobe"
```

### Ingest Documents for RAG

Add medical documents to the knowledge base:

```bash
python main.py ingest --path ./medical_documents
```

Supported formats: `.txt`, `.md`, `.json`, `.pdf`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/chat` | POST | Chat completion |
| `/explain/disease` | POST | Disease explanation |
| `/integrate/prediction` | POST | Integrate with vision model |
| `/diseases` | GET | List supported diseases |
| `/model/load` | POST | Load model (background) |
| `/model/unload` | POST | Unload model |

### Example: Chat API

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is glioma?"}
    ],
    "disease_context": "glioma",
    "use_rag": true
  }'
```

### Example: Prediction Integration

```bash
curl -X POST http://localhost:8001/integrate/prediction \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "meningioma",
    "confidence": 0.92,
    "location": "right frontal convexity"
  }'
```

## Model Configuration

Default model: **LLaMA-2-7B-Chat (4-bit GGUF)**

Modify in `config/config.py`:

```python
@dataclass
class LLMConfig:
    model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_file: str = "llama-2-7b-chat.Q4_K_M.gguf"
    context_length: int = 4096
    n_gpu_layers: int = 35  # Adjust based on VRAM
```

### Alternative Models

For lower VRAM usage:
```python
model_name = "lmstudio-community/Llama-3.2-3B-Instruct-GGUF"
model_file = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

## RAG Knowledge Base

Place medical documents in `rag/knowledge_base/documents/`:

1. **Organize by disease** (recommended):
   ```
   knowledge_base/documents/
   ├── glioma/
   │   ├── overview.md
   │   └── imaging_features.txt
   ├── meningioma/
   └── ...
   ```

2. **Run ingestion**:
   ```bash
   python main.py ingest --path ./rag/knowledge_base/documents
   ```

## Fine-Tuning (Skeleton)

The fine-tuning module is a skeleton placeholder. To implement:

1. Prepare dataset in `fine_tuning/datasets/`
2. Configure training in `config/config.py`
3. Complete implementation in `fine_tuning/trainer.py`

Requirements for fine-tuning:
- Additional VRAM (may need to reduce batch size)
- Install optional dependencies:
  ```bash
  pip install torch bitsandbytes peft datasets accelerate trl
  ```

## Integration with NeuroView Frontend

The LLM service integrates with the main NeuroView application:

1. **Vision Model** (ai-training) → Predicts disease from MRI
2. **LLM Service** (this project) → Explains prediction
3. **Frontend** → Displays results to user

Typical workflow:
```
MRI Upload → Vision Model → Prediction → LLM Service → Explanation → User
```

## Safety & Ethics

This system implements:
- **Non-diagnostic language** - Never makes definitive diagnoses
- **Educational disclaimers** - All responses include disclaimers
- **Professional referral** - Encourages consulting healthcare providers
- **Accuracy focus** - Low temperature (0.3) for consistent responses

## Troubleshooting

### CUDA Out of Memory
- Reduce `n_gpu_layers` in config
- Use a smaller model (3B instead of 7B)
- Reduce `context_length`

### Slow Generation
- Increase `n_gpu_layers` if VRAM allows
- Increase `n_batch` for faster prompt processing
- Check GPU utilization with `nvidia-smi`

### Model Download Failed
- Check internet connection
- Verify HuggingFace Hub is accessible
- Try manual download with `huggingface-cli`

## Development

```bash
# Run tests
pytest tests/

# Format code
black .
isort .

# Check types
mypy llm_service/
```

## License

This project is part of the NeuroView system for educational purposes.

---

**Disclaimer**: This AI system is designed for educational purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.


# Architecture Overview

## 🍰 Three-Layer Cake Architecture

NeuroView AI is designed as a **3-layer cake architecture**, where each layer handles a distinct responsibility in the medical imaging analysis pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   LAYER 3: LLM Health Assistant                              🚧 WIP    │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │  Llama 3 7B + RAG + Supervised Fine-Tuning                    │    │
│   │  • Medical history collection & analysis                      │    │
│   │  • Likely cause detection from symptoms + scan results        │    │
│   │  • Future medical pathway recommendations                     │    │
│   └───────────────────────────────────────────────────────────────┘    │
│                                    │                                    │
├────────────────────────────────────┼────────────────────────────────────┤
│                                    ▼                                    │
│   LAYER 2: Prediction Model                                  ✅ Done   │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │  3D CNN Deep Learning (PyTorch)                               │    │
│   │  • Trained on 582 healthy brain T1 scans                      │    │
│   │  • Anomaly detection via reconstruction error                 │    │
│   │  • Confidence scoring & error metrics                         │    │
│   └───────────────────────────────────────────────────────────────┘    │
│                                    │                                    │
├────────────────────────────────────┼────────────────────────────────────┤
│                                    ▼                                    │
│   LAYER 1: Visualization                                     ✅ Done   │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │  Vite + React + Three.js                                      │    │
│   │  • 3D volume rendering with ray marching                      │    │
│   │  • Multi-planar views (Axial, Sagittal, Coronal)             │    │
│   │  • NIfTI parsing, 4D support, measurement tools               │    │
│   └───────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer Summary

| Layer | Technology Stack | Responsibility | Status |
|-------|------------------|----------------|--------|
| **1. Visualization** | Vite, React, Three.js, WebGL | 3D medical imaging display | ✅ Complete |
| **2. Prediction** | PyTorch, 3D CNN | Brain scan anomaly detection | ✅ Complete |
| **3. LLM Assistant** | Llama 3 7B, RAG, SFT | Health inspection & recommendations | 🚧 In Progress |

---

## Layer 3: LLM Health Assistant (In Development)

The upcoming intelligent health assistant layer powered by **Llama 3 7B** will provide:

### Core Features
- **Medical History Collection**: Interactive conversation to gather patient medical history
- **Likely Cause Detection**: Analyzes scan results combined with symptoms to identify potential causes
- **Medical Pathway Recommendations**: Suggests future medical steps, follow-ups, and specialist referrals

### Technical Implementation
- **RAG (Retrieval-Augmented Generation)**: Retrieves relevant medical knowledge from curated databases
- **Supervised Fine-Tuning (SFT)**: Model fine-tuned on medical domain data for specialized health insights
- **Integration**: Receives prediction results from Layer 2 and provides contextual analysis

### Planned API Endpoints
```
POST /chat          - Interactive health consultation
POST /analyze       - Combined scan + history analysis
GET  /history/:id   - Retrieve conversation history
```

---

## System Architecture

NeuroView AI follows a **completely decoupled architecture** where the frontend and backend are independent services that communicate only through HTTP API calls.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (Vercel)                            │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  React/TypeScript Application                              │ │
│  │  - NIfTI file parsing (client-side)                       │ │
│  │  - 3D visualization (Three.js)              [LAYER 1]     │ │
│  │  - UI components                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          │                                      │
│                          │ HTTP API Calls                       │
│                          │ (fetch/axios)                        │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  brainHealthService.ts                                     │ │
│  │  - checkApiHealth()                                        │ │
│  │  - predictFromFile()                        [LAYER 2]     │ │
│  │  - predictFromVolume()                                     │ │
│  │  - predictFromVolumeData()                                 │ │
│  │                                                             │ │
│  │  ⚠️ NO direct model access                                 │ │
│  │  ⚠️ NO PyTorch imports                                     │ │
│  │  ⚠️ NO model file references                               │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                          │
                          │ HTTPS
                          │
┌─────────────────────────────────────────────────────────────────┐
│                    Backend API (Render)                         │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Flask API Server (api_server.py)                          │ │
│  │  - /health endpoint                                        │ │
│  │  - /predict endpoint                        [LAYER 2]     │ │
│  │  - /predict_from_array endpoint                            │ │
│  │  - /chat (coming soon)                      [LAYER 3]     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          │                                      │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  AI Models                                                  │ │
│  │  ├─ brain_model.py (3D CNN)                 [LAYER 2]     │ │
│  │  │  └─ checkpoints/best_model.pth                          │ │
│  │  │                                                          │ │
│  │  └─ llm_assistant.py (Coming Soon)          [LAYER 3]     │ │
│  │     ├─ Llama 3 7B base model                               │ │
│  │     ├─ RAG knowledge retrieval                             │ │
│  │     └─ SFT medical fine-tuning                             │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Key Principles

### 1. Complete Separation

- **Frontend** has zero knowledge of:
  - Model architecture
  - PyTorch or any ML framework
  - Model checkpoint files
  - Backend implementation details

- **Backend** has zero knowledge of:
  - Frontend UI components
  - React/TypeScript code
  - Frontend state management

### 2. API-Only Communication

All communication between frontend and backend happens through:
- **HTTP REST API** (JSON)
- **Standard fetch API** (no special libraries)
- **Environment variable** for API URL (`VITE_API_URL`)

### 3. Independent Deployment

- **Frontend** deploys to Vercel (static hosting)
  - No server-side code
  - No model files
  - Fast CDN distribution

- **Backend** deploys to Render (Python service)
  - Contains model files
  - Handles inference
  - Can scale independently

## File Structure

```
neuroview-ai/
├── Frontend (Vercel)                    # Deployed separately
│   ├── components/                     # React components
│   ├── services/
│   │   └── brainHealthService.ts      # ⚠️ API client only
│   ├── utils/                          # Frontend utilities
│   └── App.tsx                         # Main app
│
└── ai-training/                        # Backend (Render)
    ├── api_server.py                   # Flask API server
    ├── models/
    │   ├── brain_model.py             # 3D CNN model [LAYER 2]
    │   └── llm_assistant.py           # LLM model (coming) [LAYER 3]
    ├── checkpoints/
    │   └── best_model.pth             # Trained CNN model
    ├── knowledge_base/                 # RAG documents (coming) [LAYER 3]
    └── requirements.txt                # Python dependencies
```

## API Contract

### Current Endpoints (Layers 1 & 2)

#### `GET /health`
Check if API is running and model is loaded.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

#### `POST /predict`
Predict from uploaded NIfTI file.

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "prediction": "healthy",
  "confidence": 0.85,
  "anomaly_score": 0.008,
  "error_metrics": {
    "mse": 0.008,
    "mae": 0.045,
    "max_error": 0.123
  }
}
```

#### `POST /predict_from_array`
Predict from volume data array.

**Request:**
```json
{
  "volume": [[[...]]],
  "shape": [128, 128, 128]
}
```

**Response:** Same as `/predict`

### Planned Endpoints (Layer 3)

#### `POST /chat` (Coming Soon)
Interactive health consultation with LLM.

**Request:**
```json
{
  "message": "What could cause these anomalies?",
  "scan_results": { ... },
  "medical_history": { ... }
}
```

**Response:**
```json
{
  "response": "Based on the scan results and your medical history...",
  "likely_causes": ["...", "..."],
  "recommended_actions": ["...", "..."],
  "confidence": 0.78
}
```

## Environment Variables

### Frontend (Vercel)
- `VITE_API_URL` - Backend API URL (required)
- `GEMINI_API_KEY` - Google Gemini API key (optional, for AI analysis)

### Backend (Render)
- `PORT` - Server port (auto-set by Render)
- `PYTHON_VERSION` - Python version (3.10)
- `LLAMA_MODEL_PATH` - Path to Llama 3 model (Layer 3, coming soon)

## Benefits of This Architecture

### 1. **Scalability**
- Frontend and backend can scale independently
- Backend can handle multiple frontend clients
- Can add mobile apps, CLI tools, etc. using same API

### 2. **Security**
- Model files never exposed to frontend
- API can implement authentication/rate limiting
- Backend can validate and sanitize inputs

### 3. **Development**
- Teams can work independently
- Frontend developers don't need Python/ML knowledge
- Backend developers don't need React knowledge

### 4. **Deployment**
- Frontend: Fast static hosting (Vercel CDN)
- Backend: Can use GPU instances if needed
- Can swap hosting providers independently

### 5. **Maintenance**
- Update model without frontend changes
- Update frontend UI without backend changes
- Clear separation of concerns

## Current Progress

| Component | Status | Notes |
|-----------|--------|-------|
| Layer 1: Visualization | ✅ Complete | Vite + Three.js working |
| Layer 2: CNN Prediction | ✅ Complete | Trained on 582 brain scans |
| Layer 3: LLM Assistant | 🚧 In Progress | Llama 3 7B + RAG + SFT |

---

**This architecture ensures complete independence between frontend and backend while maintaining a clean, maintainable codebase.**

# Architecture Overview

## System Architecture

NeuroView AI follows a **completely decoupled architecture** where the frontend and backend are independent services that communicate only through HTTP API calls.

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Vercel)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  React/TypeScript Application                         │  │
│  │  - NIfTI file parsing (client-side)                  │  │
│  │  - 3D visualization (Three.js)                      │  │
│  │  - UI components                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          │ HTTP API Calls                    │
│                          │ (fetch/axios)                    │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  brainHealthService.ts                                │  │
│  │  - checkApiHealth()                                   │  │
│  │  - predictFromFile()                                  │  │
│  │  - predictFromVolume()                                │  │
│  │  - predictFromVolumeData()                            │  │
│  │                                                        │  │
│  │  ⚠️ NO direct model access                            │  │
│  │  ⚠️ NO PyTorch imports                                │  │
│  │  ⚠️ NO model file references                          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ HTTPS
                          │
┌─────────────────────────────────────────────────────────────┐
│                    Backend API (Render)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Flask API Server (api_server.py)                    │  │
│  │  - /health endpoint                                   │  │
│  │  - /predict endpoint                                  │  │
│  │  - /predict_from_array endpoint                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AI Model (PyTorch)                                    │  │
│  │  - brain_model.py                                     │  │
│  │  - checkpoints/best_model.pth                        │  │
│  │  - Inference only (no training)                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
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
    │   └── brain_model.py             # Model architecture
    ├── checkpoints/
    │   └── best_model.pth             # Trained model
    └── requirements.txt                # Python dependencies
```

## API Contract

### Endpoints

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
  "volume": [[[...]]],  // 3D nested array
  "shape": [128, 128, 128]
}
```

**Response:** Same as `/predict`

## Environment Variables

### Frontend (Vercel)
- `VITE_API_URL` - Backend API URL (required)
- `GEMINI_API_KEY` - Google Gemini API key (optional, for AI analysis)

### Backend (Render)
- `PORT` - Server port (auto-set by Render)
- `PYTHON_VERSION` - Python version (3.10)

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

## Verification

To verify the frontend has no direct model access:

```bash
# Search for any model imports in frontend
grep -r "import.*model\|from.*model\|\.pth\|torch\|pytorch" --include="*.ts" --include="*.tsx" .

# Should only find references in:
# - Documentation files (.md)
# - Service files (API calls only)
# - No actual model imports
```

## Migration Path

If you need to add new model features:

1. **Backend:** Add new endpoint in `api_server.py`
2. **Backend:** Implement model logic
3. **Frontend:** Add new function in `brainHealthService.ts`
4. **Frontend:** Update UI to call new function
5. **Deploy:** Both services independently

No need to coordinate deployments or share code!

---

**This architecture ensures complete independence between frontend and backend while maintaining a clean, maintainable codebase.**


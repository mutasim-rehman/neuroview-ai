# Deployment Files Summary

This document explains all the deployment-related files in the repository.

## Deployment Configuration Files

### 1. `render.yaml`
**Purpose:** Render.com blueprint configuration  
**Location:** Root directory  
**Usage:** 
- Import in Render dashboard: "Apply Render Blueprint"
- Or manually configure using the settings in this file
- Defines service type, build commands, start commands, and environment variables

**Key Settings:**
- Service type: Web service
- Build: `cd ai-training && pip install -r requirements.txt`
- Start: `cd ai-training && gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --worker-class sync api_server:app`
- Health check: `/health`

---

### 2. `Procfile`
**Purpose:** Railway.app start command  
**Location:** Root directory  
**Usage:**
- Railway automatically detects and uses this file
- Defines the web process command
- Format: `web: <command>`

**Content:**
```
web: cd ai-training && gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --worker-class sync api_server:app
```

---

### 3. `ai-training/runtime.txt`
**Purpose:** Explicit Python version specification (optional)  
**Location:** `ai-training/` directory  
**Usage:**
- Some platforms use this to determine Python version
- Format: `python-<version>`
- Current: `python-3.10.12`

**Note:** Not required for Render or Railway (they use environment variables), but included for compatibility.

---

## Documentation Files

### 4. `DEPLOYMENT_GUIDE.md`
**Purpose:** Comprehensive deployment guide  
**Location:** Root directory  
**Contents:**
- Step-by-step instructions for Render
- Step-by-step instructions for Railway
- Post-deployment testing
- Troubleshooting guide
- Platform comparison

**Use this for:** Detailed deployment walkthrough

---

### 5. `DEPLOYMENT_QUICK_REFERENCE.md`
**Purpose:** Quick reference for deployment commands  
**Location:** Root directory  
**Contents:**
- Quick deploy commands
- Gunicorn configuration options
- Testing commands
- Common issues and solutions
- Deployment checklist

**Use this for:** Quick lookups and common tasks

---

### 6. `DEPLOYMENT_FILES_SUMMARY.md` (this file)
**Purpose:** Overview of all deployment files  
**Location:** Root directory  
**Contents:**
- Explanation of each deployment file
- Where to find what you need

---

## Application Files

### 7. `ai-training/api_server.py`
**Purpose:** Main Flask API server  
**Location:** `ai-training/` directory  
**Key Features:**
- Flask application with CORS enabled
- Model loading on startup
- Health check endpoint (`/health`)
- Prediction endpoints (`/predict`, `/predict_from_array`)
- Uses `PORT` environment variable

---

### 8. `ai-training/requirements.txt`
**Purpose:** Python dependencies  
**Location:** `ai-training/` directory  
**Key Dependencies:**
- Flask & Flask-CORS
- PyTorch
- Gunicorn (for production)
- Nibabel, NumPy, SciPy
- Other ML/data processing libraries

---

### 9. `ai-training/checkpoints/best_model.pth`
**Purpose:** Trained model checkpoint  
**Location:** `ai-training/checkpoints/` directory  
**Required:** Yes - API won't work without this file  
**Status:** Should be committed to git for deployment

---

## Deployment Workflow

### For Render:
1. Push code to GitHub
2. In Render dashboard:
   - Option A: Use "Apply Render Blueprint" → Select `render.yaml`
   - Option B: Manually configure using settings from `render.yaml`
3. Render builds and deploys automatically

### For Railway:
1. Push code to GitHub
2. In Railway dashboard:
   - Create new project from GitHub repo
   - Railway auto-detects Python and uses `Procfile`
   - Or manually set build/start commands
3. Railway builds and deploys automatically

---

## File Dependencies

```
Deployment Flow:
├── render.yaml (Render config)
├── Procfile (Railway config)
├── ai-training/
│   ├── api_server.py (main app)
│   ├── requirements.txt (dependencies)
│   ├── runtime.txt (Python version - optional)
│   ├── checkpoints/
│   │   └── best_model.pth (required model)
│   ├── config/
│   │   └── config.py (app config)
│   ├── models/
│   │   └── brain_model.py (model architecture)
│   └── data/
│       └── preprocessing.py (data processing)
└── Documentation files
    ├── DEPLOYMENT_GUIDE.md
    ├── DEPLOYMENT_QUICK_REFERENCE.md
    └── DEPLOYMENT_FILES_SUMMARY.md
```

---

## Quick Start

**Want to deploy to Render?**
→ Read `DEPLOYMENT_GUIDE.md` → Render section  
→ Or use `render.yaml` blueprint in Render dashboard

**Want to deploy to Railway?**
→ Read `DEPLOYMENT_GUIDE.md` → Railway section  
→ Railway will auto-detect `Procfile`

**Need quick commands?**
→ Check `DEPLOYMENT_QUICK_REFERENCE.md`

**Having issues?**
→ Check troubleshooting in `DEPLOYMENT_GUIDE.md`  
→ Review logs in platform dashboard

---

## Notes

- All deployment files are in the root directory or `ai-training/` directory
- Model files must be committed to git for deployment
- Both platforms support auto-deploy on git push
- Environment variables are set in platform dashboards
- Health check endpoint: `/health`
- API runs on port specified by `$PORT` environment variable


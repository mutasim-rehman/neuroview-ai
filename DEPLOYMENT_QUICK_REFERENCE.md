# Deployment Quick Reference

Quick commands and configurations for deploying the NeuroView AI API.

## Render Quick Deploy

### Using Render Dashboard:
1. Go to [render.com](https://render.com) → New Web Service
2. Connect GitHub repo
3. Use these settings:

**Build Command:**
```bash
cd ai-training && pip install -r requirements.txt
```

**Start Command:**
```bash
cd ai-training && gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --worker-class sync api_server:app
```

**Environment Variables:**
- `PORT` = `10000`
- `PYTHON_VERSION` = `3.10`

**Health Check:** `/health`

### Using render.yaml (Blueprint):
```bash
# In Render dashboard, select "Apply Render Blueprint"
# render.yaml is already configured in the root directory
```

---

## Railway Quick Deploy

### Using Railway Dashboard:
1. Go to [railway.app](https://railway.app) → New Project
2. Connect GitHub repo
3. Railway auto-detects Python

**Build Command:**
```bash
cd ai-training && pip install -r requirements.txt
```

**Start Command:**
```bash
cd ai-training && gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --worker-class sync api_server:app
```

**Or use Procfile** (already created in root):
- Railway will automatically detect and use `Procfile`

**Environment Variables:**
- `PORT` = `${{PORT}}` (auto-provided by Railway)
- `PYTHON_VERSION` = `3.10` (optional)

---

## Gunicorn Configuration

Current configuration:
- **Workers:** 2
- **Timeout:** 120 seconds
- **Worker Class:** sync
- **Bind:** 0.0.0.0:$PORT

### Adjusting for Different Workloads:

**Light Traffic:**
```bash
gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --worker-class sync api_server:app
```

**Heavy Traffic:**
```bash
gunicorn --bind 0.0.0.0:$PORT --workers 4 --timeout 300 --worker-class sync api_server:app
```

**Memory Constrained:**
```bash
gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 180 --worker-class sync --max-requests 1000 api_server:app
```

---

## Testing Commands

### Health Check:
```bash
curl https://your-api-url/health
```

### Test Prediction:
```bash
curl -X POST \
  -F "file=@path/to/brain_scan.nii.gz" \
  https://your-api-url/predict
```

### Test from Array:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"volume": [[[...]]], "shape": [128, 128, 128]}' \
  https://your-api-url/predict_from_array
```

---

## Environment Variables Reference

| Variable | Render | Railway | Description |
|----------|--------|---------|-------------|
| `PORT` | Auto-set | Auto-set | Server port (required) |
| `PYTHON_VERSION` | 3.10 | 3.10 | Python version |
| `FLASK_ENV` | production | production | Flask environment |

---

## File Structure for Deployment

```
neuroview-ai/
├── ai-training/
│   ├── api_server.py          # Main API file
│   ├── requirements.txt       # Dependencies
│   ├── checkpoints/
│   │   └── best_model.pth     # Trained model (must exist)
│   ├── config/
│   │   └── config.py          # Configuration
│   ├── models/
│   │   └── brain_model.py     # Model architecture
│   └── data/
│       └── preprocessing.py   # Data preprocessing
├── Procfile                   # Railway start command
├── render.yaml               # Render configuration
└── DEPLOYMENT_GUIDE.md       # Full deployment guide
```

---

## Common Issues & Solutions

### Issue: Model not loading
**Solution:** Ensure `ai-training/checkpoints/best_model.pth` exists and is committed to git.

### Issue: Build timeout
**Solution:** Increase build timeout in platform settings or optimize requirements.txt.

### Issue: Out of memory
**Solution:** 
- Reduce workers: `--workers 1`
- Upgrade plan
- Optimize model loading

### Issue: CORS errors
**Solution:** Verify `flask-cors` is in requirements.txt and `CORS(app)` is in api_server.py.

### Issue: Port binding error
**Solution:** Use `$PORT` environment variable (auto-provided by platforms).

---

## Deployment Checklist

- [ ] All code committed to git
- [ ] Model files (`best_model.pth`) committed
- [ ] `requirements.txt` up to date
- [ ] Tested API locally
- [ ] Environment variables configured
- [ ] Health endpoint working (`/health`)
- [ ] Prediction endpoint tested
- [ ] Frontend API URL updated
- [ ] Custom domain configured (optional)
- [ ] Monitoring set up

---

## Quick Links

- **Full Guide:** See `DEPLOYMENT_GUIDE.md`
- **Render Docs:** https://render.com/docs
- **Railway Docs:** https://docs.railway.app
- **Gunicorn Docs:** https://docs.gunicorn.org


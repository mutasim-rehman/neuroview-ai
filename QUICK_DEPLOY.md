# Quick Deployment Reference

## 🚀 Fast Track Deployment

### Backend (Render) - 5 Steps

1. **Go to [Render Dashboard](https://dashboard.render.com/)**
2. **New + → Web Service**
3. **Connect GitHub repo**
4. **Configure:**
   - Root Directory: `ai-training`
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --worker-class sync api_server:app`
   - Env: `PYTHON_VERSION=3.10`
5. **Deploy & Copy URL** (e.g., `https://your-api.onrender.com`)

### Frontend (Vercel) - 4 Steps

1. **Go to [Vercel Dashboard](https://vercel.com/dashboard)**
2. **Add New Project → Import GitHub repo**
3. **Add Environment Variable:**
   - `VITE_API_URL` = `https://your-api.onrender.com` (from step above)
4. **Deploy**

### Test

```bash
# Test backend
curl https://your-api.onrender.com/health

# Open frontend URL in browser
```

---

## 📋 Required Files (Already Created)

✅ `vercel.json` - Vercel configuration
✅ `render.yaml` - Render configuration (optional, can configure via dashboard)
✅ `ai-training/requirements.txt` - Includes gunicorn
✅ `ai-training/start.sh` - Startup script (optional)

---

## 🔑 Environment Variables

### Vercel (Frontend)
```
VITE_API_URL=https://your-api.onrender.com
GEMINI_API_KEY=your_key_here (optional)
```

### Render (Backend)
```
PYTHON_VERSION=3.10
PORT=10000 (auto-set by Render)
```

---

## ⚠️ Important Notes

1. **Model Checkpoint:** Ensure `ai-training/checkpoints/best_model.pth` is committed to repo
2. **Render Free Tier:** Spins down after 15 min inactivity (upgrade to Starter $7/mo for always-on)
3. **First Request:** May take 30 seconds if service was spun down
4. **Large Files:** Model files >100MB may need Git LFS

---

## 📚 Full Guide

See `DEPLOYMENT_STEPS.md` for detailed step-by-step instructions.


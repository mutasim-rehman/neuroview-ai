# Deployment Guide for NeuroView AI

This guide covers deploying the frontend to Vercel and the backend API separately.

## Frontend Deployment (Vercel)

### Prerequisites

1. **Vercel account** - Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI** (optional, for terminal deployment)
3. **GitHub repository** with your code

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Go to [Vercel Dashboard](https://vercel.com/dashboard)**

3. **Click "Add New Project"**

4. **Import your GitHub repository**

5. **Configure Project Settings:**
   - Framework Preset: Vite
   - Root Directory: `./` (root)
   - Build Command: `npm run build`
   - Output Directory: `dist`

6. **Add Environment Variables:**
   - `VITE_API_URL` - Your backend API URL (e.g., `https://your-backend.railway.app` or `https://your-backend.render.com`)
   - `GEMINI_API_KEY` - Your Google Gemini API key (for AI analysis features)

7. **Click "Deploy"**

### Option 2: Deploy via Terminal (Vercel CLI)

1. **Install Vercel CLI globally:**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel:**
   ```bash
   vercel login
   ```

3. **Navigate to project root:**
   ```bash
   cd d:\neuroview-ai (3)
   ```

4. **Deploy to production:**
   ```bash
   vercel --prod
   ```

   Or for preview deployment:
   ```bash
   vercel
   ```

5. **Set environment variables:**
   ```bash
   vercel env add VITE_API_URL
   # Enter your backend API URL when prompted
   
   vercel env add GEMINI_API_KEY
   # Enter your Gemini API key when prompted
   ```

6. **Redeploy after adding env vars:**
   ```bash
   vercel --prod
   ```

## Backend API Deployment

The Python Flask API cannot run on Vercel (serverless functions have size limits). Deploy it to one of these platforms:

### Option 1: Railway (Recommended - Easiest)

1. **Sign up at [railway.app](https://railway.app)**

2. **Create new project from GitHub repo**

3. **Add Python service** and point it to `ai-training/` directory

4. **Set Root Directory:** `ai-training`

5. **Add start command:**
   ```
   python api_server.py
   ```

6. **Add environment variables:**
   - Set any required config values from `config/config.py`

7. **Deploy** - Railway auto-detects Python and installs dependencies

8. **Get your API URL** from Railway dashboard (e.g., `https://your-app.railway.app`)

### Option 2: Render

1. **Sign up at [render.com](https://render.com)**

2. **Create new Web Service**

3. **Connect GitHub repository**

4. **Configure:**
   - Build Command: `cd ai-training && pip install -r requirements.txt`
   - Start Command: `cd ai-training && python api_server.py`
   - Environment: Python 3

5. **Add environment variables** if needed

6. **Deploy** and get your API URL

### Option 3: Fly.io

1. **Install Fly CLI:**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login:**
   ```bash
   fly auth login
   ```

3. **In your project root, create `fly.toml`:**
   ```toml
   app = "your-app-name"
   primary_region = "iad"

   [build]
     dockerfile = "ai-training/Dockerfile"

   [http_service]
     internal_port = 5000
     force_https = true
     auto_stop_machines = true
     auto_start_machines = true
     min_machines_running = 0
     processes = ["app"]

   [[vm]]
     memory_mb = 2048
     cpu_kind = "shared"
     cpus = 1
   ```

4. **Create `ai-training/Dockerfile`:**
   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 5000

   CMD ["python", "api_server.py"]
   ```

5. **Deploy:**
   ```bash
   fly deploy
   ```

### Option 4: Heroku

1. **Install Heroku CLI**

2. **Login:**
   ```bash
   heroku login
   ```

3. **Create app:**
   ```bash
   heroku create your-app-name
   ```

4. **Set buildpack:**
   ```bash
   heroku buildpacks:set heroku/python
   ```

5. **Create `Procfile` in `ai-training/`:**
   ```
   web: python api_server.py
   ```

6. **Deploy:**
   ```bash
   cd ai-training
   git subtree push --prefix ai-training heroku main
   ```

## Environment Variables Summary

### Frontend (Vercel)
- `VITE_API_URL` - Backend API URL (e.g., `https://your-backend.railway.app`)
- `GEMINI_API_KEY` - Google Gemini API key

### Backend (Railway/Render/Fly.io)
- Set any custom config values from `config/config.py`
- Python environment variables as needed

## Update Frontend API URL

After deploying backend, update Vercel environment variables:

1. Go to Vercel Dashboard → Your Project → Settings → Environment Variables
2. Update `VITE_API_URL` with your backend URL
3. Redeploy or wait for automatic redeploy

## Testing Deployment

1. **Check frontend:** Visit your Vercel URL
2. **Check backend health:** `curl https://your-backend-url.com/health`
3. **Test prediction:** Use the frontend to upload a brain scan

## Troubleshooting

### Frontend Issues

- **API calls failing:** Check `VITE_API_URL` is set correctly
- **Build failing:** Check Node.js version (Vercel uses 18.x by default)
- **CORS errors:** Ensure backend has CORS enabled (already in `api_server.py`)

### Backend Issues

- **Model not loading:** Verify checkpoint path and file size limits
- **Memory issues:** Increase memory allocation on hosting platform
- **Timeout errors:** Increase timeout settings for large file uploads

## Continuous Deployment

Both Vercel and Railway/Render support automatic deployments:
- **Vercel:** Auto-deploys on push to main branch
- **Railway/Render:** Auto-deploys on push to connected branch

Just push to GitHub and both will redeploy automatically!

## Quick Deploy Commands (Copy & Paste)

### Initial Setup
```bash
# Commit and push code
git add .
git commit -m "Ready for deployment"
git push origin main
```

### Vercel Deployment (Terminal)
```bash
npm install -g vercel
vercel login
vercel --prod
```

### Set Environment Variables
```bash
# Vercel
vercel env add VITE_API_URL production
vercel env add GEMINI_API_KEY production

# Then redeploy
vercel --prod
```

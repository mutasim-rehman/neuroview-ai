# Quick Deploy Commands

## Step 1: Push to GitHub

```bash
cd "d:\neuroview-ai (3)"
git add .
git commit -m "Prepare for Vercel deployment"
git push origin main
```

## Step 2: Deploy Frontend to Vercel (via Terminal)

### Install Vercel CLI (first time only)
```bash
npm install -g vercel
```

### Login to Vercel (first time only)
```bash
vercel login
```

### Deploy to Vercel
```bash
vercel --prod
```

### Set Environment Variables in Vercel
After first deploy, set these in Vercel Dashboard OR via CLI:

```bash
# Set API URL (replace with your backend URL after you deploy backend)
vercel env add VITE_API_URL production
# When prompted, enter: https://your-backend-url.railway.app (or your backend URL)

# Set Gemini API key
vercel env add GEMINI_API_KEY production
# When prompted, enter your Google Gemini API key
```

### Redeploy after setting env vars
```bash
vercel --prod
```

## Step 3: Deploy Backend API (Railway - Recommended)

### Install Railway CLI (optional)
```bash
npm install -g @railway/cli
```

### Login to Railway
```bash
railway login
```

### Initialize Railway project
```bash
cd "d:\neuroview-ai (3)\ai-training"
railway init
```

### Deploy to Railway
```bash
railway up
```

### Get your Railway URL
After deployment, Railway will give you a URL like: `https://your-app.railway.app`

### Update Vercel with Railway URL
Go back to Vercel and update `VITE_API_URL` environment variable with your Railway URL, then redeploy:

```bash
vercel --prod
```

## Alternative: Deploy Backend via Railway Dashboard

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Select your repository
5. Set Root Directory to: `ai-training`
6. Railway will auto-detect Python and deploy
7. Get your API URL from Railway dashboard
8. Update `VITE_API_URL` in Vercel with this URL

## Quick Reference

```bash
# Frontend deployment
vercel --prod

# Check Vercel deployments
vercel ls

# View Vercel logs
vercel logs

# Backend deployment (Railway CLI)
railway up

# Check Railway status
railway status
```

## Environment Variables Checklist

### Vercel (Frontend)
- ✅ `VITE_API_URL` - Your Railway/Render backend URL
- ✅ `GEMINI_API_KEY` - Google Gemini API key

### Railway/Render (Backend)
- ✅ Auto-configured from `config.py`
- ✅ Model path: `./checkpoints/best_model.pth`

## Troubleshooting

If deployment fails:
1. Check logs: `vercel logs` or Railway dashboard
2. Ensure model checkpoint exists in `ai-training/checkpoints/`
3. Verify environment variables are set correctly

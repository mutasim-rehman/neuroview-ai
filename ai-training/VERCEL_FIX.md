# URGENT: Fix Vercel Install Command

## The Problem

Vercel is running `python api_server.py` as the **Install Command**, which is wrong. This tries to run your server before dependencies are installed, causing the `ModuleNotFoundError: No module named 'numpy'` error.

## The Solution - YOU MUST DO THIS IN VERCEL DASHBOARD

**Dashboard settings override `vercel.json`**, so you need to manually update them:

### Step-by-Step Instructions:

1. **Go to Vercel Dashboard**
   - Navigate to: https://vercel.com/dashboard
   - Select your project: `neuroview-ai-zifv` (or whatever it's named)

2. **Open Project Settings**
   - Click on **Settings** tab
   - Go to **General** → **Build & Development Settings**

3. **Fix the Install Command** ⚠️ **CRITICAL**
   - Find **"Install Command"**
   - **DELETE** the current value: `python api_server.py`
   - **REPLACE** with: `pip install -r requirements.txt`
   - OR **LEAVE IT EMPTY** (Vercel will auto-detect `requirements.txt`)

4. **Fix the Build Command**
   - Find **"Build Command"**
   - **DELETE** any value there
   - **LEAVE IT EMPTY**

5. **Save Settings**
   - Click **Save** at the bottom

6. **Redeploy**
   - Go to **Deployments** tab
   - Click the **"..."** menu on the latest deployment
   - Click **Redeploy**

## Why This Happens

Vercel dashboard settings take **precedence** over `vercel.json`. Even though we have a `vercel.json` file, if you manually set the Install Command in the dashboard, it will use that instead.

## After Fixing

Once you update the Install Command to `pip install -r requirements.txt` (or leave it empty), Vercel will:
1. ✅ Install all dependencies from `requirements.txt`
2. ✅ Then build your Python serverless function
3. ✅ Deploy successfully

## Verification

After redeploying, check the build logs. You should see:
```
Running "install" command: `pip install -r requirements.txt`...
```

**NOT:**
```
Running "install" command: `python api_server.py`...
```

## If You Still Have Issues

If after fixing the Install Command you still get errors:

1. **Check Root Directory**: Should be set to `ai-training`
2. **Check requirements.txt exists**: Should be in `ai-training/requirements.txt`
3. **Check Python version**: Vercel should auto-detect, but you can set it to 3.11 in settings


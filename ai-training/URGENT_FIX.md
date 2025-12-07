# ⚠️ URGENT: Fix Required in Vercel Dashboard

## The Problem
Vercel is running `python api_server.py` as the **Install Command**, which fails because dependencies aren't installed yet.

## The Solution (YOU MUST DO THIS)

**Dashboard settings override `vercel.json`**. You MUST manually update the Vercel dashboard.

### Step-by-Step (with exact locations):

1. **Open Vercel Dashboard**
   - Go to: https://vercel.com/dashboard
   - Click on project: `neuroview-ai-zifv` (or your project name)

2. **Navigate to Settings**
   - Click the **"Settings"** tab (top navigation)
   - OR click the **gear icon** ⚙️ next to your project name

3. **Go to Build Settings**
   - In the left sidebar, click **"General"**
   - Scroll down to section: **"Build & Development Settings"**

4. **Fix Install Command** ⚠️ CRITICAL
   - Find the field labeled **"Install Command"**
   - **CURRENT VALUE** (WRONG): `python api_server.py`
   - **DELETE IT ALL**
   - **TYPE NEW VALUE**: `pip install -r requirements.txt`
   - OR **LEAVE IT EMPTY** (Vercel auto-detects requirements.txt)

5. **Fix Build Command**
   - Find the field labeled **"Build Command"**
   - **DELETE** any value
   - **LEAVE IT EMPTY**

6. **Verify Root Directory**
   - Find **"Root Directory"**
   - Should be: `ai-training`
   - If not, set it to: `ai-training`

7. **SAVE**
   - Scroll to bottom
   - Click **"Save"** button

8. **Redeploy**
   - Go to **"Deployments"** tab
   - Find the latest failed deployment
   - Click **"..."** (three dots menu)
   - Click **"Redeploy"**

## Why This Keeps Happening

Vercel dashboard settings **ALWAYS take precedence** over `vercel.json` for:
- Install Command
- Build Command  
- Output Directory

Even though we have `vercel.json` with correct settings, the dashboard override is causing the issue.

## After Fixing

Check the build logs. You should see:
```
Running "install" command: `pip install -r requirements.txt`...
Collecting numpy...
Collecting torch...
...
Successfully installed numpy-1.24.3 torch-2.0.0 ...
```

**NOT:**
```
Running "install" command: `python api_server.py`...
Traceback (most recent call last):
  File "/vercel/path0/ai-training/api_server.py", line 10, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

## Still Not Working?

If after updating the dashboard you still get the error:

1. **Clear Vercel cache**: In project settings → "General" → scroll to bottom → "Clear Build Cache"
2. **Check Root Directory**: Must be exactly `ai-training` (case-sensitive)
3. **Verify requirements.txt exists**: Should be at `ai-training/requirements.txt`
4. **Try deleting and recreating the project** (last resort)

## Alternative: Delete Project Settings

If you can't find the settings:
1. Go to project Settings
2. Look for "Reset" or "Clear" options
3. This will make Vercel use `vercel.json` instead


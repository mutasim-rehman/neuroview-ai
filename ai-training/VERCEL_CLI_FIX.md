# Fix Vercel Install Command - CLI Method

If the dashboard isn't working, you can use the Vercel CLI to fix this programmatically.

## Option 1: Using Vercel CLI (Recommended)

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Link your project** (from the repository root):
   ```bash
   cd "D:\neuroview-ai (3)"
   vercel link
   ```
   - Select your existing project
   - Confirm the settings

4. **Update project settings via CLI**:
   ```bash
   vercel env pull .env.local
   ```
   
   Then update the project configuration. However, CLI doesn't directly support changing build commands. You'll need to use the dashboard or API.

## Option 2: Using Vercel API

You can use the Vercel API to update project settings:

```bash
# Get your Vercel token from: https://vercel.com/account/tokens
export VERCEL_TOKEN="your_token_here"
export PROJECT_ID="your_project_id"  # Find this in project settings

# Update project configuration
curl -X PATCH "https://api.vercel.com/v9/projects/$PROJECT_ID" \
  -H "Authorization: Bearer $VERCEL_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "installCommand": "pip install -r requirements.txt",
    "buildCommand": "",
    "rootDirectory": "ai-training"
  }'
```

## Option 3: Manual Dashboard Fix (Most Reliable)

**You MUST do this in the Vercel Dashboard:**

1. Go to: https://vercel.com/dashboard
2. Click on your project: `neuroview-ai-zifv`
3. Click **Settings** (gear icon in top right)
4. Scroll down to **"Build & Development Settings"**
5. Find **"Install Command"** field
6. **DELETE** the text: `python api_server.py`
7. **TYPE**: `pip install -r requirements.txt`
8. Find **"Build Command"** field
9. **DELETE** any text there (leave empty)
10. Click **"Save"** button at the bottom
11. Go to **Deployments** tab
12. Click **"..."** on latest deployment → **"Redeploy"**

## Why This Keeps Happening

Vercel dashboard settings **ALWAYS override** `vercel.json` for build/install commands. Even though we have `vercel.json` with the correct settings, if the dashboard has different values, those take precedence.

## Verification

After fixing, check the build logs. You should see:
```
Running "install" command: `pip install -r requirements.txt`...
```

**NOT:**
```
Running "install" command: `python api_server.py`...
```


# Vercel Install Commands (Try in Order)

## Option 1: Minimal Requirements with CPU PyTorch (Recommended)
```
pip install --extra-index-url https://download.pytorch.org/whl/cpu --only-binary :all: -r requirements-api.txt
```

## Option 2: Without --only-binary flag
```
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-api.txt
```

## Option 3: Split Installation (if Option 1 & 2 fail)
```
pip install -r requirements-api-simple.txt && pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Option 4: Install PyTorch Last (most memory-efficient)
```
pip install numpy nibabel scipy flask flask-cors && pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## How to Update

1. Vercel Dashboard → Project → Settings → General
2. Find **"Install Command"**
3. Paste one of the commands above
4. **Save** and **Redeploy**

## Which to Try First

Start with **Option 1**. If it fails, try **Option 2**, then **Option 3**, then **Option 4**.


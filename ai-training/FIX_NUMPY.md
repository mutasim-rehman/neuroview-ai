# Fix NumPy Compatibility Issue

## Problem
You're encountering this error because NumPy 2.0+ removed `np.sctypes`, which nibabel still uses:
```
AttributeError: `np.sctypes` was removed in the NumPy 2.0 release.
```

## Solution

You need to downgrade NumPy to a version < 2.0.0. Here are the steps:

### Option 1: Using pip (Recommended)

1. **Activate your virtual environment** (if you're using one):
   ```bash
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

2. **Downgrade NumPy**:
   ```bash
   pip install "numpy>=1.24.0,<2.0.0" --upgrade
   ```

3. **Verify the installation**:
   ```bash
   python -c "import numpy; print(numpy.__version__)"
   ```
   Should show something like `1.26.x` (not 2.x.x)

### Option 2: Reinstall all dependencies

If you want to ensure all packages are compatible:

1. **Uninstall NumPy**:
   ```bash
   pip uninstall numpy -y
   ```

2. **Reinstall from requirements.txt**:
   ```bash
   cd ai-training
   pip install -r requirements.txt --upgrade
   ```

### Option 3: Use Python 3.11 or earlier

If you continue having issues, you might also want to ensure you're using Python 3.11 or earlier, as Python 3.13 might have additional compatibility issues.

## Verification

After downgrading, test if the import works:

```bash
python -c "import nibabel; import numpy; print('Success! NumPy version:', numpy.__version__)"
```

## Why This Happened

- NumPy 2.0 was released with breaking changes
- `np.sctypes` was deprecated and removed
- Current versions of nibabel haven't been fully updated for NumPy 2.0 yet
- We've constrained NumPy to < 2.0.0 in requirements.txt to prevent this issue

## Long-term Solution

Eventually, nibabel will release a version compatible with NumPy 2.0+. When that happens, you can update the requirements.txt file to allow NumPy 2.0+.


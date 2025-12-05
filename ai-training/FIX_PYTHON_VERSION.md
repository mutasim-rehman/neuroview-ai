# Fix: Python 3.13 Compatibility Issue

## Problem

You're using **Python 3.13**, which is very new. NumPy 1.x versions don't have pre-built wheels for Python 3.13, and NumPy 2.x is incompatible with current nibabel.

## Solutions

### Option 1: Use Python 3.11 or 3.12 (RECOMMENDED)

Python 3.11 and 3.12 have full NumPy 1.x support.

**Steps:**
1. Download and install Python 3.11 or 3.12 from python.org
2. Create a new virtual environment with the correct Python version:
   ```bash
   # Windows (if Python 3.11 is installed as py -3.11)
   py -3.11 -m venv venv
   
   # Or if you have multiple Python versions:
   python3.11 -m venv venv
   ```

3. Activate the new venv:
   ```bash
   venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Upgrade nibabel to support NumPy 2.0

Try installing a newer version of nibabel that might support NumPy 2.0:

```bash
pip install --upgrade nibabel
pip install numpy  # Will install NumPy 2.x
```

Then test:
```bash
python -c "import numpy, nibabel; print('Success!')"
```

### Option 3: Use conda (if available)

Conda has better Python version management:

```bash
conda create -n brain-training python=3.11
conda activate brain-training
pip install -r requirements.txt
```

## Quick Check: What Python versions do you have?

Run these commands to see what's available:

```bash
py --list
# or
python --version
python3.11 --version
python3.12 --version
```

## Recommended Action

**Create a new virtual environment with Python 3.11 or 3.12** - this is the most reliable solution.


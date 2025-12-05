"""
Check what Python versions are available on your system.
"""

import sys
import subprocess
import os

print("="*60)
print("PYTHON VERSION CHECK")
print("="*60)
print(f"\nCurrent Python version: {sys.version}")
print(f"Current Python executable: {sys.executable}")
print(f"Python version info: {sys.version_info}")

print("\n" + "="*60)
print("Checking for other Python versions...")
print("="*60)

# Common Python executable names to check
python_versions = ['python3.11', 'python3.12', 'py -3.11', 'py -3.12']

print("\nTrying to find Python 3.11 or 3.12...")
found_versions = []

for py_cmd in python_versions:
    try:
        # Try to get version
        if py_cmd.startswith('py -'):
            cmd = py_cmd.split() + ['--version']
        else:
            cmd = [py_cmd, '--version']
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            found_versions.append((py_cmd, version))
            print(f"✓ Found: {py_cmd} -> {version}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as e:
        pass

if not found_versions:
    print("✗ No Python 3.11 or 3.12 found automatically.")
    print("\nTry running manually:")
    print("  py -3.11 --version")
    print("  py -3.12 --version")
else:
    print(f"\n✓ Found {len(found_versions)} compatible Python version(s)")
    print("\nRecommendation: Create a new venv with one of these:")
    for cmd, version in found_versions:
        if '3.11' in cmd or '3.12' in cmd:
            print(f"\n  {cmd} -m venv venv311")
            print(f"  venv311\\Scripts\\activate")
            print(f"  pip install -r requirements.txt")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("\nSince you're on Python 3.13 and NumPy 1.x doesn't support it:")
print("\n1. Install Python 3.11 or 3.12 from python.org")
print("2. Create new venv: py -3.11 -m venv venv")
print("3. Activate: venv\\Scripts\\activate")
print("4. Install: pip install -r requirements.txt")
print("\n" + "="*60)


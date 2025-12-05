"""
Quick script to fix NumPy version compatibility.
Run this to downgrade NumPy to a compatible version.
"""

import subprocess
import sys

print("Fixing NumPy compatibility issue...")
print("Current NumPy version will be downgraded to < 2.0.0")
print()

try:
    # Uninstall current numpy
    print("Step 1: Uninstalling current NumPy...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "uninstall", "numpy", "-y"
    ])
    
    # Install compatible version
    print("\nStep 2: Installing NumPy 1.26.x...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "numpy>=1.24.0,<2.0.0"
    ])
    
    # Verify
    print("\nStep 3: Verifying installation...")
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
    
    # Test nibabel import
    print("\nStep 4: Testing nibabel import...")
    import nibabel as nib
    print("✓ nibabel imported successfully!")
    
    print("\n" + "="*50)
    print("SUCCESS! NumPy has been downgraded.")
    print("You can now run: python main_train_healthy.py")
    print("="*50)
    
except subprocess.CalledProcessError as e:
    print(f"\n✗ Error during installation: {e}")
    print("Please run manually:")
    print('  pip uninstall numpy -y')
    print('  pip install "numpy>=1.24.0,<2.0.0"')
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    sys.exit(1)


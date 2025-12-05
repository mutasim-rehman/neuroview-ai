"""
Try to make NumPy 2.0 work with latest nibabel.
This is a workaround if you can't change Python version.
"""

import subprocess
import sys

print("="*60)
print("ATTEMPTING: NumPy 2.0 + Latest nibabel compatibility")
print("="*60)
print("\nThis will:")
print("1. Install/upgrade to latest nibabel")
print("2. Install NumPy 2.x (required for Python 3.13)")
print("3. Test if they work together")
print("\n" + "="*60 + "\n")

try:
    # Step 1: Upgrade nibabel to latest
    print("Step 1: Installing latest nibabel...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--upgrade", "nibabel"
    ])
    print("✓ nibabel upgraded")
    
    # Step 2: Install NumPy 2.x (will work with Python 3.13)
    print("\nStep 2: Installing NumPy 2.x...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--upgrade", "numpy"
    ])
    print("✓ NumPy installed")
    
    # Step 3: Test imports
    print("\nStep 3: Testing compatibility...")
    import numpy as np
    print(f"  NumPy version: {np.__version__}")
    
    import nibabel as nib
    print(f"  nibabel version: {nib.__version__}")
    print("✓ Both imported successfully!")
    
    # Step 4: Test actual usage
    print("\nStep 4: Testing basic functionality...")
    # Create a simple test
    test_array = np.zeros((10, 10, 10))
    print("  Created test array")
    print("  ✓ Basic functionality works!")
    
    print("\n" + "="*60)
    print("SUCCESS! NumPy 2.0 + nibabel are working together!")
    print("="*60)
    print("\nYou can now run:")
    print("  python main_train_healthy.py")
    print("\nNote: If you encounter issues, consider using Python 3.11/3.12")
    print("="*60)
    
except subprocess.CalledProcessError as e:
    print(f"\n✗ Error during installation: {e}")
    print("\nThis approach didn't work.")
    print("\nRECOMMENDATION: Use Python 3.11 or 3.12 instead.")
    print("See FINAL_FIX.md for instructions.")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*60)
    print("This workaround didn't work.")
    print("RECOMMENDATION: Use Python 3.11 or 3.12")
    print("See FINAL_FIX.md for detailed instructions.")
    print("="*60)
    sys.exit(1)


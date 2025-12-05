"""
Fix NumPy by installing pre-built binary wheel instead of building from source.
"""

import subprocess
import sys
import platform

print("Fixing NumPy compatibility issue with pre-built wheels...")
print()

try:
    # Uninstall current numpy
    print("Step 1: Uninstalling current NumPy...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "uninstall", "numpy", "-y", "--quiet"
    ])
    
    # Install pre-built wheel (use --only-binary to prevent building from source)
    print("\nStep 2: Installing NumPy from pre-built wheel...")
    print("(This avoids compilation issues)")
    
    # Try to install with --only-binary flag
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--only-binary", ":all:",
            "numpy>=1.24.0,<2.0.0",
            "--upgrade"
        ])
    except subprocess.CalledProcessError:
        # If that fails, try installing a specific version that definitely has wheels
        print("Trying specific version 1.26.4...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--only-binary", ":all:",
            "numpy==1.26.4",
            "--upgrade"
        ])
    
    # Verify
    print("\nStep 3: Verifying installation...")
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
    
    if np.__version__.startswith('2.'):
        print("⚠ Warning: Still NumPy 2.x, trying to force downgrade...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "numpy==1.26.4", "--force-reinstall", "--no-deps"
        ])
        import importlib
        importlib.reload(np)
        print(f"✓ NumPy version after force: {np.__version__}")
    
    # Test nibabel import
    print("\nStep 4: Testing nibabel import...")
    import nibabel as nib
    print("✓ nibabel imported successfully!")
    
    print("\n" + "="*50)
    print("SUCCESS! NumPy has been installed from pre-built wheel.")
    print("You can now run: python main_train_healthy.py")
    print("="*50)
    
except subprocess.CalledProcessError as e:
    print(f"\n✗ Error during installation: {e}")
    print("\nAlternative solution:")
    print("  Try: pip install --upgrade pip wheel setuptools")
    print("  Then: pip install numpy==1.26.4 --only-binary :all:")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


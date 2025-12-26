"""
Simple test for NeuroView AI API - tests model prediction without file upload.
This helps diagnose if the issue is with file upload or model processing.
"""

import requests
import json
import numpy as np
import base64

API_BASE_URL = "https://neuroview-ai.onrender.com"

def test_predict_from_array():
    """Test prediction using JSON array data (no file upload needed)."""
    print("\n" + "="*60)
    print("Testing /predict_from_array endpoint")
    print("="*60)
    
    # Create a small synthetic 3D volume (128x128x128)
    # This simulates a brain scan with some random values
    print("\nCreating synthetic 128x128x128 volume...")
    volume = np.random.rand(128, 128, 128).astype(np.float32)
    
    # Convert to nested list for JSON
    volume_list = volume.tolist()
    
    print(f"Volume shape: {volume.shape}")
    print(f"Sending to {API_BASE_URL}/predict_from_array...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_from_array",
            json={
                'volume': volume_list,
                'shape': [128, 128, 128]
            },
            timeout=300
        )
        
        print(f"\nStatus code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n[OK] Prediction successful!")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"\n[FAIL] Prediction failed!")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(f"Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("\n[FAIL] Request timed out!")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=30)
        print(f"\nStatus code: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
            return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_with_smaller_volume():
    """Test with a very small volume to check if memory is the issue."""
    print("\n" + "="*60)
    print("Testing with small 32x32x32 volume (JSON nested list)")
    print("="*60)
    
    # Create a very small volume
    volume = np.random.rand(32, 32, 32).astype(np.float32)
    volume_list = volume.tolist()
    
    print(f"Volume shape: {volume.shape}")
    print(f"Sending to {API_BASE_URL}/predict_from_array...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_from_array",
            json={
                'volume': volume_list,
                'shape': [32, 32, 32]
            },
            timeout=120
        )
        
        print(f"\nStatus code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n[OK] Prediction successful!")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"\n[FAIL] Prediction failed!")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(f"Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_base64_prediction():
    """Test with base64-encoded volume (more memory efficient)."""
    print("\n" + "="*60)
    print("Testing /predict_base64 endpoint (memory efficient)")
    print("="*60)
    
    # Create a small volume
    volume = np.random.rand(64, 64, 64).astype(np.float32)
    
    # Encode as base64
    volume_bytes = volume.tobytes()
    volume_b64 = base64.b64encode(volume_bytes).decode('utf-8')
    
    print(f"Volume shape: {volume.shape}")
    print(f"Raw bytes: {len(volume_bytes)} bytes ({len(volume_bytes)/1024:.1f} KB)")
    print(f"Base64 size: {len(volume_b64)} chars ({len(volume_b64)/1024:.1f} KB)")
    print(f"Sending to {API_BASE_URL}/predict_base64...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_base64",
            json={
                'data': volume_b64,
                'shape': list(volume.shape),
                'dtype': 'float32'
            },
            timeout=120
        )
        
        print(f"\nStatus code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n[OK] Base64 prediction successful!")
            print(json.dumps(result, indent=2))
            return True
        elif response.status_code == 404:
            print("\n[WARN] /predict_base64 endpoint not found (needs redeployment)")
            return None
        else:
            print(f"\n[FAIL] Base64 prediction failed!")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(f"Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("NeuroView AI - Simple API Test (No File Upload)")
    print("="*60)
    print(f"\nAPI URL: {API_BASE_URL}")
    
    # Test health first
    if not test_health():
        print("\n[WARN] Health check failed - API may be down")
        print("Waiting 30 seconds for API to recover...")
        import time
        time.sleep(30)
        if not test_health():
            print("\n[FAIL] API still down. Exiting.")
            exit(1)
    
    # Test base64 prediction first (most memory efficient)
    print("\n" + "-"*60)
    print("Testing base64 endpoint (most memory efficient)...")
    base64_result = test_base64_prediction()
    
    if base64_result is None:
        print("Base64 endpoint not available yet - needs redeployment")
    elif base64_result:
        print("\n[OK] Base64 prediction works! This is the recommended method.")
    
    # Test with small volume (JSON nested list)
    print("\n" + "-"*60)
    print("Testing JSON nested list (small volume)...")
    test_with_smaller_volume()
    
    # Skip large volume test as it's known to cause issues
    print("\n" + "-"*60)
    print("Skipping 128x128x128 JSON test (known to cause memory issues)")
    print("Use /predict with file upload or /predict_base64 for large volumes")
    
    print("\n" + "="*60)
    print("Tests complete!")
    print("="*60)


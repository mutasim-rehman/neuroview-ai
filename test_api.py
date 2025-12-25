"""
Test script for NeuroView AI API deployed on Render.
Tests API health, model status, and prediction functionality with .nii files.

Requirements:
    pip install requests

Usage:
    python test_api.py [path_to_nii_file]
    
    If no file path is provided, the script will prompt for one.
    Default file: ai-training/demo.nii (if exists)
"""

import requests
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# API base URL
API_BASE_URL = "https://neuroview-ai.onrender.com"

# Colors for terminal output (Windows compatible)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def test_health_endpoint() -> Optional[Dict[str, Any]]:
    """
    Test the /health endpoint to check if API is working.
    
    Returns:
        Health check response dictionary or None if failed
    """
    print_header("TEST 1: API Health Check")
    
    try:
        print_info(f"Checking API health at {API_BASE_URL}/health...")
        response = requests.get(f"{API_BASE_URL}/health", timeout=30)
        
        if response.status_code == 200:
            health_data = response.json()
            print_success(f"API is responding! Status code: {response.status_code}")
            print(f"\nHealth Check Response:")
            print(json.dumps(health_data, indent=2))
            
            # Check model status
            if health_data.get('model_loaded', False):
                print_success("Model is loaded and ready!")
            else:
                print_warning("Model is not loaded - predictions may fail")
            
            return health_data
        else:
            print_error(f"API returned status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print_error("Request timed out. The API might be starting up (cold start on Render).")
        print_info("Render free tier services can take 30-60 seconds to wake up.")
        return None
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to the API. Check if the URL is correct.")
        return None
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return None

def test_root_endpoint() -> bool:
    """
    Test the root endpoint to verify API is accessible.
    
    Returns:
        True if successful, False otherwise
    """
    print_header("TEST 2: Root Endpoint Check")
    
    try:
        print_info(f"Checking root endpoint at {API_BASE_URL}/...")
        response = requests.get(API_BASE_URL, timeout=30)
        
        if response.status_code == 200:
            root_data = response.json()
            print_success("Root endpoint is accessible!")
            print(f"\nRoot Endpoint Response:")
            print(json.dumps(root_data, indent=2))
            return True
        else:
            print_error(f"Root endpoint returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error accessing root endpoint: {str(e)}")
        return False

def predict_from_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Upload a .nii file and get prediction from the API.
    
    Args:
        file_path: Path to the .nii file
        
    Returns:
        Prediction response dictionary or None if failed
    """
    print_header("TEST 3: Prediction from .nii File")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print_error(f"File not found: {file_path}")
        return None
    
    # Check file extension
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in ['.nii', '.gz'] and not file_path.endswith('.nii.gz'):
        print_warning(f"File extension '{file_ext}' might not be a valid NIfTI file.")
        print_info("Expected: .nii or .nii.gz")
    
    file_size = os.path.getsize(file_path)
    print_info(f"File: {file_path}")
    print_info(f"Size: {file_size / (1024*1024):.2f} MB")
    
    try:
        print_info(f"Uploading file to {API_BASE_URL}/predict...")
        print_info("This may take a while depending on file size and API cold start...")
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            response = requests.post(
                f"{API_BASE_URL}/predict",
                files=files,
                timeout=300  # 5 minutes timeout for large file upload and processing
            )
        
        if response.status_code == 200:
            prediction_data = response.json()
            print_success("Prediction successful!")
            return prediction_data
        else:
            print_error(f"Prediction failed with status code: {response.status_code}")
            try:
                error_data = response.json()
                print_error(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print_error(f"Response text: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print_error("Request timed out after 5 minutes. The file might be too large or API is slow.")
        print_info("Try again - Render free tier can be slow on cold starts.")
        print_info("Check Render logs for more details.")
        return None
    except Exception as e:
        print_error(f"Error during prediction: {str(e)}")
        return None

def display_prediction_results(results: Dict[str, Any]):
    """
    Display prediction results in a formatted way.
    
    Args:
        results: Prediction results dictionary
    """
    print_header("PREDICTION RESULTS")
    
    if 'error' in results:
        print_error(f"Prediction Error: {results['error']}")
        return
    
    # Main prediction
    prediction = results.get('prediction', 'unknown')
    confidence = results.get('confidence', 0.0)
    anomaly_score = results.get('anomaly_score', 0.0)
    
    print(f"\n{Colors.BOLD}Prediction:{Colors.END}")
    if prediction == 'healthy':
        print(f"  {Colors.GREEN}Status: HEALTHY{Colors.END}")
    else:
        print(f"  {Colors.RED}Status: DEFECT DETECTED{Colors.END}")
    
    print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"  Anomaly Score: {anomaly_score:.6f}")
    
    # Error metrics
    if 'error_metrics' in results:
        print(f"\n{Colors.BOLD}Error Metrics:{Colors.END}")
        error_metrics = results['error_metrics']
        print(f"  MSE (Mean Squared Error): {error_metrics.get('mse', 'N/A'):.6f}")
        print(f"  MAE (Mean Absolute Error): {error_metrics.get('mae', 'N/A'):.6f}")
        print(f"  Max Error: {error_metrics.get('max_error', 'N/A'):.6f}")
    
    # Feature vector info
    if 'feature_vector' in results and results['feature_vector'] is not None:
        feature_vector = results['feature_vector']
        if isinstance(feature_vector, list):
            print(f"\n{Colors.BOLD}Feature Vector:{Colors.END}")
            print(f"  Length: {len(feature_vector)}")
            print(f"  First 5 values: {feature_vector[:5]}")
            print(f"  Last 5 values: {feature_vector[-5:]}")
    
    # Full JSON output
    print(f"\n{Colors.BOLD}Full JSON Response:{Colors.END}")
    print(json.dumps(results, indent=2))

def get_nii_file() -> Optional[str]:
    """
    Get .nii file path from user input.
    
    Returns:
        File path or None if cancelled
    """
    print_header("File Selection")
    
    # Check if file path provided as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            print_info(f"Using file from command line: {file_path}")
            return file_path
        else:
            print_error(f"File not found: {file_path}")
    
    # Interactive file selection
    print_info("Please provide the path to a .nii file:")
    print_info("  - You can drag and drop the file into this terminal")
    print_info("  - Or type the full path")
    print_info("  - Or press Enter to use default: ai-training/demo.nii")
    
    user_input = input("\nFile path: ").strip()
    
    if not user_input:
        # Try default file
        default_path = "ai-training/demo.nii"
        if os.path.exists(default_path):
            print_info(f"Using default file: {default_path}")
            return default_path
        else:
            print_error(f"Default file not found: {default_path}")
            return None
    
    # Remove quotes if user pasted path with quotes
    user_input = user_input.strip('"').strip("'")
    
    if os.path.exists(user_input):
        return user_input
    else:
        print_error(f"File not found: {user_input}")
        return None

def main():
    """Main test function."""
    print_header("NeuroView AI API Test Suite")
    print_info(f"Testing API at: {API_BASE_URL}")
    print_info("This script will test:")
    print_info("  1. API health and availability")
    print_info("  2. Model loading status")
    print_info("  3. Prediction functionality with .nii file")
    print()
    
    # Test 1: Health check
    health_data = test_health_endpoint()
    
    if health_data is None:
        print_warning("\nAPI health check failed. Continuing with other tests...")
        print_warning("The API might be starting up (cold start on Render).")
    else:
        # Verify model is loaded
        if not health_data.get('model_loaded', False):
            print_warning("\nModel is not loaded. Predictions will likely fail.")
            print_warning("Check the API server logs on Render.")
    
    # Test 2: Root endpoint
    test_root_endpoint()
    
    # Test 3: Prediction
    file_path = get_nii_file()
    
    if file_path is None:
        print_error("\nNo valid file provided. Exiting.")
        return
    
    print()
    prediction_results = predict_from_file(file_path)
    
    if prediction_results:
        display_prediction_results(prediction_results)
        
        # Summary
        print_header("TEST SUMMARY")
        print_success("✓ API is accessible")
        if health_data and health_data.get('model_loaded', False):
            print_success("✓ Model is loaded")
        else:
            print_warning("⚠ Model status unknown or not loaded")
        
        if 'error' not in prediction_results:
            print_success("✓ Prediction completed successfully")
            print_success("✓ API can be used for predictions")
        else:
            print_error("✗ Prediction failed")
    else:
        print_header("TEST SUMMARY")
        print_error("✗ Prediction test failed")
        print_info("Possible reasons:")
        print_info("  - API is still starting up (cold start)")
        print_info("  - Model is not loaded")
        print_info("  - File format issue")
        print_info("  - Network connectivity issue")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print_error(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


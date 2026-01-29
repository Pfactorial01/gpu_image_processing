"""
Test client for GPU Image Processing API
Tests all endpoints and saves results
"""

import requests
import base64
import json
from PIL import Image
import io
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def load_image_as_base64(image_path):
    """Load image file and convert to base64"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    base64_str = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

def save_base64_image(base64_str, output_path):
    """Save base64 image to file"""
    # Remove data URL prefix
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # Decode
    image_data = base64.b64decode(base64_str)
    
    # Save
    image = Image.open(io.BytesIO(image_data))
    image.save(output_path)
    print(f"  💾 Saved: {output_path}")

def test_root():
    """Test root endpoint"""
    print("\n" + "="*70)
    print("TEST 1: Root Endpoint")
    print("="*70)
    
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_health():
    """Test health check"""
    print("\n" + "="*70)
    print("TEST 2: Health Check")
    print("="*70)
    
    response = requests.get(f"{API_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_list_filters():
    """Test list filters"""
    print("\n" + "="*70)
    print("TEST 3: List Filters")
    print("="*70)
    
    response = requests.get(f"{API_URL}/api/filters")
    print(f"Status: {response.status_code}")
    data = response.json()
    
    print(f"\nAvailable filters: {list(data['filters'].keys())}")
    for filter_name, filter_info in data['filters'].items():
        print(f"\n  {filter_name}:")
        print(f"    Name: {filter_info['name']}")
        print(f"    Description: {filter_info['description']}")
    
    return response.status_code == 200

def test_process_gaussian(input_image):
    """Test Gaussian blur processing"""
    print("\n" + "="*70)
    print("TEST 4: Process with Gaussian Blur")
    print("="*70)
    
    # Load and encode image
    print(f"Loading image: {input_image}")
    base64_image = load_image_as_base64(input_image)
    
    # Prepare request
    request_data = {
        "image": base64_image,
        "filter": "gaussian",
        "level": 2,
        "sigma": 2.0,
        "radius": 3
    }
    
    print("Sending request...")
    response = requests.post(f"{API_URL}/api/process", json=request_data)
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return False
    
    data = response.json()
    
    print(f"\n✅ Success!")
    print(f"Metrics:")
    print(f"  Time: {data['metrics']['time_ms']:.3f} ms")
    print(f"  Bandwidth: {data['metrics']['bandwidth_gbps']:.2f} GB/s")
    print(f"  FPS: {data['metrics']['fps']:.2f}")
    
    print(f"\nInfo:")
    print(f"  Filter: {data['info']['filter']}")
    print(f"  Level: {data['info']['level']}")
    print(f"  Resolution: {data['info']['width']}x{data['info']['height']}")
    print(f"  Channels: {data['info']['channels']}")
    
    # Save result
    output_path = "output_gaussian_api.png"
    save_base64_image(data['processed_image'], output_path)
    
    return True

def test_process_box(input_image):
    """Test Box blur processing"""
    print("\n" + "="*70)
    print("TEST 5: Process with Box Blur")
    print("="*70)
    
    print(f"Loading image: {input_image}")
    base64_image = load_image_as_base64(input_image)
    
    request_data = {
        "image": base64_image,
        "filter": "box",
        "level": 2,
        "radius": 3
    }
    
    print("Sending request...")
    response = requests.post(f"{API_URL}/api/process", json=request_data)
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return False
    
    data = response.json()
    
    print(f"\n✅ Success!")
    print(f"Metrics:")
    print(f"  Time: {data['metrics']['time_ms']:.3f} ms")
    print(f"  Bandwidth: {data['metrics']['bandwidth_gbps']:.2f} GB/s")
    print(f"  FPS: {data['metrics']['fps']:.2f}")
    
    output_path = "output_box_api.png"
    save_base64_image(data['processed_image'], output_path)
    
    return True

def test_comparison(input_image):
    """Test Level 1 vs Level 2 comparison"""
    print("\n" + "="*70)
    print("TEST 6: Level 1 vs Level 2 Comparison")
    print("="*70)
    
    base64_image = load_image_as_base64(input_image)
    
    levels = [1, 2]
    results = {}
    
    for level in levels:
        print(f"\nTesting Level {level}...")
        request_data = {
            "image": base64_image,
            "filter": "gaussian",
            "level": level,
            "sigma": 2.0,
            "radius": 3
        }
        
        response = requests.post(f"{API_URL}/api/process", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            results[level] = data['metrics']
            print(f"  Level {level}: {data['metrics']['time_ms']:.3f} ms")
        else:
            print(f"  ❌ Failed")
            return False
    
    # Compare
    if len(results) == 2:
        speedup = results[1]['time_ms'] / results[2]['time_ms']
        print(f"\n📊 Speedup (Level 2 vs Level 1): {speedup:.2f}x")
    
    return True

def main():
    """Run all tests"""
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <path_to_test_image>")
        print("Example: python test_client.py ../build/sample.jpg")
        sys.exit(1)
    
    input_image = sys.argv[1]
    
    if not Path(input_image).exists():
        print(f"Error: Image not found: {input_image}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("GPU IMAGE PROCESSING API - TEST SUITE")
    print("="*70)
    print(f"API URL: {API_URL}")
    print(f"Test Image: {input_image}")
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("List Filters", test_list_filters),
        ("Gaussian Blur", lambda: test_process_gaussian(input_image)),
        ("Box Blur", lambda: test_process_box(input_image)),
        ("Level Comparison", lambda: test_comparison(input_image))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())


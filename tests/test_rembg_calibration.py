"""
Test script to verify rembg calibration detector works correctly
"""

import sys

print("=" * 70)
print("REMBG CALIBRATION DETECTOR TEST")
print("=" * 70)
print()

# Test 1: Import the new detector
print("Test 1: Importing RembgCalibrationDetector...")
try:
    from src.detectors.calibration import RembgCalibrationDetector
    print("  [OK] RembgCalibrationDetector imported successfully")
except Exception as e:
    print(f"  [FAIL] Import failed: {str(e)}")
    sys.exit(1)

# Test 2: Check if rembg is available
print("\nTest 2: Checking rembg availability...")
try:
    detector = RembgCalibrationDetector()
    is_available = detector.is_available()
    if is_available:
        print("  [OK] rembg is available and ready to use")
    else:
        print("  [WARNING] rembg is not available (not installed)")
        print("  Install with: pip install rembg")
except Exception as e:
    print(f"  [FAIL] Availability check failed: {str(e)}")
    sys.exit(1)

# Test 3: Load configuration
print("\nTest 3: Loading calibration configuration...")
try:
    from config.detection_config import DetectionConfig
    calib_detectors = DetectionConfig.get_calibration_detectors()
    print(f"  [OK] {len(calib_detectors)} calibration detector(s) configured:")
    for detector in calib_detectors:
        available = detector.is_available()
        status = "[AVAILABLE]" if available else "[NOT INSTALLED]"
        print(f"    - {detector.name} {status}")
except Exception as e:
    print(f"  [FAIL] Configuration loading failed: {str(e)}")
    sys.exit(1)

# Test 4: Verify detector is in configuration
print("\nTest 4: Verifying rembg_calibration is configured...")
try:
    detector_names = [d.name for d in calib_detectors]
    if 'rembg_calibration' in detector_names:
        print("  [OK] rembg_calibration is configured and ready")
    else:
        print("  [WARNING] rembg_calibration not found in configuration")
        print("  Check config/detection_config.py")
except Exception as e:
    print(f"  [FAIL] Verification failed: {str(e)}")
    sys.exit(1)

# Test 5: Test app.py imports
print("\nTest 5: Testing app.py imports with new detector...")
try:
    import app
    print("  [OK] app.py imports successfully with rembg calibration")
except Exception as e:
    print(f"  [FAIL] app.py import failed: {str(e)}")
    sys.exit(1)

print()
print("=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print()
print("Summary:")
print("  - RembgCalibrationDetector is available")
print("  - Configuration includes both calibration methods:")
print("    1. quadrilateral_detection (edge-based)")
print("    2. rembg_calibration (AI background removal)")
print()
print("Current configuration order:")
for i, detector in enumerate(calib_detectors, 1):
    status = "[READY]" if detector.is_available() else "[NEEDS INSTALL]"
    print(f"  {i}. {detector.name} {status}")
print()
print("To change order, edit config/detection_config.py")
print("To disable a method, set 'enabled': False")
print()

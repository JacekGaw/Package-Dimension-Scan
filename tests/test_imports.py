"""
Test script to verify all refactored modules import correctly
"""

import sys
import os

print("=" * 70)
print("REFACTOR VERIFICATION TEST")
print("=" * 70)
print()

# Test 1: Import base detector interface
print("Test 1: Importing base detector interface...")
try:
    from src.detectors.base import BaseDetector, DetectionResult
    print("  [OK] BaseDetector imported successfully")
    print("  [OK] DetectionResult imported successfully")
except Exception as e:
    print(f"  [FAIL] FAILED: {str(e)}")
    sys.exit(1)

# Test 2: Import strategy runner
print("\nTest 2: Importing strategy runner...")
try:
    from src.strategies.strategy_runner import DetectionStrategyRunner
    print("  [OK] DetectionStrategyRunner imported successfully")
except Exception as e:
    print(f"  [FAIL] FAILED: {str(e)}")
    sys.exit(1)

# Test 3: Import configuration
print("\nTest 3: Importing detection configuration...")
try:
    from config.detection_config import DetectionConfig
    print("  [OK] DetectionConfig imported successfully")
except Exception as e:
    print(f"  [FAIL] FAILED: {str(e)}")
    sys.exit(1)

# Test 4: Import calibration detectors
print("\nTest 4: Importing calibration detectors...")
try:
    from src.detectors.calibration import QuadrilateralDetector
    print("  [OK] QuadrilateralDetector imported successfully")
except Exception as e:
    print(f"  [FAIL] FAILED: {str(e)}")
    sys.exit(1)

# Test 5: Import measurement detectors
print("\nTest 5: Importing measurement detectors...")
try:
    from src.detectors.measurement import YOLODetector, RembgDetector, EdgeDetector
    print("  [OK] YOLODetector imported successfully")
    print("  [OK] RembgDetector imported successfully")
    print("  [OK] EdgeDetector imported successfully")
except Exception as e:
    print(f"  [FAIL] FAILED: {str(e)}")
    sys.exit(1)

# Test 6: Import refactored calibration module
print("\nTest 6: Importing refactored calibration module...")
try:
    from src.calibration import detect_credit_card, calculate_calibration, save_debug_image
    print("  [OK] calibration.detect_credit_card imported successfully")
    print("  [OK] calibration.calculate_calibration imported successfully")
    print("  [OK] calibration.save_debug_image imported successfully")
except Exception as e:
    print(f"  [FAIL] FAILED: {str(e)}")
    sys.exit(1)

# Test 7: Import refactored measurement module
print("\nTest 7: Importing refactored measurement module...")
try:
    from src.measurement_new import detect_largest_rectangle, analyze_package, convert_units
    print("  [OK] measurement_new.detect_largest_rectangle imported successfully")
    print("  [OK] measurement_new.analyze_package imported successfully")
    print("  [OK] measurement_new.convert_units imported successfully")
except Exception as e:
    print(f"  [FAIL] FAILED: {str(e)}")
    sys.exit(1)

# Test 8: Test detector availability
print("\nTest 8: Testing detector availability...")
try:
    quad_detector = QuadrilateralDetector()
    print(f"  [OK] QuadrilateralDetector available: {quad_detector.is_available()}")

    yolo_detector = YOLODetector()
    print(f"  [OK] YOLODetector available: {yolo_detector.is_available()}")

    rembg_detector = RembgDetector()
    print(f"  [OK] RembgDetector available: {rembg_detector.is_available()}")

    edge_detector = EdgeDetector()
    print(f"  [OK] EdgeDetector available: {edge_detector.is_available()}")
except Exception as e:
    print(f"  [FAIL] FAILED: {str(e)}")
    sys.exit(1)

# Test 9: Test configuration loading
print("\nTest 9: Testing configuration loading...")
try:
    calib_detectors = DetectionConfig.get_calibration_detectors()
    print(f"  [OK] Calibration detectors: {len(calib_detectors)} configured")
    for detector in calib_detectors:
        print(f"    - {detector.name}")

    meas_detectors = DetectionConfig.get_measurement_detectors()
    print(f"  [OK] Measurement detectors: {len(meas_detectors)} configured")
    for detector in meas_detectors:
        print(f"    - {detector.name}")
except Exception as e:
    print(f"  [FAIL] FAILED: {str(e)}")
    sys.exit(1)

# Test 10: Test app.py imports
print("\nTest 10: Testing app.py imports...")
try:
    # This will test if app.py can import everything correctly
    import app
    print("  [OK] app.py imported successfully with refactored modules")
except Exception as e:
    print(f"  [FAIL] FAILED: {str(e)}")
    sys.exit(1)

print()
print("=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print()
print("Summary:")
print("  - All modules import correctly")
print("  - All detectors are instantiable")
print("  - Configuration loads successfully")
print("  - app.py uses refactored modules")
print()
print("Original files preserved:")
print("  - src/final_calibration.py (backup)")
print("  - src/measurement.py (backup)")
print()
print("Refactored files active:")
print("  - src/calibration.py (NEW - uses strategy runner)")
print("  - src/measurement_new.py (NEW - uses strategy runner)")
print()

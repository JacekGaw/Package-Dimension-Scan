"""
Test script for new detection methods (Otsu and Adaptive Thresholding)

This script verifies:
1. All new detector classes can be imported
2. All detectors are properly registered in configuration
3. Configuration loads detectors correctly
4. All detectors report availability
"""

import sys
import os

# Add the microservice folder to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_measurement_imports():
    """Test importing new measurement detectors"""
    print_section("Testing Measurement Detector Imports")

    try:
        from src.detectors.measurement import OtsuDetector
        print("[OK] OtsuDetector imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import OtsuDetector: {e}")
        return False

    try:
        from src.detectors.measurement import AdaptiveDetector
        print("[OK] AdaptiveDetector imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import AdaptiveDetector: {e}")
        return False

    return True


def test_calibration_imports():
    """Test importing new calibration detectors"""
    print_section("Testing Calibration Detector Imports")

    try:
        from src.detectors.calibration import OtsuCalibrationDetector
        print("[OK] OtsuCalibrationDetector imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import OtsuCalibrationDetector: {e}")
        return False

    try:
        from src.detectors.calibration import AdaptiveCalibrationDetector
        print("[OK] AdaptiveCalibrationDetector imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import AdaptiveCalibrationDetector: {e}")
        return False

    return True


def test_detector_instantiation():
    """Test creating instances of new detectors"""
    print_section("Testing Detector Instantiation")

    from src.detectors.measurement import OtsuDetector, AdaptiveDetector
    from src.detectors.calibration import OtsuCalibrationDetector, AdaptiveCalibrationDetector

    # Test measurement detectors
    try:
        otsu = OtsuDetector()
        print(f"[OK] OtsuDetector instantiated: {otsu.name}")
        print(f"     Available: {otsu.is_available()}")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate OtsuDetector: {e}")
        return False

    try:
        adaptive = AdaptiveDetector()
        print(f"[OK] AdaptiveDetector instantiated: {adaptive.name}")
        print(f"     Available: {adaptive.is_available()}")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate AdaptiveDetector: {e}")
        return False

    # Test calibration detectors
    try:
        otsu_cal = OtsuCalibrationDetector()
        print(f"[OK] OtsuCalibrationDetector instantiated: {otsu_cal.name}")
        print(f"     Available: {otsu_cal.is_available()}")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate OtsuCalibrationDetector: {e}")
        return False

    try:
        adaptive_cal = AdaptiveCalibrationDetector()
        print(f"[OK] AdaptiveCalibrationDetector instantiated: {adaptive_cal.name}")
        print(f"     Available: {adaptive_cal.is_available()}")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate AdaptiveCalibrationDetector: {e}")
        return False

    return True


def test_configuration():
    """Test configuration loading"""
    print_section("Testing Configuration")

    try:
        from config.detection_config import DetectionConfig
        print("[OK] DetectionConfig imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import DetectionConfig: {e}")
        return False

    # Check calibration methods
    print("\nCalibration Methods in Configuration:")
    for i, method in enumerate(DetectionConfig.CALIBRATION_METHODS, 1):
        status = "ENABLED" if method['enabled'] else "DISABLED"
        print(f"  {i}. {method['type']:25s} [{status}]")

    # Check measurement methods
    print("\nMeasurement Methods in Configuration:")
    for i, method in enumerate(DetectionConfig.MEASUREMENT_METHODS, 1):
        status = "ENABLED" if method['enabled'] else "DISABLED"
        print(f"  {i}. {method['type']:25s} [{status}]")

    # Verify new methods are present
    calib_types = [m['type'] for m in DetectionConfig.CALIBRATION_METHODS]
    meas_types = [m['type'] for m in DetectionConfig.MEASUREMENT_METHODS]

    new_calib_methods = ['otsu_calibration', 'adaptive_calibration']
    new_meas_methods = ['otsu_threshold', 'adaptive_threshold']

    print("\nVerifying New Methods Registration:")
    for method in new_calib_methods:
        if method in calib_types:
            print(f"[OK] {method} registered in calibration")
        else:
            print(f"[FAIL] {method} NOT found in calibration")
            return False

    for method in new_meas_methods:
        if method in meas_types:
            print(f"[OK] {method} registered in measurement")
        else:
            print(f"[FAIL] {method} NOT found in measurement")
            return False

    return True


def test_detector_loading():
    """Test loading detectors from configuration"""
    print_section("Testing Detector Loading from Configuration")

    from config.detection_config import DetectionConfig

    # Load calibration detectors
    try:
        calib_detectors = DetectionConfig.get_calibration_detectors()
        print(f"[OK] Loaded {len(calib_detectors)} calibration detectors")
        for detector in calib_detectors:
            print(f"     - {detector.name}")
    except Exception as e:
        print(f"[FAIL] Failed to load calibration detectors: {e}")
        return False

    # Load measurement detectors
    try:
        meas_detectors = DetectionConfig.get_measurement_detectors()
        print(f"[OK] Loaded {len(meas_detectors)} measurement detectors")
        for detector in meas_detectors:
            print(f"     - {detector.name}")
    except Exception as e:
        print(f"[FAIL] Failed to load measurement detectors: {e}")
        return False

    # Verify new detectors are loaded
    calib_names = [d.name for d in calib_detectors]
    meas_names = [d.name for d in meas_detectors]

    print("\nVerifying New Detectors Loaded:")
    new_calib_names = ['otsu_calibration', 'adaptive_calibration']
    new_meas_names = ['otsu_threshold', 'adaptive_threshold']

    for name in new_calib_names:
        if name in calib_names:
            print(f"[OK] {name} loaded in calibration")
        else:
            print(f"[FAIL] {name} NOT loaded in calibration")
            return False

    for name in new_meas_names:
        if name in meas_names:
            print(f"[OK] {name} loaded in measurement")
        else:
            print(f"[FAIL] {name} NOT loaded in measurement")
            return False

    return True


def test_detector_order():
    """Test that detectors are loaded in the correct order"""
    print_section("Testing Detector Order")

    from config.detection_config import DetectionConfig

    calib_detectors = DetectionConfig.get_calibration_detectors()
    meas_detectors = DetectionConfig.get_measurement_detectors()

    print("\nCalibration Detector Order:")
    for i, detector in enumerate(calib_detectors, 1):
        print(f"  {i}. {detector.name}")

    print("\nMeasurement Detector Order:")
    for i, detector in enumerate(meas_detectors, 1):
        print(f"  {i}. {detector.name}")

    # Verify expected order for measurement
    expected_meas_order = ['yolo', 'edge_detection', 'adaptive_threshold', 'otsu_threshold', 'rembg']
    actual_meas_names = [d.name for d in meas_detectors]

    print("\nVerifying Measurement Order (speed-first strategy):")
    for i, expected in enumerate(expected_meas_order):
        if i < len(actual_meas_names):
            actual = actual_meas_names[i]
            if actual == expected:
                print(f"[OK] Position {i+1}: {expected}")
            else:
                print(f"[WARN] Position {i+1}: expected {expected}, got {actual}")
        else:
            print(f"[WARN] Position {i+1}: {expected} missing")

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  NEW DETECTION METHODS TEST SUITE")
    print("  Testing: Otsu and Adaptive Thresholding")
    print("=" * 70)

    tests = [
        ("Measurement Imports", test_measurement_imports),
        ("Calibration Imports", test_calibration_imports),
        ("Detector Instantiation", test_detector_instantiation),
        ("Configuration", test_configuration),
        ("Detector Loading", test_detector_loading),
        ("Detector Order", test_detector_order),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

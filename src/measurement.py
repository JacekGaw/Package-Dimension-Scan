"""
Package Detection and Measurement Module (REFACTORED)

This is the refactored version that uses the strategy runner with configurable detectors.
The original file (measurement.py) has been preserved for reference.

Detection methods (configured in config/detection_config.py):
- YOLO - Fast object detection (50-200ms, works for 80 COCO classes)
- rembg - AI-powered background removal (90-95% accuracy, works on ANY object)
- Edge detection - Canny edges with contour finding (fallback)
"""

import cv2
import numpy as np
import os
import logging
from datetime import datetime

from src.strategies.strategy_runner import DetectionStrategyRunner
from config.detection_config import DetectionConfig

logger = logging.getLogger(__name__)

# Debug image directory
DEBUG_IMAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'temp_debug_images')
os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)


def detect_largest_rectangle(image, image_name="image"):
    """
    Detect package and return bounding rectangle dimensions.
    Works for ANY shape (rectangular, triangular, circular, irregular).
    Returns minimum area bounding rectangle - matches shipping carrier requirements.

    Uses detection strategy runner to try configured methods in order
    until one succeeds. Methods are configured in config/detection_config.py.

    Args:
        image: BGR input image (numpy array)
        image_name: Name for debug images (e.g., "top_view", "side_view")

    Returns:
        dict with pixel dimensions and debug info:
            - 'longer_side_px': Longer side in pixels
            - 'shorter_side_px': Shorter side in pixels
            - 'contour': Detected contour
            - 'rect': Minimum area bounding rectangle
            - 'area': Contour area
            - 'angle': Rotation angle
            - 'method': Detection method used
            - 'confidence': Detection confidence
            - 'debug_folder': Path to debug images
            - (optional) 'object_class': Object class if detected by YOLO

    Raises:
        ValueError: If no package detected by any method (PACKAGE_NOT_DETECTED)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    debug_folder = os.path.join(DEBUG_IMAGE_DIR, f"package_{image_name}_{timestamp}")
    os.makedirs(debug_folder, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"PACKAGE DETECTION - {image_name.upper()}")
    logger.info("=" * 70)

    # Save original
    cv2.imwrite(os.path.join(debug_folder, "1_original.jpg"), image)
    logger.info(f"Original image shape: {image.shape}")

    try:
        # Get configured detectors
        detectors = DetectionConfig.get_measurement_detectors()

        if not detectors:
            logger.error("No measurement detectors configured!")
            raise ValueError("PACKAGE_NOT_DETECTED")

        logger.info(f"Loaded {len(detectors)} measurement detector(s) from configuration")

        # Create strategy runner
        runner = DetectionStrategyRunner(detectors)

        # Run detection
        result = runner.run(image, debug_folder)

        logger.info("=" * 70)
        logger.info(f"SUCCESS: Package detected using {result.method_name}")
        logger.info(f"Dimensions: {result.metadata.get('longer_side_px', 0):.1f}px × {result.metadata.get('shorter_side_px', 0):.1f}px")
        logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info("=" * 70)

        # Convert to expected format
        return {
            'longer_side_px': result.metadata.get('longer_side_px'),
            'shorter_side_px': result.metadata.get('shorter_side_px'),
            'contour': result.contour,
            'rect': result.rect,
            'area': result.metadata.get('area'),
            'angle': result.metadata.get('angle'),
            'method': result.method_name,
            'confidence': result.confidence,
            'debug_folder': debug_folder,
            # Include optional metadata
            **{k: v for k, v in result.metadata.items()
               if k not in ['longer_side_px', 'shorter_side_px', 'area', 'angle']}
        }

    except ValueError as e:
        # All methods failed
        logger.error("=" * 70)
        logger.error(f"FAILED: Package not detected in {image_name}")
        logger.error("=" * 70)

        # Save failure visualization
        failure_vis = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(failure_vis, (10, 10), (500, 100), (0, 0, 0), -1)
        cv2.rectangle(failure_vis, (10, 10), (500, 100), (0, 0, 255), 3)
        cv2.putText(failure_vis, "PACKAGE NOT DETECTED", (20, 40), font, 0.8, (0, 0, 255), 2)
        cv2.putText(failure_vis, "All detection methods failed", (20, 70), font, 0.4, (200, 200, 200), 1)
        cv2.imwrite(os.path.join(debug_folder, "11_ALL_METHODS_FAILED.jpg"), failure_vis)

        raise ValueError("PACKAGE_NOT_DETECTED")

    except Exception as e:
        logger.error(f"Unexpected error during detection: {str(e)}", exc_info=True)
        raise ValueError("PACKAGE_NOT_DETECTED")


def analyze_package(top_view, side_view, pixels_per_mm, target_units='inches'):
    """
    Analyze package dimensions from two views with cross-validation

    This method:
    1. Detects package in both views
    2. Converts pixel dimensions to millimeters
    3. Cross-validates length between views
    4. Calculates confidence based on agreement
    5. Converts to target units

    Args:
        top_view: Top view image (numpy array)
        side_view: Side view image (numpy array)
        pixels_per_mm: Calibration factor from SellerSettings
        target_units: Output units (inches, centimeters, millimeters)

    Returns:
        dict with dimensions, confidence, measurement_method, debug_data:
            - 'dimensions': Final dimensions in target units
            - 'confidence': Overall confidence score
            - 'measurementMethod': 'two_view_cross_validation'
            - 'detectionMethods': Methods used for each view
            - 'rawMeasurements': Detailed measurement data
            - 'debugImages': Paths to debug images
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info("=" * 70)
    logger.info(f"[{timestamp}] PACKAGE ANALYSIS - TWO VIEW CROSS-VALIDATION")
    logger.info(f"[{timestamp}] Pixels per millimeter: {pixels_per_mm:.3f}")
    logger.info("=" * 70)

    try:
        # Step 1: Detect package in both views
        logger.info(f"[{timestamp}] Detecting package in TOP VIEW...")
        top_rect = detect_largest_rectangle(top_view, "top_view")

        logger.info(f"[{timestamp}] Detecting package in SIDE VIEW...")
        side_rect = detect_largest_rectangle(side_view, "side_view")

        # Step 2: Extract pixel dimensions
        top_long_px = top_rect['longer_side_px']
        top_short_px = top_rect['shorter_side_px']
        side_long_px = side_rect['longer_side_px']
        side_short_px = side_rect['shorter_side_px']

        logger.info(f"[{timestamp}] Top view:  {top_long_px:.1f}px × {top_short_px:.1f}px")
        logger.info(f"[{timestamp}] Side view: {side_long_px:.1f}px × {side_short_px:.1f}px")

        # Step 3: Convert to millimeters
        length_from_top_mm = top_long_px / pixels_per_mm
        width_from_top_mm = top_short_px / pixels_per_mm
        length_from_side_mm = side_long_px / pixels_per_mm
        height_from_side_mm = side_short_px / pixels_per_mm

        logger.info(f"[{timestamp}] Top view (mm):  Length={length_from_top_mm:.1f}, Width={width_from_top_mm:.1f}")
        logger.info(f"[{timestamp}] Side view (mm): Length={length_from_side_mm:.1f}, Height={height_from_side_mm:.1f}")

        # Step 4: Cross-validate length (should match between views)
        if length_from_top_mm > 0:
            length_discrepancy = abs(length_from_top_mm - length_from_side_mm) / length_from_top_mm
        else:
            length_discrepancy = 1.0

        logger.info(f"[{timestamp}] Length discrepancy: {length_discrepancy*100:.1f}%")

        # Step 5: Calculate confidence and final dimensions
        if length_discrepancy < 0.05:  # Within 5%
            confidence = 0.95
            final_length_mm = (length_from_top_mm + length_from_side_mm) / 2
            confidence_level = "HIGH"
        elif length_discrepancy < 0.10:  # Within 10%
            confidence = 0.85
            final_length_mm = (length_from_top_mm + length_from_side_mm) / 2
            confidence_level = "MEDIUM"
        else:  # >10% discrepancy
            confidence = 0.60
            final_length_mm = max(length_from_top_mm, length_from_side_mm)
            confidence_level = "LOW"

        logger.info(f"[{timestamp}] Confidence: {confidence:.2f} ({confidence_level})")
        logger.info(f"[{timestamp}] Final length (averaged): {final_length_mm:.1f}mm")

        # Step 6: Convert to target units
        dimensions = convert_units({
            'length': final_length_mm,
            'width': width_from_top_mm,
            'height': height_from_side_mm
        }, target_units)

        logger.info(f"[{timestamp}] Final dimensions ({target_units}):")
        logger.info(f"[{timestamp}]   Length: {dimensions['length']:.2f}")
        logger.info(f"[{timestamp}]   Width:  {dimensions['width']:.2f}")
        logger.info(f"[{timestamp}]   Height: {dimensions['height']:.2f}")

        # Step 7: Create comparison debug image
        comparison_path = save_comparison_debug_image(
            top_view, side_view,
            top_rect, side_rect,
            dimensions, confidence,
            timestamp
        )

        logger.info("=" * 70)
        logger.info(f"[{timestamp}] SUCCESS: Package measurement complete")
        logger.info("=" * 70)

        return {
            'dimensions': dimensions,
            'confidence': float(confidence),
            'measurementMethod': 'two_view_cross_validation',
            'detectionMethods': {
                'topView': top_rect.get('method', 'unknown'),
                'sideView': side_rect.get('method', 'unknown')
            },
            'rawMeasurements': {
                'topView': {
                    'length_mm': float(length_from_top_mm),
                    'width_mm': float(width_from_top_mm),
                    'length_px': float(top_long_px),
                    'width_px': float(top_short_px)
                },
                'sideView': {
                    'length_mm': float(length_from_side_mm),
                    'height_mm': float(height_from_side_mm),
                    'length_px': float(side_long_px),
                    'height_px': float(side_short_px)
                },
                'lengthDiscrepancy': float(length_discrepancy),
                'pixelsPerMillimeter': float(pixels_per_mm)
            },
            'debugImages': {
                'topView': top_rect.get('debug_folder'),
                'sideView': side_rect.get('debug_folder'),
                'comparison': comparison_path
            }
        }

    except Exception as e:
        logger.error(f"[{timestamp}] Package analysis failed: {str(e)}", exc_info=True)
        raise


def convert_units(dimensions_mm, target_units):
    """
    Convert dimensions from millimeters to target units

    Args:
        dimensions_mm: dict with length, width, height in millimeters
        target_units: 'inches', 'centimeters', or 'millimeters'

    Returns:
        dict with converted dimensions and units
    """
    if target_units == 'inches':
        return {
            'length': dimensions_mm['length'] / 25.4,
            'width': dimensions_mm['width'] / 25.4,
            'height': dimensions_mm['height'] / 25.4,
            'units': 'inches'
        }
    elif target_units == 'centimeters':
        return {
            'length': dimensions_mm['length'] / 10.0,
            'width': dimensions_mm['width'] / 10.0,
            'height': dimensions_mm['height'] / 10.0,
            'units': 'centimeters'
        }
    else:  # millimeters
        return {
            **dimensions_mm,
            'units': 'millimeters'
        }


def save_comparison_debug_image(top_view, side_view, top_rect, side_rect, dimensions, confidence, timestamp):
    """
    Create a side-by-side comparison image with both views and measurements

    Args:
        top_view: Top view image
        side_view: Side view image
        top_rect: Detection results for top view
        side_rect: Detection results for side view
        dimensions: Final calculated dimensions
        confidence: Confidence score
        timestamp: Timestamp for filename

    Returns:
        Path to saved comparison image
    """
    try:
        # Resize images to same height for side-by-side display
        max_height = 600

        # Resize top view
        top_h, top_w = top_view.shape[:2]
        top_scale = max_height / top_h
        top_resized = cv2.resize(top_view, (int(top_w * top_scale), max_height))

        # Resize side view
        side_h, side_w = side_view.shape[:2]
        side_scale = max_height / side_h
        side_resized = cv2.resize(side_view, (int(side_w * side_scale), max_height))

        # Draw detections on resized images
        # Top view
        top_vis = top_resized.copy()
        box_top = cv2.boxPoints(top_rect['rect'])
        box_top = (box_top * top_scale).astype(np.int32)
        cv2.drawContours(top_vis, [box_top], 0, (0, 255, 0), 3)

        # Side view
        side_vis = side_resized.copy()
        box_side = cv2.boxPoints(side_rect['rect'])
        box_side = (box_side * side_scale).astype(np.int32)
        cv2.drawContours(side_vis, [box_side], 0, (0, 255, 0), 3)

        # Combine side by side
        comparison = np.hstack([top_vis, side_vis])

        # Add text overlay with measurements
        font = cv2.FONT_HERSHEY_SIMPLEX
        overlay_height = 150
        overlay = np.zeros((overlay_height, comparison.shape[1], 3), dtype=np.uint8)

        # Title
        cv2.putText(overlay, "PACKAGE MEASUREMENT - CROSS-VALIDATION",
                   (20, 30), font, 0.7, (0, 255, 0), 2)

        # Dimensions
        dim_text = f"L: {dimensions['length']:.2f} {dimensions['units']}  |  " \
                   f"W: {dimensions['width']:.2f} {dimensions['units']}  |  " \
                   f"H: {dimensions['height']:.2f} {dimensions['units']}"
        cv2.putText(overlay, dim_text, (20, 65), font, 0.6, (255, 255, 255), 2)

        # Confidence
        confidence_color = (0, 255, 0) if confidence >= 0.85 else (0, 255, 255) if confidence >= 0.70 else (0, 165, 255)
        cv2.putText(overlay, f"Confidence: {confidence*100:.1f}%",
                   (20, 100), font, 0.6, confidence_color, 2)

        # Method
        cv2.putText(overlay, "Method: Two-view cross-validation",
                   (20, 130), font, 0.5, (200, 200, 200), 1)

        # Labels
        mid_x = comparison.shape[1] // 2
        cv2.putText(comparison, "TOP VIEW", (50, 30), font, 0.7, (255, 255, 0), 2)
        cv2.putText(comparison, "SIDE VIEW", (mid_x + 50, 30), font, 0.7, (255, 255, 0), 2)

        # Combine overlay with comparison
        final_image = np.vstack([overlay, comparison])

        # Save
        filename = f"measurement_comparison_{timestamp}.jpg"
        output_path = os.path.join(DEBUG_IMAGE_DIR, filename)
        cv2.imwrite(output_path, final_image)

        logger.info(f"Comparison debug image saved: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to save comparison image: {str(e)}")
        return None

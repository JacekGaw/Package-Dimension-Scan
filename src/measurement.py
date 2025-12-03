"""
Package Detection and Measurement Module
Detects packages and calculates dimensions using cross-validation
Includes extensive debug logging and image generation

Detection methods (tries in order):
1. YOLO - Fast object detection (50-200ms, works for 80 COCO classes)
2. rembg - AI-powered background removal (90-95% accuracy, works on ANY object)
3. Edge detection - Canny edges with contour finding (fallback)
"""

import cv2
import numpy as np
import os
import logging
from datetime import datetime
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Try to import YOLO (optional dependency)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    YOLO_MODEL = None  # Will be loaded on first use
    logger.info("YOLO library loaded successfully - Fast object detection enabled")
except ImportError:
    YOLO_AVAILABLE = False
    YOLO_MODEL = None
    logger.warning("YOLO not available - Install with: pip install ultralytics")

# Try to import rembg (optional dependency)
try:
    from rembg import remove
    REMBG_AVAILABLE = True
    logger.info("rembg library loaded successfully - AI background removal enabled")
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg not available - will use edge detection only. Install with: pip install rembg")

# Debug image directory
DEBUG_IMAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'temp_debug_images')
os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)


def detect_with_yolo(image, debug_folder):
    """
    Detect package using YOLO object detection with segmentation

    This is the FASTEST method (50-200ms on CPU) but only works for
    objects in the COCO dataset (80 classes: boxes, bottles, books, etc.)

    Args:
        image: BGR input image (numpy array)
        debug_folder: Path to save debug images

    Returns:
        dict with pixel dimensions, debug info, and object class

    Raises:
        ValueError: If no objects detected or YOLO not available
    """
    global YOLO_MODEL

    if not YOLO_AVAILABLE:
        raise ValueError("YOLO not available")

    logger.info("  Method: YOLO object detection")

    try:
        # Load model on first use (lazy loading)
        if YOLO_MODEL is None:
            logger.info("  Loading YOLO model (first time only)...")
            YOLO_MODEL = YOLO('yolov8n-seg.pt')  # Nano model - fastest
            logger.info("  YOLO model loaded")

        # Run inference
        logger.info("  Running YOLO inference...")
        results = YOLO_MODEL(image, verbose=False)

        if len(results) == 0 or results[0].masks is None:
            logger.warning("  YOLO: No objects detected")
            raise ValueError("No objects detected")

        # Get masks and boxes
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        if len(masks) == 0:
            logger.warning("  YOLO: No masks generated")
            raise ValueError("No masks generated")

        # Get the largest detected object
        areas = [cv2.countNonZero(mask.astype(np.uint8)) for mask in masks]
        largest_idx = np.argmax(areas)

        mask = masks[largest_idx]
        detected_class = int(classes[largest_idx])
        confidence = float(confidences[largest_idx])

        # Get class name
        class_names = results[0].names
        class_name = class_names[detected_class]

        logger.info(f"  YOLO detected: {class_name} (class {detected_class}) with confidence {confidence:.2f}")

        # Convert mask to uint8 and resize to image size
        mask_uint8 = (mask * 255).astype(np.uint8)
        if mask_uint8.shape != image.shape[:2]:
            mask_uint8 = cv2.resize(mask_uint8, (image.shape[1], image.shape[0]))

        cv2.imwrite(os.path.join(debug_folder, "2_yolo_mask.jpg"), mask_uint8)

        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(os.path.join(debug_folder, "3_yolo_mask_cleaned.jpg"), mask_cleaned)

        # Find contours
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("  YOLO: No contours found in mask")
            raise ValueError("No contours found")

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Check minimum area
        min_area = (image.shape[0] * image.shape[1]) * 0.01
        if area < min_area:
            logger.warning(f"  YOLO: Contour too small: {area:.0f}px² (min: {min_area:.0f}px²)")
            raise ValueError("Contour too small")

        # Get minimum area bounding rectangle
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        center, (width, height), angle = rect

        # Ensure width >= height
        longer_side_px = max(width, height)
        shorter_side_px = min(width, height)

        logger.info(f"  YOLO SUCCESS: {longer_side_px:.1f}px × {shorter_side_px:.1f}px")

        # Visualize detection
        result_vis = image.copy()
        cv2.drawContours(result_vis, [largest], -1, (0, 255, 0), 3)
        cv2.drawContours(result_vis, [box], 0, (255, 0, 0), 2)
        cv2.circle(result_vis, (int(center[0]), int(center[1])), 8, (0, 0, 255), -1)

        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(result_vis, (10, 10), (550, 150), (0, 0, 0), -1)
        cv2.rectangle(result_vis, (10, 10), (550, 150), (0, 255, 0), 3)
        cv2.putText(result_vis, f"YOLO DETECTION: {class_name.upper()}", (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(result_vis, f"Size: {longer_side_px:.0f} x {shorter_side_px:.0f} px",
                   (20, 75), font, 0.6, (255, 255, 255), 2)
        cv2.putText(result_vis, f"Area: {area:.0f} px² | Conf: {confidence:.2f}",
                   (20, 110), font, 0.5, (200, 200, 200), 1)
        cv2.putText(result_vis, f"Class: {detected_class} ({class_name})",
                   (20, 140), font, 0.5, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "4_YOLO_DETECTION.jpg"), result_vis)

        return {
            'longer_side_px': float(longer_side_px),
            'shorter_side_px': float(shorter_side_px),
            'contour': largest,
            'rect': rect,
            'area': float(area),
            'angle': float(angle),
            'method': 'yolo',
            'object_class': class_name,
            'object_class_id': detected_class,
            'confidence': confidence,
            'debug_folder': debug_folder
        }

    except Exception as e:
        logger.warning(f"  YOLO detection failed: {str(e)}")
        raise ValueError(f"YOLO failed: {str(e)}")


def detect_with_rembg(image, debug_folder):
    """
    Detect package using rembg AI background removal

    This method uses a pre-trained U2-Net model to automatically
    remove the background and isolate the foreground object.

    Args:
        image: BGR input image (numpy array)
        debug_folder: Path to save debug images

    Returns:
        dict with pixel dimensions and debug info

    Raises:
        ValueError: If no package detected or rembg not available
    """
    if not REMBG_AVAILABLE:
        raise ValueError("rembg not available")

    logger.info("  Method: rembg AI background removal")

    try:
        # Convert OpenCV BGR to PIL RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Apply rembg to get mask only
        logger.info("  Running rembg background removal...")
        mask_pil = remove(image_pil, only_mask=True)

        # Convert PIL mask to numpy
        mask = np.array(mask_pil)

        # Save mask
        cv2.imwrite(os.path.join(debug_folder, "2_rembg_mask.jpg"), mask)
        logger.info("  rembg mask generated")

        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(os.path.join(debug_folder, "3_rembg_mask_cleaned.jpg"), mask_cleaned)

        # Find contours
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("  rembg: No contours found in mask")
            raise ValueError("No contours found")

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Check minimum area
        min_area = (image.shape[0] * image.shape[1]) * 0.01  # At least 1% of image
        if area < min_area:
            logger.warning(f"  rembg: Contour too small: {area:.0f}px² (min: {min_area:.0f}px²)")
            raise ValueError("Contour too small")

        # Get minimum area bounding rectangle
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        center, (width, height), angle = rect

        # Ensure width >= height
        longer_side_px = max(width, height)
        shorter_side_px = min(width, height)

        logger.info(f"  rembg SUCCESS: {longer_side_px:.1f}px × {shorter_side_px:.1f}px")

        # Visualize detection
        result_vis = image.copy()
        cv2.drawContours(result_vis, [largest], -1, (0, 255, 0), 3)
        cv2.drawContours(result_vis, [box], 0, (255, 0, 0), 2)
        cv2.circle(result_vis, (int(center[0]), int(center[1])), 8, (0, 0, 255), -1)

        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(result_vis, (10, 10), (500, 130), (0, 0, 0), -1)
        cv2.rectangle(result_vis, (10, 10), (500, 130), (0, 255, 0), 3)
        cv2.putText(result_vis, "REMBG DETECTION SUCCESS", (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(result_vis, f"Size: {longer_side_px:.0f} x {shorter_side_px:.0f} px",
                   (20, 75), font, 0.6, (255, 255, 255), 2)
        cv2.putText(result_vis, f"Area: {area:.0f} px²",
                   (20, 105), font, 0.5, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "4_REMBG_DETECTION.jpg"), result_vis)

        return {
            'longer_side_px': float(longer_side_px),
            'shorter_side_px': float(shorter_side_px),
            'contour': largest,
            'rect': rect,
            'area': float(area),
            'angle': float(angle),
            'method': 'rembg',
            'debug_folder': debug_folder
        }

    except Exception as e:
        logger.warning(f"  rembg detection failed: {str(e)}")
        raise ValueError(f"rembg failed: {str(e)}")


def detect_with_edge_detection(image, debug_folder):
    """
    Detect package using traditional edge detection (fallback method)

    This method uses CLAHE enhancement, Gaussian blur, and Canny edge detection.

    Args:
        image: BGR input image (numpy array)
        debug_folder: Path to save debug images

    Returns:
        dict with pixel dimensions and debug info

    Raises:
        ValueError: If no package detected
    """
    logger.info("  Method: Edge detection (Canny)")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(debug_folder, "5_grayscale.jpg"), gray)

    # Enhance with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    cv2.imwrite(os.path.join(debug_folder, "6_clahe_enhanced.jpg"), enhanced)
    logger.info("  CLAHE enhancement applied")

    # Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    cv2.imwrite(os.path.join(debug_folder, "7_gaussian_blur.jpg"), blurred)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite(os.path.join(debug_folder, "8_canny_edges.jpg"), edges)
    logger.info("  Canny edge detection applied")

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.error("  Edge detection: No contours found!")
        raise ValueError("PACKAGE_NOT_DETECTED")

    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    logger.info(f"  Edge detection: Found {len(contours)} contours")

    # Visualize all contours
    all_contours_vis = image.copy()
    cv2.drawContours(all_contours_vis, contours[:10], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(debug_folder, "9_all_contours.jpg"), all_contours_vis)

    # Get largest contour (the package)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Filter out tiny contours (noise)
    min_area = (image.shape[0] * image.shape[1]) * 0.01  # At least 1% of image
    if area < min_area:
        logger.error(f"  Edge detection: Largest contour too small: {area:.0f}px² (min: {min_area:.0f}px²)")
        raise ValueError("PACKAGE_NOT_DETECTED")

    logger.info(f"  Edge detection: Largest contour area: {area:.0f}px²")

    # Get minimum area bounding rectangle
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    center, (width, height), angle = rect

    # Ensure width >= height
    longer_side_px = max(width, height)
    shorter_side_px = min(width, height)

    logger.info(f"  Edge detection: Bounding rectangle - {longer_side_px:.1f}px × {shorter_side_px:.1f}px")
    logger.info(f"           Center: ({center[0]:.1f}, {center[1]:.1f}), Angle: {angle:.1f}°")

    # Visualize detection
    result_vis = image.copy()
    cv2.drawContours(result_vis, [largest], -1, (0, 255, 0), 3)
    cv2.drawContours(result_vis, [box], 0, (255, 0, 0), 2)
    cv2.circle(result_vis, (int(center[0]), int(center[1])), 8, (0, 0, 255), -1)

    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(result_vis, (10, 10), (500, 130), (0, 0, 0), -1)
    cv2.rectangle(result_vis, (10, 10), (500, 130), (0, 255, 0), 3)
    cv2.putText(result_vis, "EDGE DETECTION SUCCESS", (20, 40), font, 0.7, (0, 255, 0), 2)
    cv2.putText(result_vis, f"Size: {longer_side_px:.0f} x {shorter_side_px:.0f} px",
               (20, 75), font, 0.6, (255, 255, 255), 2)
    cv2.putText(result_vis, f"Area: {area:.0f} px²",
               (20, 105), font, 0.5, (200, 200, 200), 1)

    cv2.imwrite(os.path.join(debug_folder, "10_EDGE_DETECTION.jpg"), result_vis)

    return {
        'longer_side_px': float(longer_side_px),
        'shorter_side_px': float(shorter_side_px),
        'contour': largest,
        'rect': rect,
        'area': float(area),
        'angle': float(angle),
        'method': 'edge_detection',
        'debug_folder': debug_folder
    }


def detect_largest_rectangle(image, image_name="image"):
    """
    Detect package and return bounding rectangle dimensions.
    Works for ANY shape (rectangular, triangular, circular, irregular).
    Returns minimum area bounding rectangle - matches shipping carrier requirements.

    Detection Strategy (tries in order):
    1. YOLO object detection - Fastest (50-200ms), works for 80 COCO classes
    2. rembg AI background removal - Most accurate (90-95%), works on ANY object
    3. Edge detection - Last resort fallback (60-75%)

    Args:
        image: BGR input image (numpy array)
        image_name: Name for debug images (e.g., "top_view", "side_view")

    Returns:
        dict with pixel dimensions and debug info
            - 'method': 'yolo', 'rembg', or 'edge_detection'
            - 'object_class': (optional) object class name if detected by YOLO

    Raises:
        ValueError: If no package detected by any method
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

    detection_result = None
    methods_tried = []

    # ----------------------------
    # Method 1: Try YOLO (fastest, but only for known objects)
    # ----------------------------
    if YOLO_AVAILABLE:
        logger.info("Attempting detection with YOLO (fast object detection)...")
        try:
            detection_result = detect_with_yolo(image, debug_folder)
            logger.info("[SUCCESS] YOLO detection successful!")
        except Exception as e:
            logger.info(f"[FAILED] YOLO detection failed: {str(e)}")
            methods_tried.append(f"yolo: {str(e)}")
    else:
        logger.info("YOLO not available, skipping")
        methods_tried.append("yolo: not installed")

    # ----------------------------
    # Method 2: Try rembg (most accurate, works on any object)
    # ----------------------------
    if detection_result is None and REMBG_AVAILABLE:
        logger.info("Attempting detection with rembg (AI background removal)...")
        try:
            detection_result = detect_with_rembg(image, debug_folder)
            logger.info("[SUCCESS] rembg detection successful!")
        except Exception as e:
            logger.warning(f"[FAILED] rembg detection failed: {str(e)}")
            methods_tried.append(f"rembg: {str(e)}")
    elif detection_result is None and not REMBG_AVAILABLE:
        logger.info("rembg not available, skipping AI detection")
        methods_tried.append("rembg: not installed")

    # ----------------------------
    # Method 3: Fallback to edge detection
    # ----------------------------
    if detection_result is None:
        logger.info("Attempting detection with edge detection (last resort)...")
        try:
            detection_result = detect_with_edge_detection(image, debug_folder)
            logger.info("[SUCCESS] Edge detection successful!")
        except Exception as e:
            logger.error(f"[FAILED] Edge detection failed: {str(e)}")
            methods_tried.append(f"edge_detection: {str(e)}")

    # ----------------------------
    # Final result
    # ----------------------------
    if detection_result is None:
        # All methods failed
        logger.error("=" * 70)
        logger.error(f"FAILED: Package not detected in {image_name}")
        logger.error(f"Methods tried: {', '.join(methods_tried)}")
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

    # Success!
    logger.info("=" * 70)
    logger.info(f"SUCCESS: Package detected in {image_name}")
    logger.info(f"Detection method used: {detection_result['method']}")
    logger.info(f"Dimensions: {detection_result['longer_side_px']:.1f}px × {detection_result['shorter_side_px']:.1f}px")
    logger.info("=" * 70)

    return detection_result


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
        dict with dimensions, confidence, measurement_method, debug_data
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info("=" * 70)
    logger.info(f"[{timestamp}] PACKAGE ANALYSIS - TWO VIEW CROSS-VALIDATION")
    logger.info(f"[{timestamp}] Pixels per millimeter: {pixels_per_mm:.3f}")
    logger.info("=" * 70)

    try:
        # ----------------------------
        # Step 1: Detect package in both views
        # ----------------------------
        logger.info(f"[{timestamp}] Detecting package in TOP VIEW...")
        top_rect = detect_largest_rectangle(top_view, "top_view")

        logger.info(f"[{timestamp}] Detecting package in SIDE VIEW...")
        side_rect = detect_largest_rectangle(side_view, "side_view")

        # ----------------------------
        # Step 2: Extract pixel dimensions
        # ----------------------------
        top_long_px = top_rect['longer_side_px']
        top_short_px = top_rect['shorter_side_px']
        side_long_px = side_rect['longer_side_px']
        side_short_px = side_rect['shorter_side_px']

        logger.info(f"[{timestamp}] Top view:  {top_long_px:.1f}px × {top_short_px:.1f}px")
        logger.info(f"[{timestamp}] Side view: {side_long_px:.1f}px × {side_short_px:.1f}px")

        # ----------------------------
        # Step 3: Convert to millimeters
        # ----------------------------
        length_from_top_mm = top_long_px / pixels_per_mm
        width_from_top_mm = top_short_px / pixels_per_mm
        length_from_side_mm = side_long_px / pixels_per_mm
        height_from_side_mm = side_short_px / pixels_per_mm

        logger.info(f"[{timestamp}] Top view (mm):  Length={length_from_top_mm:.1f}, Width={width_from_top_mm:.1f}")
        logger.info(f"[{timestamp}] Side view (mm): Length={length_from_side_mm:.1f}, Height={height_from_side_mm:.1f}")

        # ----------------------------
        # Step 4: Cross-validate length (should match between views)
        # ----------------------------
        if length_from_top_mm > 0:
            length_discrepancy = abs(length_from_top_mm - length_from_side_mm) / length_from_top_mm
        else:
            length_discrepancy = 1.0

        logger.info(f"[{timestamp}] Length discrepancy: {length_discrepancy*100:.1f}%")

        # ----------------------------
        # Step 5: Calculate confidence and final dimensions
        # ----------------------------
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

        # ----------------------------
        # Step 6: Convert to target units
        # ----------------------------
        dimensions = convert_units({
            'length': final_length_mm,
            'width': width_from_top_mm,
            'height': height_from_side_mm
        }, target_units)

        logger.info(f"[{timestamp}] Final dimensions ({target_units}):")
        logger.info(f"[{timestamp}]   Length: {dimensions['length']:.2f}")
        logger.info(f"[{timestamp}]   Width:  {dimensions['width']:.2f}")
        logger.info(f"[{timestamp}]   Height: {dimensions['height']:.2f}")

        # ----------------------------
        # Step 7: Create comparison debug image
        # ----------------------------
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

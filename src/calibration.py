"""
Credit Card Calibration Module (REFACTORED)

This is the refactored version that uses the strategy runner with configurable detectors.
The original file (final_calibration.py) has been preserved for reference.
"""

import cv2
import numpy as np
import os
import logging
from datetime import datetime

from src.strategies.strategy_runner import DetectionStrategyRunner
from config.detection_config import DetectionConfig

logger = logging.getLogger(__name__)

# Credit card standard dimensions (ISO/IEC 7810 ID-1)
CREDIT_CARD_WIDTH_MM = 85.60
CREDIT_CARD_HEIGHT_MM = 53.98
CREDIT_CARD_ASPECT_RATIO = CREDIT_CARD_WIDTH_MM / CREDIT_CARD_HEIGHT_MM  # 1.586

# Debug image directory
DEBUG_IMAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'temp_debug_images')
os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)


def detect_credit_card(image):
    """
    Detect credit card using configured detection methods

    Uses the detection strategy runner to try configured methods in order
    until one succeeds. Methods are configured in config/detection_config.py.

    Args:
        image: BGR input image (numpy array)

    Returns:
        (contour, rect, method_name, confidence) or (None, None, None, 0)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    debug_folder = os.path.join(DEBUG_IMAGE_DIR, f"calibration_{timestamp}")
    os.makedirs(debug_folder, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CREDIT CARD CALIBRATION - Starting detection")
    logger.info("=" * 70)

    try:
        # Get configured detectors
        detectors = DetectionConfig.get_calibration_detectors()

        if not detectors:
            logger.error("No calibration detectors configured!")
            return None, None, None, 0

        logger.info(f"Loaded {len(detectors)} calibration detector(s) from configuration")

        # Create strategy runner
        runner = DetectionStrategyRunner(detectors)

        # Run detection
        result = runner.run(image, debug_folder)

        logger.info("=" * 70)
        logger.info(f"SUCCESS: Card detected using {result.method_name}")
        logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info("=" * 70)

        return (
            result.contour,
            result.rect,
            result.method_name,
            result.confidence
        )

    except ValueError as e:
        # All detectors failed
        logger.error("=" * 70)
        logger.error("FAILED: No credit card detected by any method")
        logger.error("=" * 70)
        return None, None, None, 0

    except Exception as e:
        logger.error(f"Unexpected error during detection: {str(e)}", exc_info=True)
        return None, None, None, 0


def calculate_calibration(contour, rect, image):
    """
    Calculate pixels per millimeter from detected credit card

    Uses perspective-corrected measurements for accurate results
    even when card is rotated or has perspective distortion.

    Args:
        contour: Detected card contour
        rect: Tuple (center, (width, height), angle) with perspective-corrected measurements
        image: Original image

    Returns:
        Calibration data dictionary with:
            - pixelsPerMillimeter: Calibration factor
            - confidence: Detection confidence
            - referenceType: 'credit_card'
            - imageDimensions: Original image size
            - cardDimensions: Detected card size in pixels

    Raises:
        ValueError: If contour or rect is None (REFERENCE_NOT_FOUND)
    """
    if contour is None or rect is None:
        raise ValueError("REFERENCE_NOT_FOUND")

    width, height = rect[1]

    if width == 0 or height == 0:
        raise ValueError("REFERENCE_NOT_FOUND")

    # Card dimensions in pixels
    card_width_px = max(width, height)
    card_height_px = min(width, height)

    # Calculate pixels per millimeter (average of width and height)
    px_per_mm_w = card_width_px / CREDIT_CARD_WIDTH_MM
    px_per_mm_h = card_height_px / CREDIT_CARD_HEIGHT_MM
    pixels_per_mm = (px_per_mm_w + px_per_mm_h) / 2

    # Calculate confidence based on aspect ratio
    aspect_ratio = card_width_px / card_height_px
    aspect_diff = abs(aspect_ratio - CREDIT_CARD_ASPECT_RATIO) / CREDIT_CARD_ASPECT_RATIO
    confidence = max(0.5, 1.0 - aspect_diff)

    # Image dimensions
    img_height, img_width = image.shape[:2]

    logger.info(f"Calibration calculated: {pixels_per_mm:.3f} px/mm")
    logger.info(f"Card dimensions: {card_width_px:.1f}px × {card_height_px:.1f}px")
    logger.info(f"Aspect ratio: {aspect_ratio:.3f} (expected: {CREDIT_CARD_ASPECT_RATIO:.3f})")

    return {
        'pixelsPerMillimeter': float(pixels_per_mm),
        'confidence': float(confidence),
        'referenceType': 'credit_card',
        'imageDimensions': {
            'width': img_width,
            'height': img_height
        },
        'cardDimensions': {
            'width_px': float(card_width_px),
            'height_px': float(card_height_px),
            'aspect_ratio': float(aspect_ratio)
        }
    }


def save_debug_image(image, contour, rect, filename, calibration_result):
    """
    Save debug image with detection annotations

    Args:
        image: Original BGR image
        contour: Detected contour (or None)
        rect: minAreaRect (or None)
        filename: Output filename
        calibration_result: Calibration result dict (or None)

    Returns:
        Path to saved debug image
    """
    try:
        debug_image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        if contour is not None and rect is not None:
            # Draw contour
            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 3)

            # Draw rectangle
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(debug_image, [box], 0, (255, 0, 0), 2)

            # Add text
            if calibration_result:
                cv2.rectangle(debug_image, (10, 10), (450, 150), (0, 0, 0), -1)
                cv2.rectangle(debug_image, (10, 10), (450, 150), (0, 255, 0), 2)

                cv2.putText(debug_image, "CARD DETECTED", (20, 40), font, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_image, f"PPM: {calibration_result['pixelsPerMillimeter']:.2f}",
                           (20, 75), font, 0.6, (255, 255, 255), 2)
                cv2.putText(debug_image, f"Confidence: {calibration_result['confidence']*100:.1f}%",
                           (20, 110), font, 0.6, (255, 255, 255), 2)

                if 'detectionMethod' in calibration_result:
                    method = calibration_result['detectionMethod']
                    cv2.putText(debug_image, f"Method: {method}",
                               (20, 140), font, 0.5, (255, 255, 0), 1)
        else:
            # No detection
            cv2.rectangle(debug_image, (10, 10), (400, 60), (0, 0, 0), -1)
            cv2.rectangle(debug_image, (10, 10), (400, 60), (0, 0, 255), 2)
            cv2.putText(debug_image, "NO CARD DETECTED", (20, 40), font, 0.7, (0, 0, 255), 2)

        # Save
        output_path = os.path.join(DEBUG_IMAGE_DIR, filename)
        cv2.imwrite(output_path, debug_image)
        logger.info(f"Debug image saved: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Failed to save debug image: {str(e)}")
        return None

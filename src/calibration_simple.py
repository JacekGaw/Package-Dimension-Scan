"""
Simplified Credit Card Detection
Based on OpenCV best practices for contour detection

Simple and effective approach:
1. Preprocess (grayscale + enhancement)
2. Threshold or Canny
3. Find contours
4. Filter by area and aspect ratio
5. Validate and return best match
"""

import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Credit card standard dimensions (ISO/IEC 7810 ID-1)
CREDIT_CARD_WIDTH_MM = 85.60
CREDIT_CARD_HEIGHT_MM = 53.98
CREDIT_CARD_ASPECT_RATIO = CREDIT_CARD_WIDTH_MM / CREDIT_CARD_HEIGHT_MM  # 1.586

# Create temp folder for debug images
DEBUG_IMAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'temp_debug_images')
os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)


def preprocess_image(image):
    """
    Simple preprocessing: grayscale + CLAHE for contrast enhancement

    Args:
        image: BGR image

    Returns:
        Enhanced grayscale image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    logger.debug("Applied preprocessing: grayscale + CLAHE")
    return enhanced


def detect_with_threshold(gray):
    """
    Method 1: Binary thresholding + contours
    Good for high contrast scenarios

    Args:
        gray: Grayscale image

    Returns:
        Contours found
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding (handles varying lighting)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger.debug(f"Threshold method: found {len(contours)} contours")
    return contours, binary


def detect_with_canny(gray):
    """
    Method 2: Canny edge detection + contours
    Good for clear edges

    Args:
        gray: Grayscale image

    Returns:
        Contours found
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger.debug(f"Canny method: found {len(contours)} contours")
    return contours, edges


def is_credit_card_contour(contour, image_shape):
    """
    Simple validation: check if contour matches credit card properties

    Args:
        contour: Contour to validate
        image_shape: Image dimensions

    Returns:
        (is_valid, confidence, rect)
    """
    # Get minimum area rectangle
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]

    if width == 0 or height == 0:
        return False, 0, None

    # Calculate aspect ratio
    aspect_ratio = max(width, height) / min(width, height)

    # Check aspect ratio (1.4 to 1.9 for credit card)
    if not (1.4 <= aspect_ratio <= 1.9):
        return False, 0, None

    # Check area (should be 5-70% of image)
    area = cv2.contourArea(contour)
    image_area = image_shape[0] * image_shape[1]
    area_ratio = area / image_area

    if not (0.05 <= area_ratio <= 0.70):
        return False, 0, None

    # Check minimum area
    if area < 5000:
        return False, 0, None

    # Calculate confidence based on aspect ratio match
    aspect_diff = abs(aspect_ratio - CREDIT_CARD_ASPECT_RATIO)
    confidence = max(0.5, 1.0 - aspect_diff / 0.5)

    # Boost confidence if area is in optimal range
    if 0.15 <= area_ratio <= 0.50:
        confidence *= 1.1

    confidence = min(1.0, confidence)

    logger.debug(f"Contour validation: aspect={aspect_ratio:.2f}, area_ratio={area_ratio:.3f}, confidence={confidence:.3f}")

    return True, confidence, rect


def detect_credit_card(image):
    """
    Main detection function - tries multiple simple methods

    Args:
        image: BGR image

    Returns:
        (contour, rect, method_name, confidence) or (None, None, None, 0)
    """
    logger.info("=" * 60)
    logger.info("Starting simplified credit card detection")
    logger.info("=" * 60)

    # Preprocess
    gray = preprocess_image(image)

    # Try Method 1: Adaptive Thresholding
    logger.info("Method 1: Adaptive Thresholding")
    contours_thresh, binary = detect_with_threshold(gray)

    best_contour = None
    best_rect = None
    best_confidence = 0
    best_method = None

    # Validate contours from thresholding
    for contour in contours_thresh:
        is_valid, confidence, rect = is_credit_card_contour(contour, image.shape)
        if is_valid and confidence > best_confidence:
            best_contour = contour
            best_rect = rect
            best_confidence = confidence
            best_method = "adaptive_threshold"

    # Try Method 2: Canny Edge Detection
    logger.info("Method 2: Canny Edge Detection")
    contours_canny, edges = detect_with_canny(gray)

    # Validate contours from Canny
    for contour in contours_canny:
        is_valid, confidence, rect = is_credit_card_contour(contour, image.shape)
        if is_valid and confidence > best_confidence:
            best_contour = contour
            best_rect = rect
            best_confidence = confidence
            best_method = "canny_edges"

    if best_contour is not None:
        logger.info(f"Card detected! Method: {best_method}, Confidence: {best_confidence:.3f}")
    else:
        logger.warning("No credit card detected")

    return best_contour, best_rect, best_method, best_confidence


def calculate_calibration(contour, rect, image):
    """
    Calculate pixels per millimeter from detected credit card

    Args:
        contour: Detected card contour
        rect: minAreaRect
        image: Original image

    Returns:
        Calibration data dictionary
    """
    if contour is None or rect is None:
        raise ValueError("REFERENCE_NOT_FOUND")

    width, height = rect[1]

    if width == 0 or height == 0:
        raise ValueError("REFERENCE_NOT_FOUND")

    card_width_px = max(width, height)
    card_height_px = min(width, height)

    # Calculate pixels per millimeter
    pixels_per_mm = card_width_px / CREDIT_CARD_WIDTH_MM

    # Calculate confidence based on aspect ratio
    aspect_ratio = card_width_px / card_height_px
    aspect_diff = abs(aspect_ratio - CREDIT_CARD_ASPECT_RATIO) / CREDIT_CARD_ASPECT_RATIO
    confidence = max(0.5, 1.0 - aspect_diff)

    # Get image dimensions
    img_height, img_width = image.shape[:2]

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
    """
    try:
        debug_image = image.copy()

        if contour is not None and rect is not None:
            # Draw contour in green
            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 3)

            # Draw rectangle in blue
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(debug_image, [box], 0, (255, 0, 0), 2)

            # Add text
            if calibration_result:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(debug_image, (10, 10), (400, 120), (0, 0, 0), -1)
                cv2.rectangle(debug_image, (10, 10), (400, 120), (0, 255, 0), 2)

                cv2.putText(debug_image, "CARD DETECTED", (20, 40),
                           font, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_image, f"PPM: {calibration_result['pixelsPerMillimeter']:.2f}",
                           (20, 70), font, 0.6, (255, 255, 255), 2)
                cv2.putText(debug_image, f"Confidence: {calibration_result['confidence']*100:.1f}%",
                           (20, 100), font, 0.6, (255, 255, 255), 2)
        else:
            # No detection
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(debug_image, (10, 10), (400, 60), (0, 0, 0), -1)
            cv2.rectangle(debug_image, (10, 10), (400, 60), (0, 0, 255), 2)
            cv2.putText(debug_image, "CARD NOT DETECTED",
                       (20, 40), font, 0.7, (0, 0, 255), 2)

        # Save
        output_path = os.path.join(DEBUG_IMAGE_DIR, filename)
        cv2.imwrite(output_path, debug_image)
        logger.info(f"Debug image saved: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Failed to save debug image: {str(e)}")
        return None

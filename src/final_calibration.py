"""
Credit Card Detection Using Quadrilateral Detection
Based on user's proposition - looks for 4-sided polygons with credit card aspect ratio
"""

import cv2
import numpy as np
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Credit card standard dimensions (ISO/IEC 7810 ID-1)
CREDIT_CARD_WIDTH_MM = 85.60
CREDIT_CARD_HEIGHT_MM = 53.98
CREDIT_CARD_ASPECT_RATIO = CREDIT_CARD_WIDTH_MM / CREDIT_CARD_HEIGHT_MM  # 1.586

# Debug image directory
DEBUG_IMAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'temp_debug_images')
os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)


def order_points(pts):
    """
    Consistently order 4 corner points: [top-left, top-right, bottom-right, bottom-left]

    Args:
        pts: Array of 4 points

    Returns:
        Ordered array of 4 points
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def perspective_transform_and_measure(image, corners):
    """
    Apply perspective transform to get accurate width/height measurements

    This is MORE accurate than minAreaRect because:
    - Measures actual quadrilateral, not bounding box
    - Not affected by rotation artifacts
    - Handles perspective distortion correctly

    Args:
        image: Source image
        corners: 4 corner points of the card

    Returns:
        (width_px, height_px, warped_card_image)
    """
    # Order points
    pts = order_points(corners)
    tl, tr, br, bl = pts

    # Calculate width and height from the actual quadrilateral edges
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width_px = max(width_top, width_bottom)

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height_px = max(height_left, height_right)

    # Destination points for perspective transform
    dst = np.array([
        [0, 0],
        [width_px - 1, 0],
        [width_px - 1, height_px - 1],
        [0, height_px - 1]
    ], dtype="float32")

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst)

    # Apply perspective transform
    warped = cv2.warpPerspective(image, M, (int(width_px), int(height_px)))

    return width_px, height_px, warped


def detect_credit_card(image):
    """
    Detect credit card using quadrilateral detection

    This method:
    1. Enhances image quality (CLAHE + Gaussian blur)
    2. Creates binary threshold
    3. Applies morphological operations
    4. Detects edges with Canny
    5. Finds 4-sided polygons
    6. Filters by aspect ratio

    Args:
        image: BGR input image (numpy array)

    Returns:
        (contour, rect, method_name, confidence) or (None, None, None, 0)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    debug_folder = os.path.join(DEBUG_IMAGE_DIR, f"detection_{timestamp}")
    os.makedirs(debug_folder, exist_ok=True)

    logger.info("=" * 70)
    logger.info("QUADRILATERAL CREDIT CARD DETECTION")
    logger.info("Looking for 4-sided polygon with credit card aspect ratio")
    logger.info("=" * 70)

    try:
        # Save original
        cv2.imwrite(os.path.join(debug_folder, "1_original.jpg"), image)

        # ----------------------------
        # Step 1: Convert to grayscale
        # ----------------------------
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(debug_folder, "2_grayscale.jpg"), gray)

        # ----------------------------
        # Step 2: Improve image quality with CLAHE
        # ----------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        cv2.imwrite(os.path.join(debug_folder, "3_clahe_enhanced.jpg"), enhanced)
        logger.info("Step 2: CLAHE enhancement applied")

        # ----------------------------
        # Step 3: Gaussian Blur to reduce noise
        # ----------------------------
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        cv2.imwrite(os.path.join(debug_folder, "4_gaussian_blur.jpg"), blurred)
        logger.info("Step 3: Gaussian blur applied")

        # ----------------------------
        # Step 4: Adaptive threshold for strong edges
        # ----------------------------
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            4
        )
        cv2.imwrite(os.path.join(debug_folder, "5_adaptive_threshold.jpg"), thresh)
        logger.info("Step 4: Adaptive threshold applied")

        # ----------------------------
        # Step 5: Morphology to close gaps in edges
        # ----------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(os.path.join(debug_folder, "6_morphology_close.jpg"), morphed)
        logger.info("Step 5: Morphological closing applied")

        # ----------------------------
        # Step 6: Canny edge detection
        # ----------------------------
        edges = cv2.Canny(morphed, 50, 200)
        cv2.imwrite(os.path.join(debug_folder, "7_canny_edges.jpg"), edges)
        logger.info("Step 6: Canny edge detection applied")

        # ----------------------------
        # Step 7: Find contours
        # ----------------------------
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        logger.info(f"Step 7: Found {len(contours)} contours")

        # Visualize all contours
        all_contours_vis = image.copy()
        cv2.drawContours(all_contours_vis, contours[:20], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_folder, "8_all_contours.jpg"), all_contours_vis)

        # ----------------------------
        # Step 8: Look for quadrilateral with credit card aspect ratio
        # ----------------------------
        card_contour = None
        card_approx = None
        best_confidence = 0
        best_ratio = 0

        checked_contours = []

        for i, c in enumerate(contours[:50]):  # Check top 50 contours
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # Must be a quadrilateral (4 sides)
            if len(approx) != 4:
                checked_contours.append({
                    'index': i,
                    'sides': len(approx),
                    'reason': f'Not quadrilateral ({len(approx)} sides)'
                })
                continue

            # Compute bounding box size
            pts = approx.reshape(4, 2)
            ordered_pts = order_points(pts)

            tl, tr, br, bl = ordered_pts

            # Compute width and height in pixels
            width_px = np.linalg.norm(tr - tl)
            height_px = np.linalg.norm(bl - tl)

            if height_px == 0:
                checked_contours.append({
                    'index': i,
                    'sides': 4,
                    'reason': 'Zero height'
                })
                continue

            ratio = width_px / height_px

            # Check if ratio matches credit card (1.50 < ratio < 1.65)
            # Expected ratio ~1.586
            if 1.50 < ratio < 1.65:
                # Calculate confidence based on how close to ideal ratio
                confidence = 1.0 - abs(ratio - CREDIT_CARD_ASPECT_RATIO) / CREDIT_CARD_ASPECT_RATIO
                confidence = max(0.5, min(1.0, confidence))

                if confidence > best_confidence:
                    card_contour = c
                    card_approx = approx
                    best_confidence = confidence
                    best_ratio = ratio

                checked_contours.append({
                    'index': i,
                    'sides': 4,
                    'ratio': ratio,
                    'confidence': confidence,
                    'status': 'ACCEPTED'
                })
                logger.info(f"  Contour {i}: 4 sides, ratio={ratio:.3f}, confidence={confidence:.3f} ✓")
            else:
                checked_contours.append({
                    'index': i,
                    'sides': 4,
                    'ratio': ratio,
                    'reason': f'Aspect ratio {ratio:.3f} outside range [1.50, 1.65]'
                })

        # Visualize rejected contours
        rejected_vis = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        for check in checked_contours[:10]:
            if 'status' not in check:
                reason = check.get('reason', 'Unknown')
                cv2.putText(rejected_vis, f"#{check['index']}: {reason}",
                           (10, y_offset), font, 0.4, (0, 0, 255), 1)
                y_offset += 20
        cv2.imwrite(os.path.join(debug_folder, "9_rejected_contours.jpg"), rejected_vis)

        if card_contour is None:
            logger.warning("Step 8: No quadrilateral with credit card aspect ratio found")
            logger.warning("=" * 70)
            logger.warning("FAILED: No credit card detected")
            logger.warning("=" * 70)

            # Save failure visualization
            failure_vis = image.copy()
            cv2.rectangle(failure_vis, (10, 10), (500, 80), (0, 0, 0), -1)
            cv2.rectangle(failure_vis, (10, 10), (500, 80), (0, 0, 255), 3)
            cv2.putText(failure_vis, "NO CARD DETECTED", (20, 40), font, 0.8, (0, 0, 255), 2)
            cv2.putText(failure_vis, "No 4-sided polygon with card aspect ratio", (20, 70), font, 0.4, (200, 200, 200), 1)
            cv2.imwrite(os.path.join(debug_folder, "10_FINAL_RESULT.jpg"), failure_vis)

            return None, None, None, 0

        logger.info(f"Step 8: Credit card found! Aspect ratio={best_ratio:.3f}, Confidence={best_confidence:.3f}")

        # ----------------------------
        # Step 9: Use perspective transform for accurate measurements
        # ----------------------------
        pts = card_approx.reshape(4, 2).astype(np.float32)
        width_px, height_px, warped_card = perspective_transform_and_measure(image, pts)
        logger.info(f"Step 9: Perspective transform - Width={width_px:.1f}px, Height={height_px:.1f}px")

        # Save warped card for verification
        cv2.imwrite(os.path.join(debug_folder, "9a_warped_card.jpg"), warped_card)

        # Create minAreaRect for compatibility (but use accurate measurements)
        rect_center = cv2.minAreaRect(card_contour)[0]
        rect_angle = cv2.minAreaRect(card_contour)[2]
        rect = (rect_center, (width_px, height_px), rect_angle)

        # Save final result
        result_vis = image.copy()

        # Draw the approximated quadrilateral
        cv2.drawContours(result_vis, [card_approx], -1, (0, 255, 0), 3)

        # Draw corner points
        pts = card_approx.reshape(4, 2)
        for pt in pts:
            cv2.circle(result_vis, tuple(pt.astype(int)), 8, (0, 0, 255), -1)

        # Draw bounding rectangle
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(result_vis, [box], 0, (255, 0, 0), 2)

        # Add text overlay
        cv2.rectangle(result_vis, (10, 10), (600, 140), (0, 0, 0), -1)
        cv2.rectangle(result_vis, (10, 10), (600, 140), (0, 255, 0), 3)
        cv2.putText(result_vis, "QUADRILATERAL DETECTION SUCCESS", (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(result_vis, f"Aspect Ratio: {best_ratio:.3f} (target: 1.586)", (20, 70), font, 0.5, (255, 255, 255), 1)
        cv2.putText(result_vis, f"Confidence: {best_confidence:.3f}", (20, 95), font, 0.5, (255, 255, 255), 1)
        cv2.putText(result_vis, f"Size: {width_px:.0f}x{height_px:.0f}px (perspective corrected)", (20, 120), font, 0.45, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "10_FINAL_RESULT.jpg"), result_vis)

        logger.info("=" * 70)
        logger.info(f"SUCCESS: Quadrilateral detection (confidence: {best_confidence:.3f})")
        logger.info("=" * 70)

        return card_contour, rect, "quadrilateral_detection", best_confidence

    except Exception as e:
        logger.error(f"Detection failed with error: {str(e)}", exc_info=True)
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
        Calibration data dictionary
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

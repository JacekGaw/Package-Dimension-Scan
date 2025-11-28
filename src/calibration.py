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

def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve contrast across different lighting conditions
    """
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge back
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    logger.info("Applied CLAHE for contrast enhancement")
    return enhanced


def preprocess_for_card_detection(image):
    """
    Advanced preprocessing to isolate card from background
    Returns: preprocessed binary mask
    """
    logger.info("Starting advanced preprocessing...")

    # Apply CLAHE for better contrast across different backgrounds
    enhanced = apply_clahe(image)

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    # Create multiple masks for different potential card colors
    # This helps detect cards of various colors against wooden backgrounds

    # Mask 1: Blues and cyans (common card colors)
    lower_blue = np.array([80, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Mask 2: Reds and oranges
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                               cv2.inRange(hsv, lower_red2, upper_red2))

    # Mask 3: Greens
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Mask 4: Yellows
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Mask 5: High saturation (any color) - cards are usually more saturated than wood
    _, saturation, _ = cv2.split(hsv)
    _, mask_saturated = cv2.threshold(saturation, 40, 255, cv2.THRESH_BINARY)

    # Combine all color masks
    color_mask = cv2.bitwise_or(mask_blue, mask_red)
    color_mask = cv2.bitwise_or(color_mask, mask_green)
    color_mask = cv2.bitwise_or(color_mask, mask_yellow)
    color_mask = cv2.bitwise_or(color_mask, mask_saturated)

    logger.info("Created color segmentation mask")

    # Morphological operations to clean up the mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    # Remove small noise
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_small)
    # Fill small holes
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_large)

    logger.info("Applied morphological operations")

    return color_mask


def detect_credit_card(image):
    """
    Detect credit card in image using color segmentation and contour analysis
    Returns: (contour, rect) or (None, None)
    """
    logger.info("Starting credit card detection...")

    # Step 1: Color-based preprocessing
    color_mask = preprocess_for_card_detection(image)

    # Step 2: Edge detection on the color mask
    blurred_mask = cv2.GaussianBlur(color_mask, (5, 5), 0)
    edges = cv2.Canny(blurred_mask, 50, 150)
    logger.info("Performed Canny edge detection on color mask")

    # Additional morphological closing to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"Found {len(contours)} contours")

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find credit card by filtering criteria
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # Skip very small contours early
        if area < 5000:  # Increased minimum area threshold
            logger.debug(f"Contour {i}: Area too small ({area:.0f}), skipping remaining")
            break

        # Get minimum area rectangle
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]

        if width == 0 or height == 0:
            continue

        # Calculate aspect ratio (ensure width > height)
        aspect_ratio = max(width, height) / min(width, height)

        # Calculate contour properties for better filtering
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Calculate rectangularity (how close to a rectangle)
        rect_area = width * height
        extent = area / rect_area if rect_area > 0 else 0

        # Calculate solidity (convexity)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        logger.debug(f"Contour {i}: corners={len(approx)}, aspect_ratio={aspect_ratio:.2f}, "
                    f"area={area:.0f}, extent={extent:.2f}, solidity={solidity:.2f}")

        # Credit card criteria:
        # 1. Aspect ratio between 1.4 and 1.9 (credit card is ~1.586)
        # 2. Has 4 corners (or close to it)
        # 3. High rectangularity (extent > 0.8)
        # 4. High solidity (solidity > 0.9) - cards are convex
        # 5. Large enough area

        if (1.4 < aspect_ratio < 1.9 and
            4 <= len(approx) <= 6 and  # Allow slight variation in corner detection
            extent > 0.75 and
            solidity > 0.85 and
            area > 5000):

            logger.info(f"[SUCCESS] Credit card found! Contour {i}, aspect_ratio={aspect_ratio:.2f}, "
                       f"area={area:.0f}, extent={extent:.2f}, solidity={solidity:.2f}")
            return contour, rect

    logger.warning("No credit card found matching criteria")
    return None, None


def detect_card_hough_lines(image):
    """
    Detect credit card using Hough Line Transform
    This method is robust across different backgrounds and card colors
    Returns: (contour, rect) or (None, None)
    """
    logger.info("Starting Hough Line detection...")

    # Apply CLAHE for better edge detection
    enhanced = apply_clahe(image)

    # Convert to grayscale
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    # Edge detection with automatic threshold
    median_val = np.median(bilateral)
    lower = int(max(0, 0.7 * median_val))
    upper = int(min(255, 1.3 * median_val))
    edges = cv2.Canny(bilateral, lower, upper)

    # Dilate edges slightly to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    logger.info("Applying Hough Line Transform...")

    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=100, maxLineGap=20)

    if lines is None or len(lines) < 4:
        logger.warning("Not enough lines detected for rectangle")
        return None, None

    # Group lines by orientation (horizontal vs vertical)
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        # Classify as horizontal or vertical
        if angle < 20 or angle > 160:  # Horizontal (allow some tolerance)
            horizontal_lines.append(line[0])
        elif 70 < angle < 110:  # Vertical
            vertical_lines.append(line[0])

    logger.info(f"Found {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines")

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        logger.warning("Not enough parallel lines for rectangle")
        return None, None

    # Find the best rectangle from line intersections
    best_rect = None
    best_score = 0

    # Try different combinations of lines
    for h1 in horizontal_lines[:5]:  # Limit iterations
        for h2 in horizontal_lines[:5]:
            if np.array_equal(h1, h2):
                continue

            for v1 in vertical_lines[:5]:
                for v2 in vertical_lines[:5]:
                    if np.array_equal(v1, v2):
                        continue

                    # Calculate potential rectangle
                    rect_candidate = get_rectangle_from_lines(h1, h2, v1, v2, image.shape)

                    if rect_candidate is not None:
                        score = score_rectangle_candidate(rect_candidate, image.shape)
                        if score > best_score:
                            best_score = score
                            best_rect = rect_candidate

    if best_rect is not None and best_score > 0.3:
        # Convert rectangle to contour format
        box = cv2.boxPoints(best_rect)
        contour = box.astype(np.int32).reshape((-1, 1, 2))
        logger.info(f"[SUCCESS] Card detected via Hough lines! Score: {best_score:.2f}")
        return contour, best_rect

    logger.warning("No valid rectangle found from Hough lines")
    return None, None


def get_rectangle_from_lines(h1, h2, v1, v2, img_shape):
    """
    Create rectangle from 4 lines (2 horizontal, 2 vertical)
    Returns: minAreaRect or None
    """
    try:
        # Find intersections of lines to get 4 corners
        corners = []

        for h_line in [h1, h2]:
            for v_line in [v1, v2]:
                intersection = line_intersection(h_line, v_line)
                if intersection is not None:
                    x, y = intersection
                    # Check if intersection is within image bounds
                    if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                        corners.append([x, y])

        if len(corners) != 4:
            return None

        # Create contour from corners
        corners_array = np.array(corners, dtype=np.float32)
        rect = cv2.minAreaRect(corners_array)

        return rect

    except:
        return None


def line_intersection(line1, line2):
    """
    Find intersection point of two lines
    Each line is [x1, y1, x2, y2]
    Returns: (x, y) or None
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:  # Lines are parallel
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (x, y)


def score_rectangle_candidate(rect, img_shape):
    """
    Score a rectangle candidate based on credit card properties
    Returns: score between 0 and 1
    """
    try:
        width, height = rect[1]

        if width == 0 or height == 0:
            return 0

        # Calculate aspect ratio
        aspect_ratio = max(width, height) / min(width, height)

        # Score based on aspect ratio (ideal: 1.586)
        aspect_score = 0
        if 1.4 < aspect_ratio < 1.9:
            # Closer to 1.586 is better
            aspect_diff = abs(aspect_ratio - CREDIT_CARD_ASPECT_RATIO)
            aspect_score = max(0, 1.0 - aspect_diff / 0.5)

        # Score based on size (should be reasonable size, not too small/large)
        area = width * height
        img_area = img_shape[0] * img_shape[1]
        size_ratio = area / img_area

        size_score = 0
        if 0.05 < size_ratio < 0.7:  # Card should be 5-70% of image
            size_score = 1.0

        # Check if rectangle is within image bounds
        center = rect[0]
        bounds_score = 0
        if (0 < center[0] < img_shape[1] and
            0 < center[1] < img_shape[0]):
            bounds_score = 1.0

        # Combined score
        total_score = (aspect_score * 0.5 + size_score * 0.3 + bounds_score * 0.2)

        return total_score

    except:
        return 0

def calculate_texture_score(image, contour):
    """
    Calculate texture score for a contour region
    Cards should have low texture (smooth, uniform)
    Returns: score between 0 (textured) and 1 (smooth)
    """
    try:
        # Create mask for the contour
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate standard deviation in the region (texture measure)
        mean, stddev = cv2.meanStdDev(gray, mask=mask)

        # Calculate gradient magnitude (edge density)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)

        # Mean gradient in the region
        masked_gradient = cv2.bitwise_and(gradient_mag, gradient_mag, mask=mask)
        mean_gradient = np.mean(masked_gradient[mask > 0]) if np.any(mask > 0) else 0

        # Cards typically have:
        # - Low to medium standard deviation (unless patterned)
        # - Low internal gradient (edges only at boundaries)

        # Normalize scores (lower is better for cards)
        stddev_score = max(0, 1.0 - (stddev[0][0] / 60.0))  # Good if stddev < 60
        gradient_score = max(0, 1.0 - (mean_gradient / 30.0))  # Good if gradient < 30

        # Combined texture score
        texture_score = (stddev_score * 0.4 + gradient_score * 0.6)

        logger.debug(f"Texture analysis - stddev: {stddev[0][0]:.1f}, gradient: {mean_gradient:.1f}, score: {texture_score:.2f}")

        return texture_score

    except Exception as e:
        logger.debug(f"Texture calculation failed: {e}")
        return 0.5  # Neutral score on error


def detect_card_with_ensemble(image):
    """
    Ensemble detection method that tries multiple strategies
    and picks the best result
    Returns: (contour, rect, method_name, confidence, debug_info) or (None, None, None, 0, {})
    """
    logger.info("Starting ensemble detection with multiple strategies...")

    candidates = []
    debug_info = {
        'color_segmentation': None,
        'hough_lines': None
    }

    # Strategy 1: Color segmentation (original improved method)
    try:
        logger.info("--- Strategy 1: Color segmentation ---")
        contour1, rect1 = detect_credit_card(image)
        if contour1 is not None:
            # Calculate confidence score
            area = cv2.contourArea(contour1)
            width, height = rect1[1]
            aspect_ratio = max(width, height) / min(width, height)
            aspect_diff = abs(aspect_ratio - CREDIT_CARD_ASPECT_RATIO) / CREDIT_CARD_ASPECT_RATIO

            # Calculate geometric scores
            rect_area = width * height
            extent = area / rect_area if rect_area > 0 else 0
            hull = cv2.convexHull(contour1)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Calculate texture score
            texture_score = calculate_texture_score(image, contour1)

            # Combined confidence
            confidence = (
                (1.0 - aspect_diff) * 0.3 +
                extent * 0.2 +
                solidity * 0.2 +
                texture_score * 0.3
            )

            candidates.append({
                'contour': contour1,
                'rect': rect1,
                'method': 'color_segmentation',
                'confidence': confidence
            })
            debug_info['color_segmentation'] = {
                'contour': contour1,
                'rect': rect1,
                'confidence': confidence
            }
            logger.info(f"Color segmentation: confidence={confidence:.2f}")
        else:
            logger.info("Color segmentation: no card detected")
    except Exception as e:
        logger.warning(f"Color segmentation failed: {e}")

    # Strategy 2: Hough Line Transform
    try:
        logger.info("--- Strategy 2: Hough Line Transform ---")
        contour2, rect2 = detect_card_hough_lines(image)
        if contour2 is not None:
            # Calculate confidence score
            width, height = rect2[1]
            aspect_ratio = max(width, height) / min(width, height)
            aspect_diff = abs(aspect_ratio - CREDIT_CARD_ASPECT_RATIO) / CREDIT_CARD_ASPECT_RATIO

            # Hough lines are very reliable for rectangles
            geometric_score = score_rectangle_candidate(rect2, image.shape)

            # Calculate texture score
            texture_score = calculate_texture_score(image, contour2)

            # Hough method gets bonus for being geometry-based
            confidence = (
                (1.0 - aspect_diff) * 0.3 +
                geometric_score * 0.4 +
                texture_score * 0.3
            )

            candidates.append({
                'contour': contour2,
                'rect': rect2,
                'method': 'hough_lines',
                'confidence': confidence
            })
            debug_info['hough_lines'] = {
                'contour': contour2,
                'rect': rect2,
                'confidence': confidence
            }
            logger.info(f"Hough lines: confidence={confidence:.2f}")
        else:
            logger.info("Hough lines: no card detected")
    except Exception as e:
        logger.warning(f"Hough line detection failed: {e}")

    # Select best candidate
    if not candidates:
        logger.warning("All detection strategies failed")
        return None, None, None, 0, debug_info

    # Sort by confidence
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    best = candidates[0]

    logger.info(f"[SUCCESS] Best detection: {best['method']} with confidence {best['confidence']:.2f}")

    return best['contour'], best['rect'], best['method'], best['confidence'], debug_info


def calculate_calibration(contour, rect, image):
    """
    Calculate pixels per millimeter from detected credit card
    Returns: dict with calibration data
    """
    if contour is None:
        raise ValueError("REFERENCE_NOT_FOUND")

    width, height = rect[1]
    card_width_px = max(width, height)
    card_height_px = min(width, height)

    # Calculate pixels per millimeter from width (more accurate)
    pixels_per_mm = card_width_px / CREDIT_CARD_WIDTH_MM

    # Calculate confidence based on aspect ratio accuracy
    aspect_ratio = card_width_px / card_height_px
    aspect_diff = abs(aspect_ratio - CREDIT_CARD_ASPECT_RATIO) / CREDIT_CARD_ASPECT_RATIO

    # Confidence: 1.0 perfect, decreases with aspect ratio deviation
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
        }
    }

def save_debug_image(image, contour, rect, filename, calibration_result, debug_info=None):
    """
    Save debug image with detection annotations

    Args:
        image: Original image (numpy array)
        contour: Detected contour (or None if not found)
        rect: Minimum area rectangle (or None if not found)
        filename: Output filename
        calibration_result: Calibration result dict (or None if failed)
        debug_info: Dict with strategy-specific detection results
    """
    try:
        if debug_info is None:
            debug_info = {}
        # Create a copy to draw on
        debug_image = image.copy()

        # Get all contours for visualization using the new preprocessing
        color_mask = preprocess_for_card_detection(image)
        blurred_mask = cv2.GaussianBlur(color_mask, (5, 5), 0)
        edges = cv2.Canny(blurred_mask, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        all_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours in gray (for context)
        cv2.drawContours(debug_image, all_contours, -1, (128, 128, 128), 2)

        if contour is not None and rect is not None:
            # Validate rect has valid dimensions
            try:
                width, height = rect[1]
                if width > 0 and height > 0:
                    # Draw detected credit card contour in green
                    cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 3)

                    # Draw minimum area rectangle in blue
                    box = cv2.boxPoints(rect)
                    box = np.int8(box)
                    # Validate box has 4 valid points
                    if len(box) == 4 and not np.any(np.isnan(box)) and not np.any(np.isinf(box)):
                        cv2.drawContours(debug_image, [box], 0, (255, 0, 0), 3)

                    # Draw center point
                    center = rect[0]
                    cv2.circle(debug_image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
            except Exception as e:
                logger.warning(f"Error drawing detection on debug image: {e}")

            # Add calibration info text
            try:
                width, height = rect[1]
            except:
                width, height = 0, 0

            if calibration_result and width > 0 and height > 0:
                y_offset = 40
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2

                # Background for text
                text_bg_height = y_offset + 120 if 'detectionMethod' in calibration_result else y_offset + 90
                cv2.rectangle(debug_image, (10, 10), (500, text_bg_height), (0, 0, 0), -1)
                cv2.rectangle(debug_image, (10, 10), (500, text_bg_height), (0, 255, 0), 2)

                # Text lines
                cv2.putText(debug_image, "CREDIT CARD DETECTED", (20, y_offset),
                           font, font_scale, (0, 255, 0), thickness)

                cv2.putText(debug_image, f"PPM: {calibration_result['pixelsPerMillimeter']:.2f}",
                           (20, y_offset + 30), font, font_scale, (255, 255, 255), thickness)

                cv2.putText(debug_image, f"Confidence: {calibration_result['confidence']*100:.1f}%",
                           (20, y_offset + 60), font, font_scale, (255, 255, 255), thickness)

                # Show detection method if available
                if 'detectionMethod' in calibration_result:
                    method_text = f"Method: {calibration_result['detectionMethod']}"
                    cv2.putText(debug_image, method_text,
                               (20, y_offset + 90), font, 0.6, (255, 255, 0), 2)

                # Draw measurements on the card
                try:
                    card_width_px = max(width, height)
                    card_height_px = min(width, height)
                    center = rect[0]

                    cv2.putText(debug_image, f"{card_width_px:.0f}px",
                               (int(center[0] - 40), int(center[1] - 20)),
                               font, 0.5, (255, 255, 0), 2)
                    cv2.putText(debug_image, f"{card_height_px:.0f}px",
                               (int(center[0] + 20), int(center[1])),
                               font, 0.5, (255, 255, 0), 2)
                except Exception as e:
                    logger.warning(f"Error adding measurements to debug image: {e}")
            else:
                # Invalid dimensions
                logger.warning("Rect has invalid dimensions (width or height is 0)")
        else:
            # No card detected - show failure message
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(debug_image, (10, 10), (400, 60), (0, 0, 0), -1)
            cv2.rectangle(debug_image, (10, 10), (400, 60), (0, 0, 255), 2)
            cv2.putText(debug_image, "CREDIT CARD NOT DETECTED", (20, 40),
                       font, 0.7, (0, 0, 255), 2)

        # Save main debug image
        output_path = os.path.join(DEBUG_IMAGE_DIR, filename)
        cv2.imwrite(output_path, debug_image)
        logger.info(f"Debug image saved: {output_path}")

        # Save preprocessing steps for detailed debugging
        base_name = filename.rsplit('.', 1)[0]

        # Save color mask
        mask_path = os.path.join(DEBUG_IMAGE_DIR, f"{base_name}_mask.jpg")
        cv2.imwrite(mask_path, color_mask)

        # Save edges
        edges_path = os.path.join(DEBUG_IMAGE_DIR, f"{base_name}_edges.jpg")
        cv2.imwrite(edges_path, edges)

        # Create a side-by-side comparison for easy viewing
        h, w = image.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = image
        comparison[:, w:] = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)

        comparison_path = os.path.join(DEBUG_IMAGE_DIR, f"{base_name}_comparison.jpg")
        cv2.imwrite(comparison_path, comparison)

        logger.info(f"Additional debug images saved: {base_name}_mask.jpg, {base_name}_edges.jpg, {base_name}_comparison.jpg")

        # Save strategy-specific visualizations
        if debug_info:
            save_strategy_visualizations(image, debug_info, base_name)

        return output_path

    except Exception as e:
        logger.error(f"Failed to save debug image: {str(e)}", exc_info=True)
        return None


def save_strategy_visualizations(image, debug_info, base_name):
    """
    Save separate visualization images for each detection strategy

    Args:
        image: Original image
        debug_info: Dict with strategy results
        base_name: Base filename (without extension)
    """
    try:
        h, w = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Strategy 1: Color Segmentation Visualization
        if 'color_segmentation' in debug_info and debug_info['color_segmentation'] is not None:
            vis1 = image.copy()
            result = debug_info['color_segmentation']

            if result['contour'] is not None and result['rect'] is not None:
                try:
                    # Validate rect dimensions
                    width, height = result['rect'][1]
                    if width > 0 and height > 0:
                        # Draw the detected contour in green
                        cv2.drawContours(vis1, [result['contour']], -1, (0, 255, 0), 3)

                        # Draw rectangle
                        box = cv2.boxPoints(result['rect'])
                        box = np.int8(box)
                        if len(box) == 4 and not np.any(np.isnan(box)) and not np.any(np.isinf(box)):
                            cv2.drawContours(vis1, [box], 0, (0, 255, 255), 2)

                        # Add text
                        cv2.rectangle(vis1, (10, 10), (450, 80), (0, 0, 0), -1)
                        cv2.rectangle(vis1, (10, 10), (450, 80), (0, 255, 0), 2)
                        cv2.putText(vis1, "Strategy: COLOR SEGMENTATION", (20, 35),
                                   font, 0.6, (0, 255, 0), 2)
                        cv2.putText(vis1, f"Confidence: {result['confidence']:.2f}", (20, 60),
                                   font, 0.6, (255, 255, 255), 2)
                    else:
                        # Invalid dimensions
                        cv2.rectangle(vis1, (10, 10), (450, 50), (0, 0, 0), -1)
                        cv2.rectangle(vis1, (10, 10), (450, 50), (0, 0, 255), 2)
                        cv2.putText(vis1, "COLOR SEGMENTATION: INVALID RECT", (20, 35),
                                   font, 0.6, (0, 0, 255), 2)
                except Exception as e:
                    logger.warning(f"Error drawing color segmentation visualization: {e}")
            else:
                # No detection
                cv2.rectangle(vis1, (10, 10), (450, 50), (0, 0, 0), -1)
                cv2.rectangle(vis1, (10, 10), (450, 50), (0, 0, 255), 2)
                cv2.putText(vis1, "COLOR SEGMENTATION: NO DETECTION", (20, 35),
                           font, 0.6, (0, 0, 255), 2)

            path1 = os.path.join(DEBUG_IMAGE_DIR, f"{base_name}_strategy_color.jpg")
            cv2.imwrite(path1, vis1)
            logger.info(f"Saved color segmentation strategy image: {base_name}_strategy_color.jpg")

        # Strategy 2: Hough Lines Visualization
        if 'hough_lines' in debug_info and debug_info['hough_lines'] is not None:
            vis2 = image.copy()
            result = debug_info['hough_lines']

            if result['contour'] is not None and result['rect'] is not None:
                try:
                    # Validate rect dimensions
                    width, height = result['rect'][1]
                    if width > 0 and height > 0:
                        # Draw the detected contour in blue
                        cv2.drawContours(vis2, [result['contour']], -1, (255, 0, 0), 3)

                        # Draw rectangle
                        box = cv2.boxPoints(result['rect'])
                        box = np.int8(box)
                        if len(box) == 4 and not np.any(np.isnan(box)) and not np.any(np.isinf(box)):
                            cv2.drawContours(vis2, [box], 0, (255, 255, 0), 2)

                        # Add text
                        cv2.rectangle(vis2, (10, 10), (450, 80), (0, 0, 0), -1)
                        cv2.rectangle(vis2, (10, 10), (450, 80), (255, 0, 0), 2)
                        cv2.putText(vis2, "Strategy: HOUGH LINE TRANSFORM", (20, 35),
                                   font, 0.6, (255, 100, 0), 2)
                        cv2.putText(vis2, f"Confidence: {result['confidence']:.2f}", (20, 60),
                                   font, 0.6, (255, 255, 255), 2)
                    else:
                        # Invalid dimensions
                        cv2.rectangle(vis2, (10, 10), (450, 50), (0, 0, 0), -1)
                        cv2.rectangle(vis2, (10, 10), (450, 50), (0, 0, 255), 2)
                        cv2.putText(vis2, "HOUGH LINES: INVALID RECT", (20, 35),
                                   font, 0.6, (0, 0, 255), 2)
                except Exception as e:
                    logger.warning(f"Error drawing Hough lines visualization: {e}")
            else:
                # No detection
                cv2.rectangle(vis2, (10, 10), (450, 50), (0, 0, 0), -1)
                cv2.rectangle(vis2, (10, 10), (450, 50), (0, 0, 255), 2)
                cv2.putText(vis2, "HOUGH LINES: NO DETECTION", (20, 35),
                           font, 0.6, (0, 0, 255), 2)

            path2 = os.path.join(DEBUG_IMAGE_DIR, f"{base_name}_strategy_hough.jpg")
            cv2.imwrite(path2, vis2)
            logger.info(f"Saved Hough lines strategy image: {base_name}_strategy_hough.jpg")

        # Create side-by-side comparison of both strategies
        color_result = debug_info.get('color_segmentation')
        hough_result = debug_info.get('hough_lines')

        if color_result is not None or hough_result is not None:
            # Create comparison canvas
            comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)

            # Left side: Color segmentation
            left_img = image.copy()
            if color_result and color_result['contour'] is not None and color_result['rect'] is not None:
                try:
                    width, height = color_result['rect'][1]
                    if width > 0 and height > 0:
                        cv2.drawContours(left_img, [color_result['contour']], -1, (0, 255, 0), 3)
                        box = cv2.boxPoints(color_result['rect'])
                        box = np.int8(box)
                        if len(box) == 4 and not np.any(np.isnan(box)) and not np.any(np.isinf(box)):
                            cv2.drawContours(left_img, [box], 0, (0, 255, 255), 2)
                        cv2.putText(left_img, f"Color: {color_result['confidence']:.2f}", (20, 40),
                                   font, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(left_img, "Color: INVALID RECT", (20, 40),
                                   font, 0.8, (0, 0, 255), 2)
                except Exception as e:
                    cv2.putText(left_img, "Color: ERROR", (20, 40),
                               font, 0.8, (0, 0, 255), 2)
                    logger.warning(f"Error drawing color result in comparison: {e}")
            else:
                cv2.putText(left_img, "Color: NO DETECTION", (20, 40),
                           font, 0.8, (0, 0, 255), 2)

            # Right side: Hough lines
            right_img = image.copy()
            if hough_result and hough_result['contour'] is not None and hough_result['rect'] is not None:
                try:
                    width, height = hough_result['rect'][1]
                    if width > 0 and height > 0:
                        cv2.drawContours(right_img, [hough_result['contour']], -1, (255, 0, 0), 3)
                        box = cv2.boxPoints(hough_result['rect'])
                        box = np.int8(box)
                        if len(box) == 4 and not np.any(np.isnan(box)) and not np.any(np.isinf(box)):
                            cv2.drawContours(right_img, [box], 0, (255, 255, 0), 2)
                        cv2.putText(right_img, f"Hough: {hough_result['confidence']:.2f}", (20, 40),
                                   font, 0.8, (255, 100, 0), 2)
                    else:
                        cv2.putText(right_img, "Hough: INVALID RECT", (20, 40),
                                   font, 0.8, (0, 0, 255), 2)
                except Exception as e:
                    cv2.putText(right_img, "Hough: ERROR", (20, 40),
                               font, 0.8, (0, 0, 255), 2)
                    logger.warning(f"Error drawing hough result in comparison: {e}")
            else:
                cv2.putText(right_img, "Hough: NO DETECTION", (20, 40),
                           font, 0.8, (0, 0, 255), 2)

            # Combine
            comparison[:, :w] = left_img
            comparison[:, w:] = right_img

            # Add separator line
            cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)

            path_comp = os.path.join(DEBUG_IMAGE_DIR, f"{base_name}_strategies_comparison.jpg")
            cv2.imwrite(path_comp, comparison)
            logger.info(f"Saved strategies comparison image: {base_name}_strategies_comparison.jpg")

    except Exception as e:
        logger.error(f"Failed to save strategy visualizations: {str(e)}", exc_info=True)

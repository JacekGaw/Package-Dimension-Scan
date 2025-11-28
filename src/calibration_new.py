"""
New Credit Card Detection Implementation
Based on computer vision best practices for robust rectangular object detection

This module implements a multi-strategy approach for detecting credit cards in images
with improved preprocessing, edge detection, and validation techniques.
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


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_with_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for adaptive contrast enhancement across varying lighting conditions.

    Args:
        image: BGR image

    Returns:
        Enhanced grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    logger.debug("Applied CLAHE preprocessing")
    return enhanced


def preprocess_multi_colorspace(image):
    """
    Process image in multiple color spaces to handle various card colors.
    Returns the channel with best edge potential.

    Args:
        image: BGR image

    Returns:
        List of preprocessed channels [gray, hsv_v, lab_l]
    """
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HSV - Value channel (good for colored cards)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    # LAB - Lightness channel (good for lighting variations)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Apply CLAHE to all channels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    channels = [
        clahe.apply(gray),
        clahe.apply(v_channel),
        clahe.apply(l_channel)
    ]

    logger.debug("Processed image in multiple color spaces")
    return channels


def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter to reduce noise while preserving edges.

    Args:
        image: Grayscale image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def normalize_illumination(image):
    """
    NEW IMPROVEMENT #1: Background Subtraction / Illumination Normalization

    Removes background texture and shadows by estimating the background
    and subtracting it from the image. This is especially helpful for
    wooden/textured surfaces and uneven lighting.

    Args:
        image: BGR image or grayscale image

    Returns:
        Normalized grayscale image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create large-scale background model using morphological closing
    # Large kernel captures the overall lighting/background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Subtract background to remove texture and lighting variations
    # This normalizes the illumination across the image
    normalized = cv2.subtract(background, gray)

    # Invert so card becomes bright on dark background
    normalized = 255 - normalized

    logger.debug("Applied illumination normalization (background subtraction)")
    return normalized


# =============================================================================
# EDGE DETECTION FUNCTIONS
# =============================================================================

def calculate_auto_canny_thresholds(image):
    """
    NEW IMPROVEMENT #2: Automatic Canny Thresholds

    Automatically calculate optimal Canny thresholds based on image statistics.
    This adapts to different brightness levels and contrast conditions.

    Args:
        image: Grayscale image

    Returns:
        (lower_threshold, upper_threshold)
    """
    # Calculate median pixel value
    median = np.median(image)

    # Calculate standard deviation to determine image contrast
    stddev = np.std(image)

    # Adjust sensitivity based on contrast
    # High contrast images (high stddev) can use stricter thresholds
    # Low contrast images (low stddev) need more sensitive thresholds
    if stddev < 30:
        # Low contrast - more sensitive
        sigma = 0.5
    elif stddev < 60:
        # Medium contrast - balanced
        sigma = 0.66
    else:
        # High contrast - stricter
        sigma = 0.8

    # Calculate thresholds based on median and sensitivity
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    # Ensure upper is at least 2x lower (Canny recommendation)
    if upper < lower * 2:
        upper = lower * 2

    # Clamp to reasonable ranges
    lower = max(20, min(lower, 100))
    upper = max(lower * 2, min(upper, 250))

    logger.debug(f"Auto Canny thresholds: median={median:.1f}, stddev={stddev:.1f}, "
                f"thresholds=({lower}, {upper})")

    return lower, upper


def detect_edges_canny(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection with specified thresholds.

    Args:
        image: Preprocessed grayscale image
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis

    Returns:
        Binary edge image
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Canny edge detection with L2 gradient for better accuracy
    edges = cv2.Canny(blurred, low_threshold, high_threshold,
                      apertureSize=3, L2gradient=True)

    logger.debug(f"Canny edge detection: thresholds=({low_threshold}, {high_threshold})")
    return edges


def detect_edges_canny_auto(image):
    """
    Apply Canny edge detection with automatically calculated thresholds.

    Args:
        image: Preprocessed grayscale image

    Returns:
        Binary edge image
    """
    # Calculate automatic thresholds
    lower, upper = calculate_auto_canny_thresholds(image)

    # Apply Canny with auto thresholds
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, lower, upper, apertureSize=3, L2gradient=True)

    logger.debug(f"Auto Canny edge detection: thresholds=({lower}, {upper})")
    return edges


def detect_edges_multi_threshold(image):
    """
    Try multiple Canny thresholds and combine results for robustness.

    Args:
        image: Preprocessed grayscale image

    Returns:
        Combined edge image
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Strategy 1: Standard thresholds
    edges1 = cv2.Canny(blurred, 50, 150, L2gradient=True)

    # Strategy 2: Sensitive (captures weaker edges)
    edges2 = cv2.Canny(blurred, 30, 100, L2gradient=True)

    # Strategy 3: Strict (only strong edges)
    edges3 = cv2.Canny(blurred, 70, 200, L2gradient=True)

    # Combine using weighted OR (prioritize consistent edges)
    combined = cv2.bitwise_or(edges1, edges2)
    combined = cv2.bitwise_or(combined, edges3)

    logger.debug("Applied multi-threshold edge detection")
    return combined


def detect_edges_adaptive(image):
    """
    Use adaptive thresholding before Canny for extreme lighting variations.

    Args:
        image: Preprocessed grayscale image

    Returns:
        Edge image
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # Canny on thresholded image
    edges = cv2.Canny(adaptive, 50, 150)

    logger.debug("Applied adaptive threshold + Canny edge detection")
    return edges


# =============================================================================
# MORPHOLOGICAL OPERATIONS
# =============================================================================

def apply_morphological_operations(edges):
    """
    Apply morphological operations to clean up edges:
    1. Dilation to strengthen edges
    2. Closing to connect gaps
    3. Optional erosion to thin edges

    Args:
        edges: Binary edge image

    Returns:
        Cleaned edge image
    """
    # Rectangular kernel for straight edges (credit cards)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Step 1: Dilate to strengthen and connect edges
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Step 2: Closing to fill small gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Step 3: Optional erosion to prevent over-thickening
    # eroded = cv2.erode(closed, kernel, iterations=1)

    logger.debug("Applied morphological operations")
    return closed


# =============================================================================
# CORNER DETECTION FUNCTIONS
# =============================================================================

def detect_corners_shi_tomasi(image):
    """
    Detect corners using Shi-Tomasi corner detection.
    Better than Harris for rectangular objects.

    Args:
        image: Grayscale image

    Returns:
        Array of corner points [(x, y), ...]
    """
    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=10,
        useHarrisDetector=False,
        blockSize=3
    )

    if corners is None:
        return None

    # Reshape to list of points
    corners = corners.reshape(-1, 2)
    logger.debug(f"Detected {len(corners)} corners using Shi-Tomasi")
    return corners


def find_rectangular_corners(corners, target_aspect_ratio, image_shape):
    """
    Find 4 corners that form a rectangle with target aspect ratio.

    Args:
        corners: Array of corner points
        target_aspect_ratio: Expected aspect ratio (e.g., 1.586 for credit card)
        image_shape: Image shape for area validation

    Returns:
        (best_quad, best_rect, confidence) or (None, None, 0)
    """
    if corners is None or len(corners) < 4:
        return None, None, 0

    from itertools import combinations

    best_score = 0
    best_quad = None
    best_rect = None

    # Limit combinations for performance (top 50 corners by some metric)
    if len(corners) > 50:
        # Use corners with highest corner response
        corners = corners[:50]

    # Try all combinations of 4 corners
    max_combinations = 1000  # Limit to prevent timeout
    combination_count = 0

    for quad in combinations(corners, 4):
        combination_count += 1
        if combination_count > max_combinations:
            break

        quad = np.array(quad, dtype=np.float32)

        # Get minimum area rectangle
        rect = cv2.minAreaRect(quad)
        width, height = rect[1]

        if width == 0 or height == 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)

        # Check aspect ratio match
        aspect_diff = abs(aspect_ratio - target_aspect_ratio)
        if aspect_diff > 0.5:  # Too far from target
            continue

        # Score based on aspect ratio match
        aspect_score = max(0, 1.0 - aspect_diff / 0.5)

        # Check if corners form a convex quadrilateral
        area = cv2.contourArea(quad)
        if area < 5000:  # Too small
            continue

        # Validate area is reasonable (5-70% of image)
        image_area = image_shape[0] * image_shape[1]
        area_ratio = area / image_area
        if not (0.05 <= area_ratio <= 0.70):
            continue

        area_score = 1.0 if 0.15 <= area_ratio <= 0.50 else 0.7

        # Check rectangularity (how close to a perfect rectangle)
        rect_area = width * height
        extent = area / rect_area if rect_area > 0 else 0

        if extent < 0.75:  # Not rectangular enough
            continue

        rectangularity_score = extent

        # Calculate corner angle quality (should be ~90 degrees)
        angles = calculate_corner_angles(quad)
        angle_score = calculate_angle_score(angles)

        # Combined score
        total_score = (
            aspect_score * 0.3 +
            area_score * 0.2 +
            rectangularity_score * 0.3 +
            angle_score * 0.2
        )

        if total_score > best_score:
            best_score = total_score
            best_quad = quad
            best_rect = rect

    if best_quad is not None:
        logger.debug(f"Found rectangular corners with score {best_score:.3f}")

    return best_quad, best_rect, best_score


def calculate_corner_angles(quad):
    """
    Calculate angles at each corner of quadrilateral.

    Args:
        quad: 4 corner points

    Returns:
        List of 4 angles in degrees
    """
    angles = []

    for i in range(4):
        # Get three consecutive points
        p1 = quad[i]
        p2 = quad[(i + 1) % 4]
        p3 = quad[(i + 2) % 4]

        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle)

        angles.append(angle_deg)

    return angles


def calculate_angle_score(angles):
    """
    Score angles based on how close they are to 90 degrees.

    Args:
        angles: List of 4 angles in degrees

    Returns:
        Score between 0 and 1
    """
    if len(angles) != 4:
        return 0

    # Calculate deviation from 90 degrees
    deviations = [abs(angle - 90) for angle in angles]
    avg_deviation = np.mean(deviations)

    # Score: 0° deviation = 1.0, 45° deviation = 0
    score = max(0, 1.0 - avg_deviation / 45.0)

    return score


def detect_card_by_corners(image):
    """
    Detect credit card using corner detection approach.

    Args:
        image: BGR image

    Returns:
        (contour, rect, confidence) or (None, None, 0)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for better corner detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Detect corners
    corners = detect_corners_shi_tomasi(enhanced)

    if corners is None:
        logger.info("Corner detection: No corners found")
        return None, None, 0

    # Find rectangular corners
    quad, rect, confidence = find_rectangular_corners(
        corners,
        CREDIT_CARD_ASPECT_RATIO,
        image.shape
    )

    if quad is None:
        logger.info("Corner detection: No rectangular pattern found")
        return None, None, 0

    # Convert quad to contour format
    contour = quad.astype(np.int32).reshape((-1, 1, 2))

    logger.info(f"Corner detection: Found card with confidence {confidence:.3f}")
    return contour, rect, confidence


# =============================================================================
# CONTOUR DETECTION AND VALIDATION
# =============================================================================

def find_rectangular_contours(edges):
    """
    Find all contours and filter for rectangular shapes.

    Args:
        edges: Binary edge image

    Returns:
        List of rectangular contours
    """
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger.debug(f"Found {len(contours)} total contours")

    # Filter for rectangles
    rectangular_contours = []

    for contour in contours:
        # Quick area filter
        area = cv2.contourArea(contour)
        if area < 5000:  # Minimum area threshold
            continue

        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if approximately 4 corners (allow 4-6 for slight variations)
        if 4 <= len(approx) <= 6:
            rectangular_contours.append(contour)

    logger.debug(f"Found {len(rectangular_contours)} rectangular contours")
    return rectangular_contours


def validate_aspect_ratio(rect):
    """
    Validate if rectangle has credit card aspect ratio.

    Args:
        rect: minAreaRect tuple ((x, y), (width, height), angle)

    Returns:
        (is_valid, aspect_ratio, score)
    """
    width, height = rect[1]

    if width == 0 or height == 0:
        return False, 0, 0

    # Calculate aspect ratio (always > 1)
    aspect_ratio = max(width, height) / min(width, height)

    # Credit card range: 1.4 to 1.9 (or inverted: 0.53 to 0.71)
    is_valid = (1.4 <= aspect_ratio <= 1.9)

    # Calculate score based on how close to ideal (1.586)
    if is_valid:
        aspect_diff = abs(aspect_ratio - CREDIT_CARD_ASPECT_RATIO)
        score = max(0, 1.0 - aspect_diff / 0.5)  # Normalize to 0-1
    else:
        score = 0

    return is_valid, aspect_ratio, score


def validate_area(contour, image_shape):
    """
    Validate if contour area is reasonable for a credit card.

    Args:
        contour: Contour array
        image_shape: Image shape (height, width)

    Returns:
        (is_valid, area_ratio, score)
    """
    area = cv2.contourArea(contour)
    image_area = image_shape[0] * image_shape[1]
    area_ratio = area / image_area

    # Card should be 5% to 70% of image
    is_valid = 0.05 <= area_ratio <= 0.70

    # Score based on optimal range (15-50% is ideal)
    if 0.15 <= area_ratio <= 0.50:
        score = 1.0
    elif is_valid:
        if area_ratio < 0.15:
            score = area_ratio / 0.15
        else:
            score = (0.70 - area_ratio) / 0.20
    else:
        score = 0

    return is_valid, area_ratio, score


def validate_rectangularity(contour):
    """
    Validate how rectangular the contour is.

    Args:
        contour: Contour array

    Returns:
        (is_valid, extent, score)
    """
    # Get minimum area rectangle
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    rect_area = width * height

    if rect_area == 0:
        return False, 0, 0

    # Calculate extent (contour area / bounding rect area)
    contour_area = cv2.contourArea(contour)
    extent = contour_area / rect_area

    # Good rectangles have extent > 0.85
    is_valid = extent > 0.85
    score = extent if is_valid else extent * 0.5

    return is_valid, extent, score


def validate_solidity(contour):
    """
    Validate convexity of contour (credit cards are convex).

    Args:
        contour: Contour array

    Returns:
        (is_valid, solidity, score)
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return False, 0, 0

    solidity = area / hull_area

    # Cards should be very convex (solidity > 0.90)
    is_valid = solidity > 0.90
    score = solidity if is_valid else solidity * 0.8

    return is_valid, solidity, score


def validate_edge_strength(contour, edges_image):
    """
    Validate that contour perimeter has strong edges.

    Args:
        contour: Contour array
        edges_image: Binary edge image

    Returns:
        (is_valid, edge_ratio, score)
    """
    # Create mask for contour perimeter
    mask = np.zeros(edges_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, 2)

    # Count edge pixels along contour
    edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges_image, mask))
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return False, 0, 0

    edge_ratio = edge_pixels / perimeter

    # At least 30% of perimeter should have edges
    is_valid = edge_ratio > 0.30
    score = min(edge_ratio / 0.60, 1.0)  # Normalize (60% = perfect)

    return is_valid, edge_ratio, score


def calculate_confidence_score(contour, rect, image_shape, edges_image):
    """
    Calculate comprehensive confidence score for a credit card candidate.

    Args:
        contour: Contour array
        rect: minAreaRect
        image_shape: Image shape
        edges_image: Binary edge image

    Returns:
        (total_score, score_breakdown)
    """
    # Run all validations
    aspect_valid, aspect_ratio, aspect_score = validate_aspect_ratio(rect)
    area_valid, area_ratio, area_score = validate_area(contour, image_shape)
    rect_valid, extent, rect_score = validate_rectangularity(contour)
    solid_valid, solidity, solid_score = validate_solidity(contour)
    edge_valid, edge_ratio, edge_score = validate_edge_strength(contour, edges_image)

    # Check if all critical validations pass
    all_valid = aspect_valid and area_valid and rect_valid and solid_valid

    # Weighted scoring
    weights = {
        'aspect': 0.25,
        'area': 0.15,
        'rectangularity': 0.20,
        'solidity': 0.20,
        'edge_strength': 0.20
    }

    total_score = (
        aspect_score * weights['aspect'] +
        area_score * weights['area'] +
        rect_score * weights['rectangularity'] +
        solid_score * weights['solidity'] +
        edge_score * weights['edge_strength']
    )

    # Penalty if any critical validation fails
    if not all_valid:
        total_score *= 0.7

    score_breakdown = {
        'aspect_ratio': aspect_ratio,
        'aspect_score': aspect_score,
        'area_ratio': area_ratio,
        'area_score': area_score,
        'extent': extent,
        'rectangularity_score': rect_score,
        'solidity': solidity,
        'solidity_score': solid_score,
        'edge_ratio': edge_ratio,
        'edge_score': edge_score,
        'all_valid': all_valid,
        'total_score': total_score
    }

    return total_score, score_breakdown


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

def detect_credit_card_single_strategy(image, strategy_name, preprocessor, edge_detector):
    """
    Run a single detection strategy.

    Args:
        image: BGR image
        strategy_name: Name of strategy for logging
        preprocessor: Preprocessing function
        edge_detector: Edge detection function

    Returns:
        (contour, rect, confidence, score_breakdown) or (None, None, 0, {})
    """
    logger.info(f"--- Strategy: {strategy_name} ---")

    try:
        # Preprocessing
        preprocessed = preprocessor(image)

        # Handle multiple channels from multi-colorspace
        if isinstance(preprocessed, list):
            # Try each channel and pick best result
            best_result = (None, None, 0, {})

            for i, channel in enumerate(preprocessed):
                edges = edge_detector(channel)
                edges = apply_morphological_operations(edges)

                contours = find_rectangular_contours(edges)

                # Evaluate all contours
                for contour in contours:
                    rect = cv2.minAreaRect(contour)
                    confidence, breakdown = calculate_confidence_score(
                        contour, rect, image.shape, edges
                    )

                    if confidence > best_result[2]:
                        best_result = (contour, rect, confidence, breakdown)
                        logger.debug(f"Channel {i}: Found candidate with confidence {confidence:.3f}")

            return best_result
        else:
            # Single channel processing
            edges = edge_detector(preprocessed)
            edges = apply_morphological_operations(edges)

            contours = find_rectangular_contours(edges)

            # Find best contour
            best_contour = None
            best_rect = None
            best_confidence = 0
            best_breakdown = {}

            for contour in contours:
                rect = cv2.minAreaRect(contour)
                confidence, breakdown = calculate_confidence_score(
                    contour, rect, image.shape, edges
                )

                if confidence > best_confidence:
                    best_contour = contour
                    best_rect = rect
                    best_confidence = confidence
                    best_breakdown = breakdown

            if best_contour is not None:
                logger.info(f"{strategy_name}: Found candidate with confidence {best_confidence:.3f}")
                return best_contour, best_rect, best_confidence, best_breakdown
            else:
                logger.info(f"{strategy_name}: No valid candidate found")
                return None, None, 0, {}

    except Exception as e:
        logger.error(f"{strategy_name} failed: {str(e)}")
        return None, None, 0, {}


def detect_credit_card_robust(image):
    """
    Main detection function using ensemble of strategies.

    Args:
        image: BGR image (numpy array)

    Returns:
        (contour, rect, method_name, confidence, debug_info) or (None, None, None, 0, {})
    """
    logger.info("=" * 60)
    logger.info("Starting robust credit card detection with 7 strategies")
    logger.info("=" * 60)

    candidates = []
    debug_info = {}

    # Strategy 1: CLAHE + Standard Canny
    contour1, rect1, conf1, breakdown1 = detect_credit_card_single_strategy(
        image,
        "CLAHE + Standard Canny",
        preprocess_with_clahe,
        lambda img: detect_edges_canny(img, 50, 150)
    )

    if contour1 is not None:
        candidates.append({
            'contour': contour1,
            'rect': rect1,
            'method': 'clahe_standard',
            'confidence': conf1,
            'breakdown': breakdown1
        })
        debug_info['clahe_standard'] = {
            'contour': contour1,
            'rect': rect1,
            'confidence': conf1,
            'breakdown': breakdown1
        }

    # Strategy 2: CLAHE + Multi-threshold Canny
    contour2, rect2, conf2, breakdown2 = detect_credit_card_single_strategy(
        image,
        "CLAHE + Multi-threshold Canny",
        preprocess_with_clahe,
        detect_edges_multi_threshold
    )

    if contour2 is not None:
        candidates.append({
            'contour': contour2,
            'rect': rect2,
            'method': 'clahe_multi_threshold',
            'confidence': conf2,
            'breakdown': breakdown2
        })
        debug_info['clahe_multi_threshold'] = {
            'contour': contour2,
            'rect': rect2,
            'confidence': conf2,
            'breakdown': breakdown2
        }

    # Strategy 3: CLAHE + Adaptive Threshold + Canny
    contour3, rect3, conf3, breakdown3 = detect_credit_card_single_strategy(
        image,
        "CLAHE + Adaptive + Canny",
        preprocess_with_clahe,
        detect_edges_adaptive
    )

    if contour3 is not None:
        candidates.append({
            'contour': contour3,
            'rect': rect3,
            'method': 'clahe_adaptive',
            'confidence': conf3,
            'breakdown': breakdown3
        })
        debug_info['clahe_adaptive'] = {
            'contour': contour3,
            'rect': rect3,
            'confidence': conf3,
            'breakdown': breakdown3
        }

    # Strategy 4: Multi-colorspace + Standard Canny
    contour4, rect4, conf4, breakdown4 = detect_credit_card_single_strategy(
        image,
        "Multi-colorspace + Standard Canny",
        preprocess_multi_colorspace,
        lambda img: detect_edges_canny(img, 50, 150)
    )

    if contour4 is not None:
        candidates.append({
            'contour': contour4,
            'rect': rect4,
            'method': 'multi_colorspace',
            'confidence': conf4,
            'breakdown': breakdown4
        })
        debug_info['multi_colorspace'] = {
            'contour': contour4,
            'rect': rect4,
            'confidence': conf4,
            'breakdown': breakdown4
        }

    # Strategy 5: Background Normalization + Auto Canny (NEW!)
    contour5, rect5, conf5, breakdown5 = detect_credit_card_single_strategy(
        image,
        "Background Normalization + Auto Canny",
        normalize_illumination,
        detect_edges_canny_auto
    )

    if contour5 is not None:
        candidates.append({
            'contour': contour5,
            'rect': rect5,
            'method': 'normalized_auto',
            'confidence': conf5,
            'breakdown': breakdown5
        })
        debug_info['normalized_auto'] = {
            'contour': contour5,
            'rect': rect5,
            'confidence': conf5,
            'breakdown': breakdown5
        }

    # Strategy 6: Background Normalization + Multi-threshold (NEW!)
    contour6, rect6, conf6, breakdown6 = detect_credit_card_single_strategy(
        image,
        "Background Normalization + Multi-threshold",
        normalize_illumination,
        detect_edges_multi_threshold
    )

    if contour6 is not None:
        candidates.append({
            'contour': contour6,
            'rect': rect6,
            'method': 'normalized_multi',
            'confidence': conf6,
            'breakdown': breakdown6
        })
        debug_info['normalized_multi'] = {
            'contour': contour6,
            'rect': rect6,
            'confidence': conf6,
            'breakdown': breakdown6
        }

    # Strategy 7: Corner Detection (NEW!)
    logger.info("--- Strategy 7: Corner Detection ---")
    try:
        contour7, rect7, corner_confidence = detect_card_by_corners(image)

        if contour7 is not None:
            # Calculate full confidence score using our validation system
            # We need edges image for edge strength validation, create a simple one
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges_temp = cv2.Canny(gray, 50, 150)

            conf7, breakdown7 = calculate_confidence_score(
                contour7, rect7, image.shape, edges_temp
            )

            # Boost confidence since corner detection found it
            conf7 = min(1.0, conf7 * 1.1)

            candidates.append({
                'contour': contour7,
                'rect': rect7,
                'method': 'corner_detection',
                'confidence': conf7,
                'breakdown': breakdown7
            })
            debug_info['corner_detection'] = {
                'contour': contour7,
                'rect': rect7,
                'confidence': conf7,
                'breakdown': breakdown7
            }
            logger.info(f"Corner detection: confidence={conf7:.3f}")
        else:
            logger.info("Corner detection: no card found")
    except Exception as e:
        logger.warning(f"Corner detection failed: {str(e)}")

    # Select best candidate
    if not candidates:
        logger.warning("All detection strategies failed - no valid credit card found")
        return None, None, None, 0, debug_info

    # Sort by confidence
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    best = candidates[0]

    logger.info("=" * 60)
    logger.info(f"BEST DETECTION: {best['method']}")
    logger.info(f"Confidence: {best['confidence']:.3f}")
    logger.info(f"Breakdown: {best['breakdown']}")
    logger.info("=" * 60)

    return best['contour'], best['rect'], best['method'], best['confidence'], debug_info


# =============================================================================
# CALIBRATION CALCULATION
# =============================================================================

def apply_perspective_correction(image, contour):
    """
    NEW IMPROVEMENT #3: Perspective Correction

    Apply perspective transform to warp a tilted/angled card to a frontal view.
    This improves measurement accuracy for cards that are not perfectly flat
    or are viewed from an angle.

    Args:
        image: Original BGR or grayscale image
        contour: Detected card contour

    Returns:
        (warped_image, corrected_rect, was_corrected)
        - warped_image: Perspective-corrected image region
        - corrected_rect: New minAreaRect after correction
        - was_corrected: Boolean indicating if correction was applied
    """
    try:
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Check if card is significantly tilted (angle > 5 degrees from horizontal/vertical)
        angle = rect[2]
        width, height = rect[1]

        # Normalize angle to 0-90 range
        if angle < -45:
            angle = 90 + angle
        if angle > 45:
            angle = angle - 90

        # If angle is small, no need for perspective correction
        if abs(angle) < 5:
            logger.debug("Card is nearly flat - skipping perspective correction")
            return None, rect, False

        # Find the 4 corners of the card
        # We need to order them: top-left, top-right, bottom-right, bottom-left
        corners = box.astype(np.float32)

        # Order corners by their position
        # Sum: top-left has smallest sum, bottom-right has largest sum
        # Diff: top-right has smallest diff (x-y), bottom-left has largest diff
        rect_ordered = np.zeros((4, 2), dtype=np.float32)

        s = corners.sum(axis=1)
        rect_ordered[0] = corners[np.argmin(s)]  # Top-left
        rect_ordered[2] = corners[np.argmax(s)]  # Bottom-right

        diff = np.diff(corners, axis=1)
        rect_ordered[1] = corners[np.argmin(diff)]  # Top-right
        rect_ordered[3] = corners[np.argmax(diff)]  # Bottom-left

        # Calculate dimensions of the corrected card
        # Use the maximum dimensions to avoid information loss
        widthA = np.linalg.norm(rect_ordered[0] - rect_ordered[1])
        widthB = np.linalg.norm(rect_ordered[2] - rect_ordered[3])
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(rect_ordered[0] - rect_ordered[3])
        heightB = np.linalg.norm(rect_ordered[1] - rect_ordered[2])
        maxHeight = int(max(heightA, heightB))

        # Ensure aspect ratio is correct (credit card should be wider than tall)
        if maxHeight > maxWidth:
            maxWidth, maxHeight = maxHeight, maxWidth

        # Define destination points for the perspective transform
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(rect_ordered, dst)

        # Apply the perspective transform
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # Create new rect for the warped image (should be perfectly aligned now)
        center = (maxWidth / 2, maxHeight / 2)
        size = (maxWidth, maxHeight)
        angle = 0
        corrected_rect = (center, size, angle)

        logger.info(f"Applied perspective correction: angle={rect[2]:.1f}°, "
                   f"size=({maxWidth}x{maxHeight})")

        return warped, corrected_rect, True

    except Exception as e:
        logger.warning(f"Perspective correction failed: {str(e)}")
        return None, rect, False


def calculate_calibration(contour, rect, image):
    """
    Calculate pixels per millimeter from detected credit card.
    Applies perspective correction if card is tilted.

    Args:
        contour: Detected card contour
        rect: minAreaRect
        image: Original image

    Returns:
        Calibration data dictionary
    """
    if contour is None or rect is None:
        raise ValueError("REFERENCE_NOT_FOUND")

    # Try to apply perspective correction for better accuracy
    warped_image, corrected_rect, was_corrected = apply_perspective_correction(image, contour)

    # Use corrected measurements if perspective correction was applied
    if was_corrected and warped_image is not None:
        rect = corrected_rect
        logger.info("Using perspective-corrected measurements")

    width, height = rect[1]

    if width == 0 or height == 0:
        raise ValueError("REFERENCE_NOT_FOUND")

    card_width_px = max(width, height)
    card_height_px = min(width, height)

    # Calculate pixels per millimeter from width (more accurate)
    pixels_per_mm = card_width_px / CREDIT_CARD_WIDTH_MM

    # Calculate confidence based on aspect ratio accuracy
    aspect_ratio = card_width_px / card_height_px
    aspect_diff = abs(aspect_ratio - CREDIT_CARD_ASPECT_RATIO) / CREDIT_CARD_ASPECT_RATIO

    # Confidence: 1.0 perfect, decreases with aspect ratio deviation
    # Boost confidence if perspective correction was applied
    confidence = max(0.5, 1.0 - aspect_diff)
    if was_corrected:
        confidence = min(1.0, confidence * 1.1)  # 10% bonus for perspective correction

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    result = {
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
        },
        'perspectiveCorrected': was_corrected
    }

    return result


# =============================================================================
# DEBUG IMAGE SAVING
# =============================================================================

def save_debug_image(image, contour, rect, filename, calibration_result, debug_info=None):
    """
    Save debug image with detection annotations.

    Args:
        image: Original BGR image
        contour: Detected contour (or None)
        rect: minAreaRect (or None)
        filename: Output filename
        calibration_result: Calibration result dict (or None)
        debug_info: Debug information from detection strategies
    """
    try:
        debug_image = image.copy()

        if contour is not None and rect is not None:
            # Draw detected contour in green
            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 3)

            # Draw minimum area rectangle in blue
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(debug_image, [box], 0, (255, 0, 0), 2)

            # Draw center point
            center = rect[0]
            cv2.circle(debug_image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

            # Add calibration info
            if calibration_result:
                font = cv2.FONT_HERSHEY_SIMPLEX
                y_offset = 30

                # Background for text
                cv2.rectangle(debug_image, (10, 10), (550, 150), (0, 0, 0), -1)
                cv2.rectangle(debug_image, (10, 10), (550, 150), (0, 255, 0), 2)

                # Text lines
                cv2.putText(debug_image, "CREDIT CARD DETECTED",
                           (20, y_offset), font, 0.7, (0, 255, 0), 2)

                cv2.putText(debug_image,
                           f"PPM: {calibration_result['pixelsPerMillimeter']:.2f}",
                           (20, y_offset + 30), font, 0.6, (255, 255, 255), 2)

                cv2.putText(debug_image,
                           f"Confidence: {calibration_result['confidence']*100:.1f}%",
                           (20, y_offset + 60), font, 0.6, (255, 255, 255), 2)

                if 'detectionMethod' in calibration_result:
                    cv2.putText(debug_image,
                               f"Method: {calibration_result['detectionMethod']}",
                               (20, y_offset + 90), font, 0.5, (255, 255, 0), 2)

                if 'cardDimensions' in calibration_result:
                    card_dims = calibration_result['cardDimensions']
                    cv2.putText(debug_image,
                               f"Dims: {card_dims['width_px']:.0f}x{card_dims['height_px']:.0f}px",
                               (20, y_offset + 115), font, 0.5, (255, 255, 255), 2)
        else:
            # No detection
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(debug_image, (10, 10), (450, 60), (0, 0, 0), -1)
            cv2.rectangle(debug_image, (10, 10), (450, 60), (0, 0, 255), 2)
            cv2.putText(debug_image, "CREDIT CARD NOT DETECTED",
                       (20, 40), font, 0.7, (0, 0, 255), 2)

        # Save main debug image
        output_path = os.path.join(DEBUG_IMAGE_DIR, filename)
        cv2.imwrite(output_path, debug_image)
        logger.info(f"Debug image saved: {output_path}")

        # Save strategy-specific debug images
        if debug_info:
            base_name = filename.rsplit('.', 1)[0]
            save_strategy_debug_images(image, debug_info, base_name)

        return output_path

    except Exception as e:
        logger.error(f"Failed to save debug image: {str(e)}", exc_info=True)
        return None


def save_strategy_debug_images(image, debug_info, base_name):
    """
    Save individual debug images for each detection strategy.

    Args:
        image: Original BGR image
        debug_info: Dict with strategy results
        base_name: Base filename without extension
    """
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX

        for strategy_name, result in debug_info.items():
            if result is None:
                continue

            vis_image = image.copy()

            if result['contour'] is not None and result['rect'] is not None:
                # Draw detection
                cv2.drawContours(vis_image, [result['contour']], -1, (0, 255, 0), 3)

                box = cv2.boxPoints(result['rect'])
                box = np.int32(box)
                cv2.drawContours(vis_image, [box], 0, (255, 0, 0), 2)

                # Add text
                cv2.rectangle(vis_image, (10, 10), (500, 110), (0, 0, 0), -1)
                cv2.rectangle(vis_image, (10, 10), (500, 110), (0, 255, 0), 2)

                cv2.putText(vis_image, f"Strategy: {strategy_name}",
                           (20, 35), font, 0.6, (0, 255, 0), 2)
                cv2.putText(vis_image, f"Confidence: {result['confidence']:.3f}",
                           (20, 65), font, 0.6, (255, 255, 255), 2)

                if 'breakdown' in result and result['breakdown']:
                    breakdown = result['breakdown']
                    cv2.putText(vis_image,
                               f"Aspect: {breakdown.get('aspect_score', 0):.2f} " +
                               f"Area: {breakdown.get('area_score', 0):.2f} " +
                               f"Rect: {breakdown.get('rectangularity_score', 0):.2f}",
                               (20, 90), font, 0.45, (200, 200, 200), 1)
            else:
                # No detection
                cv2.rectangle(vis_image, (10, 10), (400, 50), (0, 0, 0), -1)
                cv2.rectangle(vis_image, (10, 10), (400, 50), (0, 0, 255), 2)
                cv2.putText(vis_image, f"{strategy_name}: NO DETECTION",
                           (20, 35), font, 0.6, (0, 0, 255), 2)

            # Save strategy image
            strategy_path = os.path.join(DEBUG_IMAGE_DIR, f"{base_name}_strategy_{strategy_name}.jpg")
            cv2.imwrite(strategy_path, vis_image)
            logger.debug(f"Saved strategy debug image: {strategy_path}")

    except Exception as e:
        logger.error(f"Failed to save strategy debug images: {str(e)}", exc_info=True)

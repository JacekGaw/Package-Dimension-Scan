"""
Quadrilateral Credit Card Detector

Detects credit cards by looking for 4-sided polygons with the correct aspect ratio.
Uses perspective transform for accurate measurements.
"""

import cv2
import numpy as np
import os
import logging

from src.detectors.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

# Credit card standard dimensions (ISO/IEC 7810 ID-1)
CREDIT_CARD_WIDTH_MM = 85.60
CREDIT_CARD_HEIGHT_MM = 53.98
CREDIT_CARD_ASPECT_RATIO = CREDIT_CARD_WIDTH_MM / CREDIT_CARD_HEIGHT_MM  # 1.586


class QuadrilateralDetector(BaseDetector):
    """
    Credit card detector using quadrilateral detection with perspective transform

    This method:
    1. Enhances image quality (CLAHE + Gaussian blur)
    2. Creates binary threshold
    3. Applies morphological operations
    4. Detects edges with Canny
    5. Finds 4-sided polygons
    6. Filters by aspect ratio
    7. Uses perspective transform for accurate measurements
    """

    def __init__(self, **config):
        super().__init__("quadrilateral_detection", **config)

        # Extract configuration with defaults
        self.aspect_ratio_min = config.get('aspect_ratio_min', 1.50)
        self.aspect_ratio_max = config.get('aspect_ratio_max', 1.65)
        self.clahe_clip_limit = config.get('clahe_clip_limit', 3.0)
        self.clahe_tile_size = config.get('clahe_tile_size', (8, 8))
        self.gaussian_kernel = config.get('gaussian_kernel', (5, 5))
        self.adaptive_thresh_block_size = config.get('adaptive_thresh_block_size', 15)
        self.adaptive_thresh_c = config.get('adaptive_thresh_c', 4)
        self.morph_kernel_size = config.get('morph_kernel_size', (5, 5))
        self.canny_low = config.get('canny_low', 50)
        self.canny_high = config.get('canny_high', 200)
        self.contour_epsilon_factor = config.get('contour_epsilon_factor', 0.02)

    def is_available(self) -> bool:
        """OpenCV is always available as it's a core dependency"""
        return True

    def detect(self, image: np.ndarray, debug_folder: str) -> DetectionResult:
        """
        Detect credit card using quadrilateral detection

        Args:
            image: BGR input image
            debug_folder: Path to save debug images

        Returns:
            DetectionResult with card detection data
        """
        try:
            logger.info("Starting quadrilateral credit card detection...")

            # Save original
            cv2.imwrite(os.path.join(debug_folder, "1_original.jpg"), image)

            # Step 1: Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(debug_folder, "2_grayscale.jpg"), gray)

            # Step 2: CLAHE enhancement
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_size
            )
            enhanced = clahe.apply(gray)
            cv2.imwrite(os.path.join(debug_folder, "3_clahe_enhanced.jpg"), enhanced)
            logger.info("  CLAHE enhancement applied")

            # Step 3: Gaussian blur
            blurred = cv2.GaussianBlur(enhanced, self.gaussian_kernel, 0)
            cv2.imwrite(os.path.join(debug_folder, "4_gaussian_blur.jpg"), blurred)
            logger.info("  Gaussian blur applied")

            # Step 4: Adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.adaptive_thresh_block_size,
                self.adaptive_thresh_c
            )
            cv2.imwrite(os.path.join(debug_folder, "5_adaptive_threshold.jpg"), thresh)
            logger.info("  Adaptive threshold applied")

            # Step 5: Morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(os.path.join(debug_folder, "6_morphology_close.jpg"), morphed)
            logger.info("  Morphological closing applied")

            # Step 6: Canny edge detection
            edges = cv2.Canny(morphed, self.canny_low, self.canny_high)
            cv2.imwrite(os.path.join(debug_folder, "7_canny_edges.jpg"), edges)
            logger.info("  Canny edge detection applied")

            # Step 7: Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            logger.info(f"  Found {len(contours)} contours")

            # Visualize all contours
            all_contours_vis = image.copy()
            cv2.drawContours(all_contours_vis, contours[:20], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_folder, "8_all_contours.jpg"), all_contours_vis)

            # Step 8: Look for quadrilateral with credit card aspect ratio
            card_contour = None
            card_approx = None
            best_confidence = 0
            best_ratio = 0

            for i, c in enumerate(contours[:50]):  # Check top 50 contours
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, self.contour_epsilon_factor * peri, True)

                # Must be a quadrilateral (4 sides)
                if len(approx) != 4:
                    continue

                # Compute dimensions
                pts = approx.reshape(4, 2)
                ordered_pts = self._order_points(pts)
                tl, tr, br, bl = ordered_pts

                width_px = np.linalg.norm(tr - tl)
                height_px = np.linalg.norm(bl - tl)

                if height_px == 0:
                    continue

                ratio = width_px / height_px

                # Check if ratio matches credit card
                if self.aspect_ratio_min < ratio < self.aspect_ratio_max:
                    # Calculate confidence based on how close to ideal ratio
                    confidence = 1.0 - abs(ratio - CREDIT_CARD_ASPECT_RATIO) / CREDIT_CARD_ASPECT_RATIO
                    confidence = max(0.5, min(1.0, confidence))

                    if confidence > best_confidence:
                        card_contour = c
                        card_approx = approx
                        best_confidence = confidence
                        best_ratio = ratio

                    logger.info(f"  Contour {i}: 4 sides, ratio={ratio:.3f}, confidence={confidence:.3f} ✓")

            if card_contour is None:
                logger.warning("  No quadrilateral with credit card aspect ratio found")
                self._save_failure_image(image, debug_folder)
                return DetectionResult(
                    success=False,
                    contour=None,
                    rect=None,
                    confidence=0.0,
                    method_name=self.name,
                    error="No 4-sided polygon with credit card aspect ratio found"
                )

            logger.info(f"  Credit card found! Ratio={best_ratio:.3f}, Confidence={best_confidence:.3f}")

            # Step 9: Use perspective transform for accurate measurements
            pts = card_approx.reshape(4, 2).astype(np.float32)
            width_px, height_px, warped_card = self._perspective_transform_and_measure(image, pts)
            logger.info(f"  Perspective transform - Width={width_px:.1f}px, Height={height_px:.1f}px")

            # Save warped card
            cv2.imwrite(os.path.join(debug_folder, "9a_warped_card.jpg"), warped_card)

            # Create minAreaRect for compatibility (but use accurate measurements)
            rect_center = cv2.minAreaRect(card_contour)[0]
            rect_angle = cv2.minAreaRect(card_contour)[2]
            rect = (rect_center, (width_px, height_px), rect_angle)

            # Save final result
            self._save_success_image(image, card_approx, rect, best_ratio, best_confidence,
                                    width_px, height_px, debug_folder)

            return DetectionResult(
                success=True,
                contour=card_contour,
                rect=rect,
                confidence=best_confidence,
                method_name=self.name,
                metadata={
                    'aspect_ratio': float(best_ratio),
                    'width_px': float(width_px),
                    'height_px': float(height_px),
                    'warped_card': warped_card
                }
            )

        except Exception as e:
            logger.error(f"Quadrilateral detection failed: {str(e)}", exc_info=True)
            return DetectionResult(
                success=False,
                contour=None,
                rect=None,
                confidence=0.0,
                method_name=self.name,
                error=str(e)
            )

    def _order_points(self, pts):
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

    def _perspective_transform_and_measure(self, image, corners):
        """
        Apply perspective transform to get accurate width/height measurements

        Args:
            image: Source image
            corners: 4 corner points of the card

        Returns:
            (width_px, height_px, warped_card_image)
        """
        # Order points
        pts = self._order_points(corners)
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

    def _save_success_image(self, image, card_approx, rect, ratio, confidence,
                           width_px, height_px, debug_folder):
        """Save visualization of successful detection"""
        result_vis = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

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
        cv2.putText(result_vis, f"Aspect Ratio: {ratio:.3f} (target: 1.586)", (20, 70), font, 0.5, (255, 255, 255), 1)
        cv2.putText(result_vis, f"Confidence: {confidence:.3f}", (20, 95), font, 0.5, (255, 255, 255), 1)
        cv2.putText(result_vis, f"Size: {width_px:.0f}x{height_px:.0f}px (perspective corrected)",
                   (20, 120), font, 0.45, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "10_FINAL_RESULT.jpg"), result_vis)

    def _save_failure_image(self, image, debug_folder):
        """Save visualization of failed detection"""
        failure_vis = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(failure_vis, (10, 10), (500, 80), (0, 0, 0), -1)
        cv2.rectangle(failure_vis, (10, 10), (500, 80), (0, 0, 255), 3)
        cv2.putText(failure_vis, "NO CARD DETECTED", (20, 40), font, 0.8, (0, 0, 255), 2)
        cv2.putText(failure_vis, "No 4-sided polygon with card aspect ratio",
                   (20, 70), font, 0.4, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "10_FINAL_RESULT.jpg"), failure_vis)

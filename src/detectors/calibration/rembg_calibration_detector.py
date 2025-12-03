"""
Rembg Credit Card Detector

Uses rembg AI background removal to detect credit cards.
This method is more robust in complex backgrounds or poor lighting conditions.
"""

import cv2
import numpy as np
import os
import logging
from PIL import Image

from src.detectors.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

# Credit card standard dimensions (ISO/IEC 7810 ID-1)
CREDIT_CARD_WIDTH_MM = 85.60
CREDIT_CARD_HEIGHT_MM = 53.98
CREDIT_CARD_ASPECT_RATIO = CREDIT_CARD_WIDTH_MM / CREDIT_CARD_HEIGHT_MM  # 1.586


class RembgCalibrationDetector(BaseDetector):
    """
    Credit card detector using rembg AI background removal

    This method:
    1. Uses rembg to remove background and isolate card
    2. Finds card contour from foreground mask
    3. Validates aspect ratio matches credit card
    4. Uses perspective transform for accurate measurements

    More robust than edge detection in:
    - Complex backgrounds
    - Poor lighting
    - Low contrast scenarios
    """

    def __init__(self, **config):
        super().__init__("rembg_calibration", **config)

        self.model_name = config.get('model_name', 'u2net')
        self.aspect_ratio_min = config.get('aspect_ratio_min', 1.50)
        self.aspect_ratio_max = config.get('aspect_ratio_max', 1.65)
        self.min_area_ratio = config.get('min_area_ratio', 0.01)
        self.morph_kernel_size = config.get('morph_kernel_size', (5, 5))

    def is_available(self) -> bool:
        """Check if rembg library is installed"""
        try:
            from rembg import remove
            return True
        except ImportError:
            return False

    def detect(self, image: np.ndarray, debug_folder: str) -> DetectionResult:
        """
        Detect credit card using rembg AI background removal

        Args:
            image: BGR input image
            debug_folder: Path to save debug images

        Returns:
            DetectionResult with card detection data
        """
        try:
            from rembg import remove

            logger.info("Starting rembg credit card detection...")

            # Save original
            cv2.imwrite(os.path.join(debug_folder, "1_original.jpg"), image)

            # Step 1: Convert OpenCV BGR to PIL RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            # Step 2: Apply rembg to get mask only
            logger.info("  Running rembg background removal...")
            mask_pil = remove(image_pil, only_mask=True)

            # Convert PIL mask to numpy
            mask = np.array(mask_pil)
            cv2.imwrite(os.path.join(debug_folder, "2_rembg_mask.jpg"), mask)
            logger.info("  rembg mask generated")

            # Step 3: Clean up mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_size)
            mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
            cv2.imwrite(os.path.join(debug_folder, "3_rembg_mask_cleaned.jpg"), mask_cleaned)

            # Step 4: Find contours
            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.warning("  rembg: No contours found in mask")
                self._save_failure_image(image, debug_folder)
                return DetectionResult(
                    success=False,
                    contour=None,
                    rect=None,
                    confidence=0.0,
                    method_name=self.name,
                    error="No contours found in mask"
                )

            # Sort by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            logger.info(f"  Found {len(contours)} contours")

            # Visualize all contours
            all_contours_vis = image.copy()
            cv2.drawContours(all_contours_vis, contours[:10], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_folder, "4_all_contours.jpg"), all_contours_vis)

            # Step 5: Find contour with credit card aspect ratio
            card_contour = None
            best_confidence = 0
            best_ratio = 0
            best_rect = None

            for i, contour in enumerate(contours[:10]):  # Check top 10 contours
                area = cv2.contourArea(contour)

                # Check minimum area
                min_area = (image.shape[0] * image.shape[1]) * self.min_area_ratio
                if area < min_area:
                    continue

                # Get minimum area bounding rectangle
                rect = cv2.minAreaRect(contour)
                center, (width, height), angle = rect

                if width == 0 or height == 0:
                    continue

                # Calculate aspect ratio
                ratio = max(width, height) / min(width, height)

                # Check if ratio matches credit card
                if self.aspect_ratio_min < ratio < self.aspect_ratio_max:
                    # Calculate confidence based on how close to ideal ratio
                    confidence = 1.0 - abs(ratio - CREDIT_CARD_ASPECT_RATIO) / CREDIT_CARD_ASPECT_RATIO
                    confidence = max(0.5, min(1.0, confidence))

                    if confidence > best_confidence:
                        card_contour = contour
                        best_confidence = confidence
                        best_ratio = ratio
                        best_rect = rect

                    logger.info(f"  Contour {i}: ratio={ratio:.3f}, confidence={confidence:.3f} ✓")

            if card_contour is None:
                logger.warning("  rembg: No contour with credit card aspect ratio found")
                self._save_failure_image(image, debug_folder)
                return DetectionResult(
                    success=False,
                    contour=None,
                    rect=None,
                    confidence=0.0,
                    method_name=self.name,
                    error="No contour with credit card aspect ratio found"
                )

            logger.info(f"  Credit card found! Ratio={best_ratio:.3f}, Confidence={best_confidence:.3f}")

            # Step 6: Use perspective transform for accurate measurements
            # Get 4 corners by approximating the contour
            peri = cv2.arcLength(card_contour, True)
            approx = cv2.approxPolyDP(card_contour, 0.02 * peri, True)

            # If we have 4 corners, use perspective transform
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                width_px, height_px, warped_card = self._perspective_transform_and_measure(image, pts)
                logger.info(f"  Perspective transform - Width={width_px:.1f}px, Height={height_px:.1f}px")

                # Save warped card
                cv2.imwrite(os.path.join(debug_folder, "5_warped_card.jpg"), warped_card)

                # Update rect with accurate measurements
                rect_center = best_rect[0]
                rect_angle = best_rect[2]
                rect = (rect_center, (width_px, height_px), rect_angle)
            else:
                # Use minAreaRect measurements
                rect = best_rect
                width_px = max(rect[1])
                height_px = min(rect[1])
                logger.info(f"  Using minAreaRect - Width={width_px:.1f}px, Height={height_px:.1f}px")

            # Save final result
            self._save_success_image(image, card_contour, rect, best_ratio, best_confidence,
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
                    'height_px': float(height_px)
                }
            )

        except Exception as e:
            logger.error(f"Rembg calibration detection failed: {str(e)}", exc_info=True)
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

    def _save_success_image(self, image, card_contour, rect, ratio, confidence,
                           width_px, height_px, debug_folder):
        """Save visualization of successful detection"""
        result_vis = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw contour
        cv2.drawContours(result_vis, [card_contour], -1, (0, 255, 0), 3)

        # Draw bounding rectangle
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(result_vis, [box], 0, (255, 0, 0), 2)

        # Add text overlay
        cv2.rectangle(result_vis, (10, 10), (600, 140), (0, 0, 0), -1)
        cv2.rectangle(result_vis, (10, 10), (600, 140), (0, 255, 0), 3)
        cv2.putText(result_vis, "REMBG CALIBRATION SUCCESS", (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(result_vis, f"Aspect Ratio: {ratio:.3f} (target: 1.586)", (20, 70), font, 0.5, (255, 255, 255), 1)
        cv2.putText(result_vis, f"Confidence: {confidence:.3f}", (20, 95), font, 0.5, (255, 255, 255), 1)
        cv2.putText(result_vis, f"Size: {width_px:.0f}x{height_px:.0f}px",
                   (20, 120), font, 0.45, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "6_FINAL_RESULT.jpg"), result_vis)

    def _save_failure_image(self, image, debug_folder):
        """Save visualization of failed detection"""
        failure_vis = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(failure_vis, (10, 10), (500, 80), (0, 0, 0), -1)
        cv2.rectangle(failure_vis, (10, 10), (500, 80), (0, 0, 255), 3)
        cv2.putText(failure_vis, "NO CARD DETECTED", (20, 40), font, 0.8, (0, 0, 255), 2)
        cv2.putText(failure_vis, "No contour with card aspect ratio",
                   (20, 70), font, 0.4, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "6_FINAL_RESULT.jpg"), failure_vis)

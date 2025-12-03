"""
Adaptive Thresholding Credit Card Detector

Uses adaptive thresholding for credit card detection in varying illumination.
This method calculates different thresholds for different regions of the image.
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


class AdaptiveCalibrationDetector(BaseDetector):
    """
    Credit card detector using adaptive thresholding

    Adaptive thresholding calculates threshold values for smaller regions
    of the image, making it much better than global thresholding for images
    with varying illumination (shadows, gradients, etc.).

    Two methods available:
    - ADAPTIVE_THRESH_MEAN_C: Threshold is mean of neighborhood
    - ADAPTIVE_THRESH_GAUSSIAN_C: Threshold is weighted sum of neighborhood

    Advantages:
    - Excellent for uneven lighting
    - Handles shadows and gradients
    - Fast processing
    - No manual threshold selection
    """

    def __init__(self, **config):
        super().__init__("adaptive_calibration", **config)

        self.aspect_ratio_min = config.get('aspect_ratio_min', 1.50)
        self.aspect_ratio_max = config.get('aspect_ratio_max', 1.65)
        self.min_area_ratio = config.get('min_area_ratio', 0.01)
        self.morph_kernel_size = config.get('morph_kernel_size', (5, 5))
        self.block_size = config.get('block_size', 11)  # Must be odd
        self.c_constant = config.get('c_constant', 2)
        self.method = config.get('method', 'gaussian')  # 'mean' or 'gaussian'
        self.use_blur = config.get('use_blur', True)
        self.blur_kernel = config.get('blur_kernel', (5, 5))

    def is_available(self) -> bool:
        """OpenCV is always available as it's a core dependency"""
        return True

    def detect(self, image: np.ndarray, debug_folder: str) -> DetectionResult:
        """
        Detect credit card using adaptive thresholding

        Args:
            image: BGR input image
            debug_folder: Path to save debug images

        Returns:
            DetectionResult with card detection data
        """
        try:
            logger.info("Starting adaptive calibration detection...")

            # Save original
            cv2.imwrite(os.path.join(debug_folder, "1_original.jpg"), image)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(debug_folder, "2_grayscale.jpg"), gray)

            # Optional Gaussian blur
            if self.use_blur:
                blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
                cv2.imwrite(os.path.join(debug_folder, "3_gaussian_blur.jpg"), blurred)
                logger.info("  Gaussian blur applied")
            else:
                blurred = gray

            # Select adaptive method
            if self.method == 'mean':
                adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
                method_name = "Mean"
            else:  # gaussian
                adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                method_name = "Gaussian"

            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                adaptive_method,
                cv2.THRESH_BINARY_INV,
                self.block_size,
                self.c_constant
            )
            cv2.imwrite(os.path.join(debug_folder, "4_adaptive_threshold.jpg"), thresh)
            logger.info(f"  Adaptive threshold applied ({method_name}, block={self.block_size}, C={self.c_constant})")

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
            cv2.imwrite(os.path.join(debug_folder, "5_morphology.jpg"), morphed)
            logger.info("  Morphological operations applied")

            # Find contours
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            logger.info(f"  Found {len(contours)} contours")

            # Visualize all contours
            all_contours_vis = image.copy()
            cv2.drawContours(all_contours_vis, contours[:20], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_folder, "6_all_contours.jpg"), all_contours_vis)

            # Look for credit card by aspect ratio
            card_contour = None
            best_confidence = 0
            best_ratio = 0
            best_rect = None

            for i, contour in enumerate(contours[:20]):
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
                    # Calculate confidence
                    confidence = 1.0 - abs(ratio - CREDIT_CARD_ASPECT_RATIO) / CREDIT_CARD_ASPECT_RATIO
                    confidence = max(0.5, min(1.0, confidence))

                    if confidence > best_confidence:
                        card_contour = contour
                        best_confidence = confidence
                        best_ratio = ratio
                        best_rect = rect

                    logger.info(f"  Contour {i}: ratio={ratio:.3f}, confidence={confidence:.3f} ✓")

            if card_contour is None:
                logger.warning("  Adaptive: No contour with credit card aspect ratio found")
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

            # Get accurate measurements
            width_px = max(best_rect[1])
            height_px = min(best_rect[1])

            # Save success visualization
            self._save_success_image(image, card_contour, best_rect, best_ratio,
                                    best_confidence, width_px, height_px,
                                    method_name, debug_folder)

            return DetectionResult(
                success=True,
                contour=card_contour,
                rect=best_rect,
                confidence=best_confidence,
                method_name=self.name,
                metadata={
                    'aspect_ratio': float(best_ratio),
                    'width_px': float(width_px),
                    'height_px': float(height_px),
                    'adaptive_method': method_name,
                    'block_size': self.block_size
                }
            )

        except Exception as e:
            logger.error(f"Adaptive calibration detection failed: {str(e)}", exc_info=True)
            return DetectionResult(
                success=False,
                contour=None,
                rect=None,
                confidence=0.0,
                method_name=self.name,
                error=str(e)
            )

    def _save_success_image(self, image, card_contour, rect, ratio, confidence,
                           width_px, height_px, method_name, debug_folder):
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
        cv2.rectangle(result_vis, (10, 10), (600, 160), (0, 0, 0), -1)
        cv2.rectangle(result_vis, (10, 10), (600, 160), (0, 255, 0), 3)
        cv2.putText(result_vis, "ADAPTIVE CALIBRATION SUCCESS", (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(result_vis, f"Aspect Ratio: {ratio:.3f} (target: 1.586)",
                   (20, 70), font, 0.5, (255, 255, 255), 1)
        cv2.putText(result_vis, f"Confidence: {confidence:.3f}",
                   (20, 95), font, 0.5, (255, 255, 255), 1)
        cv2.putText(result_vis, f"Size: {width_px:.0f}x{height_px:.0f}px",
                   (20, 120), font, 0.45, (200, 200, 200), 1)
        cv2.putText(result_vis, f"Method: {method_name}",
                   (20, 145), font, 0.45, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "7_FINAL_RESULT.jpg"), result_vis)

    def _save_failure_image(self, image, debug_folder):
        """Save visualization of failed detection"""
        failure_vis = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(failure_vis, (10, 10), (500, 80), (0, 0, 0), -1)
        cv2.rectangle(failure_vis, (10, 10), (500, 80), (0, 0, 255), 3)
        cv2.putText(failure_vis, "NO CARD DETECTED", (20, 40), font, 0.8, (0, 0, 255), 2)
        cv2.putText(failure_vis, "No contour with card aspect ratio",
                   (20, 70), font, 0.4, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "7_FINAL_RESULT.jpg"), failure_vis)

"""
Adaptive Thresholding Detector

Uses adaptive thresholding for object detection in images with varying illumination.
This method calculates different thresholds for different regions of the image.
"""

import cv2
import numpy as np
import os
import logging

from src.detectors.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class AdaptiveDetector(BaseDetector):
    """
    Package detector using adaptive thresholding

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
        super().__init__("adaptive_threshold", **config)

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
        Detect package using adaptive thresholding

        Args:
            image: BGR input image
            debug_folder: Path to save debug images

        Returns:
            DetectionResult with package detection data
        """
        try:
            logger.info("Starting adaptive thresholding detection...")

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(debug_folder, "2_grayscale.jpg"), gray)

            # Optional Gaussian blur to reduce noise
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

            # Clean up with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_size)
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
            cv2.imwrite(os.path.join(debug_folder, "5_morphology.jpg"), morphed)
            logger.info("  Morphological operations applied")

            # Find contours
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.error("  Adaptive: No contours found!")
                return DetectionResult(
                    success=False,
                    contour=None,
                    rect=None,
                    confidence=0.0,
                    method_name=self.name,
                    error="No contours found"
                )

            # Sort by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            logger.info(f"  Adaptive: Found {len(contours)} contours")

            # Visualize all contours
            all_contours_vis = image.copy()
            cv2.drawContours(all_contours_vis, contours[:10], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_folder, "6_all_contours.jpg"), all_contours_vis)

            # Get largest contour (the package)
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # Filter out tiny contours (noise)
            min_area = (image.shape[0] * image.shape[1]) * self.min_area_ratio
            if area < min_area:
                logger.error(f"  Adaptive: Largest contour too small: {area:.0f}px² (min: {min_area:.0f}px²)")
                return DetectionResult(
                    success=False,
                    contour=None,
                    rect=None,
                    confidence=0.0,
                    method_name=self.name,
                    error=f"Contour too small: {area:.0f}px² < {min_area:.0f}px²"
                )

            logger.info(f"  Adaptive: Largest contour area: {area:.0f}px²")

            # Get minimum area bounding rectangle
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            center, (width, height), angle = rect

            # Ensure width >= height
            longer_side_px = max(width, height)
            shorter_side_px = min(width, height)

            logger.info(f"  Adaptive: Bounding rectangle - {longer_side_px:.1f}px × {shorter_side_px:.1f}px")

            # Visualize detection
            self._save_detection_image(image, largest, box, center, longer_side_px,
                                      shorter_side_px, area, method_name, debug_folder)

            # Calculate confidence based on contour quality
            bbox_area = longer_side_px * shorter_side_px
            if bbox_area > 0:
                fill_ratio = area / bbox_area
                # Adaptive threshold gives good results with varying lighting
                confidence = min(0.88, 0.55 + fill_ratio * 0.33)  # Range: 0.55 - 0.88
            else:
                confidence = 0.55

            return DetectionResult(
                success=True,
                contour=largest,
                rect=rect,
                confidence=confidence,
                method_name=self.name,
                metadata={
                    'longer_side_px': float(longer_side_px),
                    'shorter_side_px': float(shorter_side_px),
                    'area': float(area),
                    'angle': float(angle),
                    'adaptive_method': method_name,
                    'block_size': self.block_size
                }
            )

        except Exception as e:
            logger.error(f"Adaptive threshold detection failed: {str(e)}", exc_info=True)
            return DetectionResult(
                success=False,
                contour=None,
                rect=None,
                confidence=0.0,
                method_name=self.name,
                error=str(e)
            )

    def _save_detection_image(self, image, largest, box, center, longer_side_px,
                             shorter_side_px, area, method_name, debug_folder):
        """Save visualization of detection"""
        result_vis = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.drawContours(result_vis, [largest], -1, (0, 255, 0), 3)
        cv2.drawContours(result_vis, [box], 0, (255, 0, 0), 2)
        cv2.circle(result_vis, (int(center[0]), int(center[1])), 8, (0, 0, 255), -1)

        # Add text overlay
        cv2.rectangle(result_vis, (10, 10), (550, 150), (0, 0, 0), -1)
        cv2.rectangle(result_vis, (10, 10), (550, 150), (0, 255, 0), 3)
        cv2.putText(result_vis, "ADAPTIVE THRESHOLD DETECTION", (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(result_vis, f"Size: {longer_side_px:.0f} x {shorter_side_px:.0f} px",
                   (20, 75), font, 0.6, (255, 255, 255), 2)
        cv2.putText(result_vis, f"Area: {area:.0f} px²",
                   (20, 105), font, 0.5, (200, 200, 200), 1)
        cv2.putText(result_vis, f"Method: {method_name}",
                   (20, 135), font, 0.5, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "7_ADAPTIVE_DETECTION.jpg"), result_vis)

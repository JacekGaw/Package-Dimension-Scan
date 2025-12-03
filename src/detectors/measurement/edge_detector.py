"""
Edge Detection Detector

Traditional computer vision approach using Canny edge detection.
This is the fallback method that always works but has lower accuracy (60-75%).
"""

import cv2
import numpy as np
import os
import logging

from src.detectors.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class EdgeDetector(BaseDetector):
    """
    Package detector using traditional edge detection (fallback method)

    Uses CLAHE enhancement, Gaussian blur, and Canny edge detection.
    This method always works but may have lower accuracy compared to AI methods.
    """

    def __init__(self, **config):
        super().__init__("edge_detection", **config)

        self.clahe_clip_limit = config.get('clahe_clip_limit', 2.0)
        self.clahe_tile_size = config.get('clahe_tile_size', (8, 8))
        self.gaussian_kernel = config.get('gaussian_kernel', (5, 5))
        self.canny_low = config.get('canny_low', 50)
        self.canny_high = config.get('canny_high', 150)
        self.min_area_ratio = config.get('min_area_ratio', 0.01)

    def is_available(self) -> bool:
        """OpenCV is always available as it's a core dependency"""
        return True

    def detect(self, image: np.ndarray, debug_folder: str) -> DetectionResult:
        """
        Detect package using traditional edge detection

        Args:
            image: BGR input image
            debug_folder: Path to save debug images

        Returns:
            DetectionResult with package detection data
        """
        try:
            logger.info("Starting edge detection...")

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(debug_folder, "5_grayscale.jpg"), gray)

            # Enhance with CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_size
            )
            enhanced = clahe.apply(gray)
            cv2.imwrite(os.path.join(debug_folder, "6_clahe_enhanced.jpg"), enhanced)
            logger.info("  CLAHE enhancement applied")

            # Gaussian blur
            blurred = cv2.GaussianBlur(enhanced, self.gaussian_kernel, 0)
            cv2.imwrite(os.path.join(debug_folder, "7_gaussian_blur.jpg"), blurred)

            # Edge detection
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            cv2.imwrite(os.path.join(debug_folder, "8_canny_edges.jpg"), edges)
            logger.info("  Canny edge detection applied")

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.error("  Edge detection: No contours found!")
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
            logger.info(f"  Edge detection: Found {len(contours)} contours")

            # Visualize all contours
            all_contours_vis = image.copy()
            cv2.drawContours(all_contours_vis, contours[:10], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_folder, "9_all_contours.jpg"), all_contours_vis)

            # Get largest contour (the package)
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # Filter out tiny contours (noise)
            min_area = (image.shape[0] * image.shape[1]) * self.min_area_ratio
            if area < min_area:
                logger.error(f"  Edge detection: Largest contour too small: {area:.0f}px² (min: {min_area:.0f}px²)")
                return DetectionResult(
                    success=False,
                    contour=None,
                    rect=None,
                    confidence=0.0,
                    method_name=self.name,
                    error=f"Contour too small: {area:.0f}px² < {min_area:.0f}px²"
                )

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
            logger.info(f"  Edge detection: Center: ({center[0]:.1f}, {center[1]:.1f}), Angle: {angle:.1f}°")

            # Visualize detection
            self._save_detection_image(image, largest, box, center, longer_side_px,
                                      shorter_side_px, area, debug_folder)

            # Edge detection doesn't provide confidence, so we estimate based on contour quality
            # Better contours have higher area relative to bounding box
            bbox_area = longer_side_px * shorter_side_px
            if bbox_area > 0:
                fill_ratio = area / bbox_area
                confidence = min(0.85, 0.5 + fill_ratio * 0.35)  # Range: 0.5 - 0.85
            else:
                confidence = 0.5

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
                    'angle': float(angle)
                }
            )

        except Exception as e:
            logger.error(f"Edge detection failed: {str(e)}", exc_info=True)
            return DetectionResult(
                success=False,
                contour=None,
                rect=None,
                confidence=0.0,
                method_name=self.name,
                error=str(e)
            )

    def _save_detection_image(self, image, largest, box, center, longer_side_px,
                             shorter_side_px, area, debug_folder):
        """Save visualization of detection"""
        result_vis = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.drawContours(result_vis, [largest], -1, (0, 255, 0), 3)
        cv2.drawContours(result_vis, [box], 0, (255, 0, 0), 2)
        cv2.circle(result_vis, (int(center[0]), int(center[1])), 8, (0, 0, 255), -1)

        # Add text overlay
        cv2.rectangle(result_vis, (10, 10), (500, 130), (0, 0, 0), -1)
        cv2.rectangle(result_vis, (10, 10), (500, 130), (0, 255, 0), 3)
        cv2.putText(result_vis, "EDGE DETECTION SUCCESS", (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(result_vis, f"Size: {longer_side_px:.0f} x {shorter_side_px:.0f} px",
                   (20, 75), font, 0.6, (255, 255, 255), 2)
        cv2.putText(result_vis, f"Area: {area:.0f} px²",
                   (20, 105), font, 0.5, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "10_EDGE_DETECTION.jpg"), result_vis)

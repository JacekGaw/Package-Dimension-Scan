"""
YOLO Object Detection Detector

Uses YOLO (You Only Look Once) for fast object detection with segmentation.
This is the fastest method but only works for objects in the COCO dataset (80 classes).
"""

import cv2
import numpy as np
import os
import logging

from src.detectors.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """
    Package detector using YOLO object detection

    Uses YOLOv8 segmentation model for fast and accurate detection.
    Detects objects from 80 COCO classes including boxes, bottles, books, etc.
    Processing time: 50-200ms on CPU.
    """

    def __init__(self, **config):
        super().__init__("yolo", **config)

        self.model_name = config.get('model', 'yolov8n-seg.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.min_area_ratio = config.get('min_area_ratio', 0.01)

        # Morphological operations config
        self.morph_kernel_size = config.get('morph_kernel_size', (5, 5))
        self.skip_morph_close = config.get('skip_morph_close', False)
        self.skip_morph_open = config.get('skip_morph_open', False)
        self.reverse_morph_order = config.get('reverse_morph_order', False)

        # Erosion config
        self.erosion_iterations = config.get('erosion_iterations', 0)
        self.erosion_kernel_size = config.get('erosion_kernel_size', (3, 3))

        self.model = None  # Loaded on first use (lazy loading)

    def is_available(self) -> bool:
        """Check if YOLO library is installed"""
        try:
            from ultralytics import YOLO
            return True
        except ImportError:
            return False

    def detect(self, image: np.ndarray, debug_folder: str) -> DetectionResult:
        """
        Detect package using YOLO object detection

        Args:
            image: BGR input image
            debug_folder: Path to save debug images

        Returns:
            DetectionResult with package detection data
        """
        try:
            logger.info("Starting YOLO object detection...")

            # Load model on first use
            if self.model is None:
                from ultralytics import YOLO
                logger.info(f"  Loading YOLO model '{self.model_name}'...")
                self.model = YOLO(self.model_name)
                logger.info("  YOLO model loaded")

            # Run inference
            logger.info("  Running YOLO inference...")
            results = self.model(image, verbose=False)

            if len(results) == 0 or results[0].masks is None:
                logger.warning("  YOLO: No objects detected")
                return DetectionResult(
                    success=False,
                    contour=None,
                    rect=None,
                    confidence=0.0,
                    method_name=self.name,
                    error="No objects detected"
                )

            # Get masks and boxes
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            if len(masks) == 0:
                logger.warning("  YOLO: No masks generated")
                return DetectionResult(
                    success=False,
                    contour=None,
                    rect=None,
                    confidence=0.0,
                    method_name=self.name,
                    error="No masks generated"
                )

            # Get the largest detected object
            areas = [cv2.countNonZero(mask.astype(np.uint8)) for mask in masks]
            largest_idx = np.argmax(areas)

            mask = masks[largest_idx]
            detected_class = int(classes[largest_idx])
            confidence = float(confidences[largest_idx])

            # Get class name
            class_names = results[0].names
            class_name = class_names[detected_class]

            logger.info(f"  YOLO detected: {class_name} (class {detected_class}) with confidence {confidence:.2f}")

            # Convert mask to uint8 and resize to image size
            mask_uint8 = (mask * 255).astype(np.uint8)
            if mask_uint8.shape != image.shape[:2]:
                mask_uint8 = cv2.resize(mask_uint8, (image.shape[1], image.shape[0]))

            cv2.imwrite(os.path.join(debug_folder, "2_yolo_mask.jpg"), mask_uint8)

            # Clean up mask with morphological operations
            mask_cleaned = mask_uint8.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_size)

            # Apply morphological operations based on configuration
            if self.reverse_morph_order:
                # Contract first, then expand (stricter masks)
                if not self.skip_morph_open:
                    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
                if not self.skip_morph_close:
                    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
            else:
                # Default: Expand first, then contract (smoother masks)
                if not self.skip_morph_close:
                    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
                if not self.skip_morph_open:
                    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

            # Apply erosion if configured (shrinks mask for tighter fit)
            if self.erosion_iterations > 0:
                erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.erosion_kernel_size)
                mask_cleaned = cv2.erode(mask_cleaned, erosion_kernel, iterations=self.erosion_iterations)

            cv2.imwrite(os.path.join(debug_folder, "3_yolo_mask_cleaned.jpg"), mask_cleaned)

            # Find contours
            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.warning("  YOLO: No contours found in mask")
                return DetectionResult(
                    success=False,
                    contour=None,
                    rect=None,
                    confidence=0.0,
                    method_name=self.name,
                    error="No contours found"
                )

            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # Check minimum area
            min_area = (image.shape[0] * image.shape[1]) * self.min_area_ratio
            if area < min_area:
                logger.warning(f"  YOLO: Contour too small: {area:.0f}px² (min: {min_area:.0f}px²)")
                return DetectionResult(
                    success=False,
                    contour=None,
                    rect=None,
                    confidence=0.0,
                    method_name=self.name,
                    error=f"Contour too small: {area:.0f}px² < {min_area:.0f}px²"
                )

            # Get minimum area bounding rectangle
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            center, (width, height), angle = rect

            # Ensure width >= height
            longer_side_px = max(width, height)
            shorter_side_px = min(width, height)

            logger.info(f"  YOLO SUCCESS: {longer_side_px:.1f}px × {shorter_side_px:.1f}px")

            # Visualize detection
            self._save_detection_image(image, largest, box, center, longer_side_px,
                                      shorter_side_px, area, confidence, class_name,
                                      detected_class, debug_folder)

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
                    'object_class': class_name,
                    'object_class_id': detected_class
                }
            )

        except Exception as e:
            logger.error(f"YOLO detection failed: {str(e)}", exc_info=True)
            return DetectionResult(
                success=False,
                contour=None,
                rect=None,
                confidence=0.0,
                method_name=self.name,
                error=str(e)
            )

    def _save_detection_image(self, image, largest, box, center, longer_side_px,
                             shorter_side_px, area, confidence, class_name,
                             detected_class, debug_folder):
        """Save visualization of detection"""
        result_vis = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.drawContours(result_vis, [largest], -1, (0, 255, 0), 3)
        cv2.drawContours(result_vis, [box], 0, (255, 0, 0), 2)
        cv2.circle(result_vis, (int(center[0]), int(center[1])), 8, (0, 0, 255), -1)

        # Add text overlay
        cv2.rectangle(result_vis, (10, 10), (550, 150), (0, 0, 0), -1)
        cv2.rectangle(result_vis, (10, 10), (550, 150), (0, 255, 0), 3)
        cv2.putText(result_vis, f"YOLO DETECTION: {class_name.upper()}", (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(result_vis, f"Size: {longer_side_px:.0f} x {shorter_side_px:.0f} px",
                   (20, 75), font, 0.6, (255, 255, 255), 2)
        cv2.putText(result_vis, f"Area: {area:.0f} px² | Conf: {confidence:.2f}",
                   (20, 110), font, 0.5, (200, 200, 200), 1)
        cv2.putText(result_vis, f"Class: {detected_class} ({class_name})",
                   (20, 140), font, 0.5, (200, 200, 200), 1)

        cv2.imwrite(os.path.join(debug_folder, "4_YOLO_DETECTION.jpg"), result_vis)

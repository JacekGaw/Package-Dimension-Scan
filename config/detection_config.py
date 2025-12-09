"""
Detection Configuration

Defines which detection methods to use and their priority order.
Easily configure by enabling/disabling methods or changing their order.
"""

import os
from typing import List


class DetectionConfig:
    """
    Configuration for detection methods

    Modify these arrays to change which methods are used and in what order.
    Methods are tried sequentially until one succeeds.
    """

    # ============================================================================
    # CALIBRATION DETECTION METHODS
    # ============================================================================
    # Methods for detecting credit card during calibration
    # Tried in order from first to last until one succeeds
    CALIBRATION_METHODS = [
        {
            'type': 'quadrilateral',
            'enabled': False,
            'config': {
                'aspect_ratio_min': 1.50,
                'aspect_ratio_max': 1.65,
                'clahe_clip_limit': 3.0,
                'clahe_tile_size': (8, 8),
                'gaussian_kernel': (5, 5),
                'adaptive_thresh_block_size': 15,
                'adaptive_thresh_c': 4,
                'morph_kernel_size': (5, 5),
                'canny_low': 50,
                'canny_high': 200,
                'contour_epsilon_factor': 0.02
            }
        },
        {
            'type': 'adaptive_calibration',
            'enabled': False,
            'config': {
                'aspect_ratio_min': 1.50,
                'aspect_ratio_max': 1.65,
                'min_area_ratio': 0.01,
                'morph_kernel_size': (5, 5),
                'block_size': 11,
                'c_constant': 2,
                'method': 'gaussian',
                'use_blur': True,
                'blur_kernel': (5, 5)
            }
        },
        {
            'type': 'otsu_calibration',
            'enabled': False,
            'config': {
                'aspect_ratio_min': 1.50,
                'aspect_ratio_max': 1.65,
                'min_area_ratio': 0.01,
                'morph_kernel_size': (5, 5),
                'use_blur': True,
                'blur_kernel': (5, 5)
            }
        },
        {
            'type': 'rembg_calibration',
            'enabled': True,
            'config': {
                'model_name': 'u2net',
                'aspect_ratio_min': 1.50,
                'aspect_ratio_max': 1.65,
                'min_area_ratio': 0.005,
                'morph_kernel_size': (4, 4)  # Match measurement config for consistency
            }
        },
    ]

    # ============================================================================
    # PACKAGE MEASUREMENT DETECTION METHODS
    # ============================================================================
    # Methods for detecting packages during measurement
    # Tried in order from first to last until one succeeds
    MEASUREMENT_METHODS = [
        # AI METHODS FIRST - They understand whole objects, not just edges
        {
            'type': 'yolo',
            'enabled': True,  # ENABLED - Best for whole object detection
            'config': {
                'model': 'yolov8n-seg.pt',  # Nano model - fastest
                'confidence_threshold': 0.5,  # Increase to 0.6-0.7 for stricter detections
                'min_area_ratio': 0.02,

                # MORPHOLOGICAL OPERATIONS - Control mask cleanup
                'morph_kernel_size': (1, 1),  # Kernel for OPEN/CLOSE operations. Smaller = less smoothing (try 3x3 or 2x2)
                'skip_morph_close': False,  # Set True to skip MORPH_CLOSE (expansion) step for tighter masks
                'skip_morph_open': False,  # Set True to skip MORPH_OPEN (contraction) step
                'reverse_morph_order': False,  # Set True to contract before expand (stricter masks)

                # EROSION - Shrinks mask inward from edges for tighter fit
                'erosion_iterations': 3,  # Number of erosion passes (0-3). Higher = tighter mask but may lose detail
                'erosion_kernel_size': (1, 1),  # Erosion kernel size. Larger = more aggressive shrinking
            }
        },
        {
            'type': 'rembg',
            'enabled': True,  # AI segmentation - works on any object
            'config': {
                'model_name': 'u2net',
                'min_area_ratio': 0.005,  # 1% minimum area

                # ALPHA MATTING - DISABLED for consistency
                'alpha_matting': True,  # Disabled - was causing inconsistent masks
                'alpha_matting_foreground_threshold': 240,
                'alpha_matting_background_threshold': 10,
                'alpha_matting_erode_size': 10,

                # MORPHOLOGICAL OPERATIONS - Keep minimal for consistency
                'morph_kernel_size': (3, 3),  # Small kernel for basic cleanup
                'skip_morph_close': False,  # Keep basic CLOSE for hole filling
                'skip_morph_open': False,  # Keep basic OPEN for noise removal
                'reverse_morph_order': False,  # Standard order: close then open

                # EROSION - DISABLED for now to maintain detection consistency
                'erosion_iterations': 3,  # No additional erosion
                'erosion_kernel_size': (1, 1),
            }
        },
        # TRADITIONAL CV METHODS - DISABLED (they detect logos/text instead of packages)
        # Only enable these if you need last-resort fallbacks
        {
            'type': 'edge_detection',
            'enabled': False,  # PROBLEM: Detects logo edges, text, patterns
            'config': {
                'clahe_clip_limit': 2.0,
                'clahe_tile_size': (8, 8),
                'gaussian_kernel': (5, 5),
                'canny_low': 50,
                'canny_high': 150,
                'min_area_ratio': 0.15  # Increased to filter small features
            }
        },
        {
            'type': 'adaptive_threshold',
            'enabled': False,  # PROBLEM: Detects text, logos, stickers
            'config': {
                'min_area_ratio': 0.15,  # Increased to filter small features
                'morph_kernel_size': (5, 5),
                'block_size': 11,
                'c_constant': 2,
                'method': 'gaussian',
                'use_blur': True,
                'blur_kernel': (5, 5)
            }
        },
        {
            'type': 'otsu_threshold',
            'enabled': False,  # PROBLEM: Detects patterns, textures
            'config': {
                'min_area_ratio': 0.15,  # Increased to filter small features
                'morph_kernel_size': (5, 5),
                'use_blur': True,
                'blur_kernel': (5, 5)
            }
        },

    ]

    @classmethod
    def get_calibration_detectors(cls):
        """
        Get list of enabled calibration detectors

        Returns:
            List of detector instances in priority order
        """
        from src.detectors.calibration import (
            QuadrilateralDetector,
            RembgCalibrationDetector,
            OtsuCalibrationDetector,
            AdaptiveCalibrationDetector
        )

        detectors = []
        for method in cls.CALIBRATION_METHODS:
            if method['enabled']:
                if method['type'] == 'quadrilateral':
                    detectors.append(QuadrilateralDetector(**method['config']))
                elif method['type'] == 'rembg_calibration':
                    detectors.append(RembgCalibrationDetector(**method['config']))
                elif method['type'] == 'otsu_calibration':
                    detectors.append(OtsuCalibrationDetector(**method['config']))
                elif method['type'] == 'adaptive_calibration':
                    detectors.append(AdaptiveCalibrationDetector(**method['config']))
                # Add more detector types here as they are implemented

        return detectors

    @classmethod
    def get_measurement_detectors(cls):
        """
        Get list of enabled measurement detectors

        Returns:
            List of detector instances in priority order
        """
        from src.detectors.measurement import (
            YOLODetector,
            RembgDetector,
            EdgeDetector,
            OtsuDetector,
            AdaptiveDetector
        )

        detectors = []
        for method in cls.MEASUREMENT_METHODS:
            if method['enabled']:
                if method['type'] == 'yolo':
                    detectors.append(YOLODetector(**method['config']))
                elif method['type'] == 'rembg':
                    detectors.append(RembgDetector(**method['config']))
                elif method['type'] == 'edge_detection':
                    detectors.append(EdgeDetector(**method['config']))
                elif method['type'] == 'otsu_threshold':
                    detectors.append(OtsuDetector(**method['config']))
                elif method['type'] == 'adaptive_threshold':
                    detectors.append(AdaptiveDetector(**method['config']))
                # Add more detector types here as they are implemented

        return detectors

    @classmethod
    def apply_env_overrides(cls):
        """
        Override configuration from environment variables

        Environment variables:
            CALIBRATION_METHODS: Comma-separated list of enabled methods
            MEASUREMENT_METHODS: Comma-separated list of enabled methods

        Example:
            MEASUREMENT_METHODS=yolo,edge_detection  # Skip rembg
            CALIBRATION_METHODS=quadrilateral
        """
        # Override calibration methods
        calib_methods_env = os.getenv('CALIBRATION_METHODS', None)
        if calib_methods_env:
            enabled_methods = [m.strip() for m in calib_methods_env.split(',')]
            for method in cls.CALIBRATION_METHODS:
                method['enabled'] = method['type'] in enabled_methods

        # Override measurement methods
        meas_methods_env = os.getenv('MEASUREMENT_METHODS', None)
        if meas_methods_env:
            enabled_methods = [m.strip() for m in meas_methods_env.split(',')]
            for method in cls.MEASUREMENT_METHODS:
                method['enabled'] = method['type'] in enabled_methods


# Apply environment overrides on module load
DetectionConfig.apply_env_overrides()

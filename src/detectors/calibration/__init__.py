"""
Calibration detectors

Contains detector implementations for credit card calibration.
"""

from .quadrilateral_detector import QuadrilateralDetector
from .rembg_calibration_detector import RembgCalibrationDetector
from .otsu_calibration_detector import OtsuCalibrationDetector
from .adaptive_calibration_detector import AdaptiveCalibrationDetector

__all__ = ['QuadrilateralDetector', 'RembgCalibrationDetector', 'OtsuCalibrationDetector', 'AdaptiveCalibrationDetector']

"""
Base detector interface for all detection methods

This module provides the base interface that all detectors must implement,
along with a standardized result format.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import numpy as np


@dataclass
class DetectionResult:
    """
    Standardized detection result returned by all detectors

    Attributes:
        success: Whether detection was successful
        contour: Detected contour (numpy array) or None
        rect: Minimum area bounding rectangle tuple: (center, (width, height), angle)
        confidence: Confidence score (0.0 to 1.0)
        method_name: Name of the detection method used
        metadata: Additional method-specific data
        error: Error message if detection failed
    """
    success: bool
    contour: Optional[np.ndarray]
    rect: Optional[tuple]
    confidence: float
    method_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseDetector(ABC):
    """
    Abstract base class for all detectors

    All detector implementations must inherit from this class and implement
    the detect() and is_available() methods.
    """

    def __init__(self, name: str, **config):
        """
        Initialize detector

        Args:
            name: Human-readable name for this detector
            **config: Configuration parameters specific to this detector
        """
        self.name = name
        self.config = config

    @abstractmethod
    def detect(self, image: np.ndarray, debug_folder: str) -> DetectionResult:
        """
        Detect object in image

        Args:
            image: BGR input image (numpy array)
            debug_folder: Path to save debug images

        Returns:
            DetectionResult with success status and detection data
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if detector dependencies are available

        Returns:
            True if all required dependencies are installed, False otherwise
        """
        pass

    def get_description(self) -> str:
        """
        Get human-readable description of this detector

        Returns:
            Description string
        """
        return self.name

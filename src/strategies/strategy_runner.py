"""
Detection Strategy Runner

Runs multiple detectors in priority order until one succeeds.
This implements the Chain of Responsibility pattern for detection methods.
"""

from typing import List
import logging

from src.detectors.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class DetectionStrategyRunner:
    """
    Runs multiple detectors in priority order until one succeeds

    This class takes a list of detectors and tries them sequentially.
    The first detector that succeeds is used, and its result is returned.
    If all detectors fail, raises ValueError.
    """

    def __init__(self, detectors: List[BaseDetector]):
        """
        Initialize strategy runner

        Args:
            detectors: List of detectors to try, in priority order
        """
        self.detectors = detectors
        self.attempts = []

    def run(self, image, debug_folder: str) -> DetectionResult:
        """
        Try detectors in order until one succeeds

        Args:
            image: Input image (numpy array)
            debug_folder: Path for debug images

        Returns:
            DetectionResult from first successful detector

        Raises:
            ValueError: If all detectors fail
        """
        logger.info("=" * 70)
        logger.info(f"STRATEGY RUNNER: Starting with {len(self.detectors)} detector(s)")
        logger.info("=" * 70)

        self.attempts = []

        for i, detector in enumerate(self.detectors):
            detector_num = f"[{i+1}/{len(self.detectors)}]"

            # Check if detector is available
            if not detector.is_available():
                logger.info(f"{detector_num} {detector.name}: SKIPPED (dependencies not available)")
                self.attempts.append({
                    'detector': detector.name,
                    'status': 'unavailable',
                    'error': 'Dependencies not installed'
                })
                continue

            # Try detector
            logger.info(f"{detector_num} {detector.name}: ATTEMPTING...")
            try:
                result = detector.detect(image, debug_folder)

                if result.success:
                    logger.info(f"{detector_num} {detector.name}: SUCCESS! (confidence: {result.confidence:.3f})")
                    self.attempts.append({
                        'detector': detector.name,
                        'status': 'success',
                        'confidence': result.confidence
                    })
                    return result
                else:
                    logger.warning(f"{detector_num} {detector.name}: FAILED - {result.error}")
                    self.attempts.append({
                        'detector': detector.name,
                        'status': 'failed',
                        'error': result.error
                    })

            except Exception as e:
                logger.error(f"{detector_num} {detector.name}: ERROR - {str(e)}")
                self.attempts.append({
                    'detector': detector.name,
                    'status': 'error',
                    'error': str(e)
                })

        # All detectors failed
        logger.error("=" * 70)
        logger.error(f"STRATEGY RUNNER: ALL {len(self.detectors)} DETECTOR(S) FAILED")
        logger.error("Attempts summary:")
        for attempt in self.attempts:
            logger.error(f"  - {attempt['detector']}: {attempt['status']}")
        logger.error("=" * 70)

        raise ValueError(f"All {len(self.detectors)} detection method(s) failed")

    def get_attempts_summary(self) -> List[dict]:
        """
        Get summary of all detection attempts

        Returns:
            List of attempt dictionaries with status info
        """
        return self.attempts

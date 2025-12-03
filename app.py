from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# Import REFACTORED detection modules (uses strategy runner with configurable methods)
# Original files (final_calibration.py, measurement.py) are preserved for reference
from src.calibration import (
    detect_credit_card,
    calculate_calibration,
    save_debug_image
)
from src.measurement import analyze_package
from src.utils import load_image_from_bytes, resize_if_needed

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/microservice.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS
cors_origins = os.getenv('CORS_ORIGINS', '').split(',')
CORS(app, origins=cors_origins)

# Configure upload limits
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 10485760))

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'service': 'package-dimension-scanner'
    })

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    """
    Calibrate camera using credit card reference
    Expects: image (multipart/form-data)
    Returns: { pixels_per_millimeter, confidence, reference_type }
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"[{timestamp}] Calibration request received")

    try:
        # Validate image is present
        if 'image' not in request.files:
            logger.warning(f"[{timestamp}] No image provided in request")
            return jsonify({
                'success': False,
                'error': {
                    'code': 'VALIDATION_ERROR',
                    'message': 'No image provided',
                    'details': 'Please provide an image file with key "image"'
                }
            }), 400

        # Read image file
        image_file = request.files['image']
        original_filename = image_file.filename
        logger.info(f"[{timestamp}] Processing image: {original_filename}")

        image_bytes = image_file.read()
        logger.info(f"[{timestamp}] Image size: {len(image_bytes)} bytes")

        # Load and preprocess image
        logger.info(f"[{timestamp}] Loading image from bytes...")
        image = load_image_from_bytes(image_bytes)
        logger.info(f"[{timestamp}] Original image shape: {image.shape}")

        image = resize_if_needed(image, max_width=1920, max_height=1080)
        logger.info(f"[{timestamp}] Resized image shape: {image.shape}")

        # Detect credit card using QUADRILATERAL method
        logger.info(f"[{timestamp}] Detecting credit card with quadrilateral detection...")
        contour, rect, detection_method, detection_confidence = detect_credit_card(image)

        if contour is not None:
            logger.info(f"[{timestamp}] Credit card detected! Method: {detection_method}, Confidence: {detection_confidence:.3f}")
        else:
            logger.warning(f"[{timestamp}] Credit card NOT detected")

        # Calculate calibration
        logger.info(f"[{timestamp}] Calculating calibration...")
        result = calculate_calibration(contour, rect, image)

        # Add detection method info to result
        result['detectionMethod'] = detection_method
        result['detectionConfidence'] = float(detection_confidence)

        logger.info(f"[{timestamp}] Calibration successful - PPM: {result['pixelsPerMillimeter']:.2f}, Confidence: {result['confidence']:.2f}, Method: {detection_method}")

        # Save debug image with annotations
        debug_filename = f"calibration_{timestamp}.jpg"
        logger.info(f"[{timestamp}] Saving debug image: {debug_filename}")
        save_debug_image(image, contour, rect, debug_filename, result)

        return jsonify({
            'success': True,
            **result
        })

    except ValueError as e:
        # Reference not found error
        error_code = str(e)
        logger.error(f"[{timestamp}] Calibration failed: {error_code}")

        # Save failed image for debugging
        try:
            debug_filename = f"calibration_failed_{timestamp}.jpg"
            save_debug_image(image, None, None, debug_filename, None)
            logger.info(f"[{timestamp}] Saved failed image: {debug_filename}")
        except Exception as debug_error:
            logger.warning(f"[{timestamp}] Could not save debug image: {debug_error}")
            pass

        return jsonify({
            'success': False,
            'error': {
                'code': error_code,
                'message': 'No credit card detected in image',
                'details': 'Ensure credit card is flat and fully visible in frame'
            }
        }), 400

    except Exception as e:
        # Internal error
        logger.error(f"[{timestamp}] Internal error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/analyze-package', methods=['POST'])
def analyze():
    """
    Analyze package dimensions from two views
    Expects: image1 (top view), image2 (side view), pixels_per_millimeter, units (optional)
    Returns: { dimensions, confidence, measurementMethod, processingTimeMs }
    """
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"[{timestamp}] Package analysis request received")

    try:
        # ----------------------------
        # Validate inputs
        # ----------------------------
        if 'image1' not in request.files or 'image2' not in request.files:
            logger.warning(f"[{timestamp}] Missing images in request")
            return jsonify({
                'success': False,
                'error': {
                    'code': 'VALIDATION_ERROR',
                    'message': 'Two images required (image1, image2)',
                    'details': 'Please provide both top view and side view images'
                }
            }), 400

        # Get pixels_per_millimeter from calibration
        pixels_per_mm = request.form.get('pixels_per_millimeter')
        if not pixels_per_mm:
            logger.warning(f"[{timestamp}] No pixels_per_millimeter provided")
            return jsonify({
                'success': False,
                'error': {
                    'code': 'VALIDATION_ERROR',
                    'message': 'Valid pixels_per_millimeter required',
                    'details': 'Please calibrate camera first'
                }
            }), 400

        try:
            pixels_per_mm = float(pixels_per_mm)
            if pixels_per_mm <= 0:
                raise ValueError("Must be positive")
        except (ValueError, TypeError) as e:
            logger.warning(f"[{timestamp}] Invalid pixels_per_millimeter: {pixels_per_mm}")
            return jsonify({
                'success': False,
                'error': {
                    'code': 'VALIDATION_ERROR',
                    'message': f'Invalid pixels_per_millimeter: {pixels_per_mm}',
                    'details': 'Must be a positive number from calibration'
                }
            }), 400

        # Get target units (default to inches)
        units = request.form.get('units', 'inches')
        logger.info(f"[{timestamp}] Processing with calibration: {pixels_per_mm:.3f} px/mm, units: {units}")

        # ----------------------------
        # Load images
        # ----------------------------
        image1_file = request.files['image1']
        image2_file = request.files['image2']

        logger.info(f"[{timestamp}] Loading image1 (top view): {image1_file.filename}")
        image1_bytes = image1_file.read()
        logger.info(f"[{timestamp}] Image1 size: {len(image1_bytes)} bytes")

        logger.info(f"[{timestamp}] Loading image2 (side view): {image2_file.filename}")
        image2_bytes = image2_file.read()
        logger.info(f"[{timestamp}] Image2 size: {len(image2_bytes)} bytes")

        # Load and preprocess images
        logger.info(f"[{timestamp}] Loading images from bytes...")
        image1 = load_image_from_bytes(image1_bytes)
        image2 = load_image_from_bytes(image2_bytes)

        logger.info(f"[{timestamp}] Image1 shape: {image1.shape}")
        logger.info(f"[{timestamp}] Image2 shape: {image2.shape}")

        # Resize if needed
        image1 = resize_if_needed(image1, max_width=1920, max_height=1080)
        image2 = resize_if_needed(image2, max_width=1920, max_height=1080)

        logger.info(f"[{timestamp}] Resized image1: {image1.shape}")
        logger.info(f"[{timestamp}] Resized image2: {image2.shape}")

        # ----------------------------
        # Analyze package dimensions
        # ----------------------------
        logger.info(f"[{timestamp}] Starting package analysis...")
        result = analyze_package(image1, image2, pixels_per_mm, units)

        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"[{timestamp}] Analysis complete in {processing_time}ms")

        # Log final results
        dims = result['dimensions']
        logger.info(f"[{timestamp}] RESULTS: L={dims['length']:.2f}, W={dims['width']:.2f}, H={dims['height']:.2f} {dims['units']}")
        logger.info(f"[{timestamp}] Confidence: {result['confidence']:.2f}")

        return jsonify({
            'success': True,
            **result,
            'processingTimeMs': processing_time
        })

    except ValueError as e:
        # Package not detected error
        error_code = str(e)
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"[{timestamp}] Package detection failed: {error_code} (after {processing_time}ms)")

        return jsonify({
            'success': False,
            'error': {
                'code': error_code,
                'message': 'Package not detected in images',
                'details': 'Ensure package is clearly visible with contrasting background'
            },
            'processingTimeMs': processing_time
        }), 400

    except Exception as e:
        # Internal error
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"[{timestamp}] Internal error: {str(e)} (after {processing_time}ms)", exc_info=True)

        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            },
            'processingTimeMs': processing_time
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)

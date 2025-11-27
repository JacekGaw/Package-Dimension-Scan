import cv2
import numpy as np
from PIL import Image
import io

def load_image_from_bytes(image_bytes):
    """Load image from bytes and convert to OpenCV format"""
    image = Image.open(io.BytesIO(image_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def resize_if_needed(image, max_width=1920, max_height=1080):
    """Resize image if larger than max dimensions"""
    height, width = image.shape[:2]

    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image

def preprocess_image(image):
    """Apply preprocessing to improve detection accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    return filtered

def validate_image(image_bytes, max_size_mb=10):
    """Validate image before processing"""
    # Check size
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f'Image too large: {size_mb:.2f}MB (max: {max_size_mb}MB)')

    # Try to load image
    try:
        load_image_from_bytes(image_bytes)
    except Exception as e:
        raise ValueError(f'Invalid image format: {str(e)}')

    return True

# Configuration Examples

Quick reference for common configuration patterns in `config/detection_config.py`

## Default Configuration (Current)

```python
CALIBRATION_METHODS = [
    {'type': 'quadrilateral', 'enabled': True, 'config': {...}}
]

MEASUREMENT_METHODS = [
    {'type': 'yolo', 'enabled': True, 'config': {...}},
    {'type': 'rembg', 'enabled': True, 'config': {...}},
    {'type': 'edge_detection', 'enabled': True, 'config': {...}}
]
```

**Tries:** YOLO → rembg → edge detection

---

## Example 1: Speed First (Fast Detection)

Prioritize fastest methods:

```python
MEASUREMENT_METHODS = [
    {'type': 'yolo', 'enabled': True, 'config': {...}},           # Fastest (50-200ms)
    {'type': 'edge_detection', 'enabled': True, 'config': {...}}, # Fast fallback
    {'type': 'rembg', 'enabled': False, 'config': {...}}          # Skip slow AI method
]
```

**Tries:** YOLO → edge detection
**Best for:** Real-time processing, low-latency requirements

---

## Example 2: Accuracy First (Best Quality)

Prioritize most accurate methods:

```python
MEASUREMENT_METHODS = [
    {'type': 'rembg', 'enabled': True, 'config': {...}},          # Most accurate (90-95%)
    {'type': 'yolo', 'enabled': True, 'config': {...}},           # Fast backup
    {'type': 'edge_detection', 'enabled': True, 'config': {...}}  # Last resort
]
```

**Tries:** rembg → YOLO → edge detection
**Best for:** High accuracy requirements, offline processing

---

## Example 3: YOLO Only (Known Objects)

Use only YOLO for known object types:

```python
MEASUREMENT_METHODS = [
    {'type': 'yolo', 'enabled': True, 'config': {...}},
    {'type': 'rembg', 'enabled': False, 'config': {...}},
    {'type': 'edge_detection', 'enabled': False, 'config': {...}}
]
```

**Tries:** YOLO only
**Best for:** Boxes, bottles, books, and other COCO dataset objects

---

## Example 4: No AI Dependencies

Use only traditional computer vision (no YOLO, no rembg):

```python
MEASUREMENT_METHODS = [
    {'type': 'yolo', 'enabled': False, 'config': {...}},
    {'type': 'rembg', 'enabled': False, 'config': {...}},
    {'type': 'edge_detection', 'enabled': True, 'config': {...}}
]
```

**Tries:** Edge detection only
**Best for:** Minimal dependencies, simple setups

---

## Example 5: AI Only (Skip Traditional CV)

Use only AI methods:

```python
MEASUREMENT_METHODS = [
    {'type': 'yolo', 'enabled': True, 'config': {...}},
    {'type': 'rembg', 'enabled': True, 'config': {...}},
    {'type': 'edge_detection', 'enabled': False, 'config': {...}}
]
```

**Tries:** YOLO → rembg (no edge detection fallback)
**Best for:** High-quality images, good lighting conditions

---

## Example 6: Tuned for Poor Lighting

Adjust parameters for difficult lighting:

```python
MEASUREMENT_METHODS = [
    {
        'type': 'edge_detection',
        'enabled': True,
        'config': {
            'clahe_clip_limit': 3.0,      # Increased from 2.0
            'clahe_tile_size': (12, 12),  # Larger tiles
            'gaussian_kernel': (7, 7),    # More blur
            'canny_low': 30,              # Lower threshold
            'canny_high': 120,            # Lower threshold
            'min_area_ratio': 0.005       # Allow smaller detections
        }
    },
    {'type': 'rembg', 'enabled': True, 'config': {...}},  # AI backup
    {'type': 'yolo', 'enabled': False, 'config': {...}}   # Skip in poor lighting
]
```

**Best for:** Dim lighting, shadowy environments

---

## Example 7: Tuned for High Contrast

Optimize for good lighting and high contrast:

```python
MEASUREMENT_METHODS = [
    {
        'type': 'edge_detection',
        'enabled': True,
        'config': {
            'clahe_clip_limit': 1.5,      # Decreased (already good contrast)
            'canny_low': 60,              # Higher threshold
            'canny_high': 180,            # Higher threshold
            'min_area_ratio': 0.02        # Stricter filtering
        }
    },
    {'type': 'yolo', 'enabled': True, 'config': {...}},
    {'type': 'rembg', 'enabled': False, 'config': {...}}  # Not needed
]
```

**Best for:** Studio lighting, white background, professional photos

---

## Example 8: Small Objects

Detect smaller objects:

```python
MEASUREMENT_METHODS = [
    {
        'type': 'yolo',
        'enabled': True,
        'config': {
            'model': 'yolov8n-seg.pt',
            'confidence_threshold': 0.3,   # Lower threshold
            'min_area_ratio': 0.005        # Allow smaller (0.5% of image)
        }
    },
    {
        'type': 'rembg',
        'enabled': True,
        'config': {
            'min_area_ratio': 0.005        # Allow smaller
        }
    },
    {
        'type': 'edge_detection',
        'enabled': True,
        'config': {
            'min_area_ratio': 0.005        # Allow smaller
        }
    }
]
```

**Best for:** Jewelry, small packages, compact items

---

## Example 9: Large Objects Only

Detect only large objects:

```python
MEASUREMENT_METHODS = [
    {
        'type': 'yolo',
        'enabled': True,
        'config': {
            'min_area_ratio': 0.1          # At least 10% of image
        }
    },
    {
        'type': 'edge_detection',
        'enabled': True,
        'config': {
            'min_area_ratio': 0.1          # At least 10% of image
        }
    }
]
```

**Best for:** Large packages, pallets, furniture

---

## Example 10: Development/Debug Mode

Use only fast method with verbose logging:

```python
MEASUREMENT_METHODS = [
    {
        'type': 'edge_detection',
        'enabled': True,
        'config': {
            'clahe_clip_limit': 2.0,
            'min_area_ratio': 0.01
        }
    },
    {'type': 'yolo', 'enabled': False, 'config': {...}},
    {'type': 'rembg', 'enabled': False, 'config': {...}}
]
```

**Best for:** Testing, debugging, development

---

## Environment Variable Examples

### Production (All Methods)
```bash
# Use all methods for best reliability
MEASUREMENT_METHODS=yolo,rembg,edge_detection
```

### Staging (AI Only)
```bash
# Test AI methods without fallback
MEASUREMENT_METHODS=yolo,rembg
```

### Development (Fast Only)
```bash
# Fast iteration during development
MEASUREMENT_METHODS=edge_detection
```

### High Accuracy Mode
```bash
# Accuracy first
MEASUREMENT_METHODS=rembg,yolo,edge_detection
```

---

## Parameter Tuning Guide

### CLAHE (Contrast Enhancement)
- **clip_limit**: 1.0-4.0 (higher = more enhancement)
  - Low lighting: 3.0-4.0
  - Good lighting: 1.5-2.0
- **tile_size**: (4,4) to (16,16)
  - Small: More local enhancement
  - Large: More global enhancement

### Canny Edge Detection
- **canny_low**: 20-80 (lower = more edges)
- **canny_high**: 80-200 (higher = fewer edges)
- Poor lighting: Lower thresholds (30, 120)
- Good lighting: Higher thresholds (60, 180)

### Area Filtering
- **min_area_ratio**: 0.001-0.2
  - Small objects: 0.005 (0.5%)
  - Medium objects: 0.01 (1%)
  - Large objects: 0.05 (5%)

### YOLO Confidence
- **confidence_threshold**: 0.1-0.9
  - Strict: 0.7-0.9
  - Balanced: 0.4-0.6
  - Permissive: 0.1-0.3

---

## Quick Comparison

| Configuration | Speed | Accuracy | Use Case |
|--------------|-------|----------|----------|
| YOLO only | ⚡⚡⚡ | ⭐⭐⭐ | Known objects, speed critical |
| rembg only | ⚡ | ⭐⭐⭐⭐⭐ | Any object, accuracy critical |
| Edge only | ⚡⚡ | ⭐⭐ | Simple shapes, no dependencies |
| YOLO → rembg → Edge | ⚡⚡ | ⭐⭐⭐⭐ | Best reliability (default) |
| rembg → YOLO → Edge | ⚡ | ⭐⭐⭐⭐⭐ | Best accuracy |
| YOLO → Edge | ⚡⚡⚡ | ⭐⭐⭐ | Fast with fallback |

---

## Testing Your Configuration

After changing configuration, test with:

```bash
cd microservice
venv\Scripts\python app.py
```

Watch logs to see which methods succeed:
```
[1/3] yolo: ATTEMPTING...
[1/3] yolo: SUCCESS! (confidence: 0.850)
```

Or:
```
[1/3] yolo: FAILED - No objects detected
[2/3] rembg: ATTEMPTING...
[2/3] rembg: SUCCESS! (confidence: 0.780)
```

---

## Recommended Starting Points

**If you're unsure, use these:**

### For Most Use Cases (Balanced)
```python
MEASUREMENT_METHODS = [
    {'type': 'yolo', 'enabled': True, 'config': {
        'confidence_threshold': 0.5,
        'min_area_ratio': 0.01
    }},
    {'type': 'rembg', 'enabled': True, 'config': {
        'min_area_ratio': 0.01
    }},
    {'type': 'edge_detection', 'enabled': True, 'config': {
        'clahe_clip_limit': 2.0,
        'min_area_ratio': 0.01
    }}
]
```

### For Speed-Critical Applications
```python
MEASUREMENT_METHODS = [
    {'type': 'yolo', 'enabled': True, 'config': {
        'confidence_threshold': 0.4,  # More permissive
        'min_area_ratio': 0.01
    }},
    {'type': 'edge_detection', 'enabled': True, 'config': {...}},
    {'type': 'rembg', 'enabled': False, 'config': {...}}
]
```

### For Accuracy-Critical Applications
```python
MEASUREMENT_METHODS = [
    {'type': 'rembg', 'enabled': True, 'config': {
        'min_area_ratio': 0.01
    }},
    {'type': 'yolo', 'enabled': True, 'config': {
        'confidence_threshold': 0.6,  # More strict
        'min_area_ratio': 0.01
    }},
    {'type': 'edge_detection', 'enabled': False, 'config': {...}}
]
```

---

**Remember:** You can always change the configuration and restart the service. No code changes required!

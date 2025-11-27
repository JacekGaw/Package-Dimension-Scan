# Package Dimension Scanner Microservice

Python Flask microservice for detecting package dimensions using computer vision with OpenCV.

## Features

- Credit card calibration for scale detection
- Package dimension measurement from two views (top and side)
- Simple cross-validation algorithm
- RESTful API endpoints

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

Copy `.env` file and adjust settings as needed. Key settings:
- `PORT`: Server port (default: 5000)
- `CORS_ORIGINS`: Allowed CORS origins
- `MAX_CONTENT_LENGTH`: Max upload size in bytes

### Running

```bash
python app.py
```

Server will start at `http://localhost:5000`

### Health Check

```bash
curl http://localhost:5000/api/health
```

## API Endpoints

### GET /api/health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "service": "package-dimension-scanner"
}
```

### POST /api/calibrate (Phase 2)
Calibrate camera using credit card image

### POST /api/analyze-package (Phase 3)
Analyze package dimensions from two images

## Development

### Project Structure

```
microservice/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── Dockerfile             # Docker configuration
├── src/
│   ├── __init__.py
│   ├── calibration.py     # Credit card detection (Phase 2)
│   ├── measurement.py     # Package detection (Phase 3)
│   └── utils.py           # Image preprocessing
└── tests/
    ├── __init__.py
    └── test_api.py        # API tests
```

### Testing

```bash
# Run tests (when implemented)
pytest tests/
```

## Docker

Build and run with Docker:

```bash
# Build image
docker build -t package-dimension-scanner .

# Run container
docker run -p 5000:5000 --env-file .env package-dimension-scanner
```

## License

Proprietary - ShipStation

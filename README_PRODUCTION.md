# Media File Converter - Production Setup

This Flask application has been configured for production deployment using Gunicorn as the WSGI server.

## Production Deployment

### Using Docker (Recommended)

```bash
# Build the container
docker build -t media-file-converter .

# Run the container
docker run -p 5000:5000 media-file-converter
```

### Using Docker Compose

```bash
docker-compose up --build
```

### Manual Production Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run with Gunicorn:
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 300 app:app
```

## Development Mode

For development only, you can run the Flask development server:

```bash
export FLASK_DEBUG=true
python app.py
```

Or use the provided script:
```bash
chmod +x run_dev.sh
./run_dev.sh
```

## Production vs Development

- **Production**: Uses Gunicorn WSGI server with 4 workers and 300s timeout
- **Development**: Uses Flask's built-in development server (only when FLASK_DEBUG=true)

## Key Changes Made

1. **Requirements**: Cleaned up `requirements.txt` with pinned versions and added Gunicorn
2. **Application**: Modified `app.py` to prevent running development server in production
3. **Docker**: Updated `Dockerfile` to use Gunicorn and run as non-root user
4. **Security**: Added non-root user in Docker container

## Environment Variables

- `FLASK_DEBUG=true`: Enables development mode (NOT for production)

## Endpoints

The application provides the following endpoints:
- `/convert` - Convert audio files to MP3
- `/track-meta` - Get audio track metadata and peaks
- `/convert-image` - Convert and resize images
- `/add_watermark` - Add watermark to images
- `/transform-reference` - Transform images to reference dimensions
- `/extract-first-frame` - Extract first frame from video
- `/extract-last-frame` - Extract last frame from video
- `/concat` - Concatenate two videos
- `/analyze-video` - Analyze video file properties

## Note

If you see the warning "This is a development server. Do not use it in a production deployment", it means you're running the Flask development server. Use the Docker setup or Gunicorn command above for production.

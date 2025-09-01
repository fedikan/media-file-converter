FROM python:3.9-slim

# Install FFmpeg and other system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY app.py /app/
<<<<<<< HEAD
COPY ropewalk-watermark.png /app/
=======
# Note: Add watermark file (ropewalk-watermark.png) if needed for watermark functionality

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app
>>>>>>> cb4457a4415aa03f39fa351aef418b503af381d6

# Expose the port the app runs on
EXPOSE 5000

<<<<<<< HEAD
# Set the entrypoint to the Flask application
ENTRYPOINT ["python", "/app/app.py"]
=======
# Use Gunicorn as the WSGI server for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "300", "app:app"]
>>>>>>> cb4457a4415aa03f39fa351aef418b503af381d6

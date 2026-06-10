FROM python:3.11-slim
# Bumped 3.9 → 3.11 for markitdown (`/markitdown` endpoint, replaces the prior
# `/docx-to-text` + `/pdf-to-text` pair). Markitdown requires Python ≥3.10
# and pulls cleaner-than-LibreOffice text for DOCX/PPTX/XLSX plus pdfminer-
# based PDF parsing. Loss of LibreOffice drops ~250MB from the image.

# System deps: ffmpeg for media endpoints, poppler-utils for PDF rasterization
# (`/pdf-to-pages` shells out via pdf2image). No LibreOffice — markitdown
# handles Office formats in-process.
RUN apt-get update && \
    apt-get install -y ffmpeg \
        libavcodec-extra \
        libavformat-dev \
        libavfilter-dev \
        libavdevice-dev \
        libswscale-dev \
        libswresample-dev \
        libpostproc-dev \
        poppler-utils && \
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
COPY ssrf_guard.py /app/
COPY ropewalk-watermark.png /app/
COPY og/ /app/og/
# Note: Add watermark file (ropewalk-watermark.png) if needed for watermark functionality

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose the port the app runs on
EXPOSE 5000

# Set environment variable to indicate we're running in Docker
ENV DOCKER_CONTAINER=true

# Run the application with gunicorn in production
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "600", \
     "--max-requests", "100", \
     "--max-requests-jitter", "20", \
     "--worker-class", "sync", \
     "--log-level", "info", \
     "app:app"]




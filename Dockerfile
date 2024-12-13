FROM python:3.9-slim

# Install FFmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
# Install Python dependencies
RUN pip install Flask opencv-python ffmpeg-python numpy wavfile scipy wave Pillow pillow_heif

# Set the working directory
WORKDIR /app

# Copy the Python script into the container
COPY app.py /app/
COPY aiphoria-watermark.png /app/

# Expose the port the app runs on
EXPOSE 5000

# Set the entrypoint to the Flask application
ENTRYPOINT ["python", "/app/app.py"]
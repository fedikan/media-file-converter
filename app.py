from flask import Flask, request, jsonify, send_file
import ffmpeg
import os
import uuid
from scipy.io import wavfile
from scipy.signal import find_peaks
import numpy as np
import io
import wave
from PIL import Image
from pillow_heif import register_heif_opener
import cv2
import tempfile
import shutil
import time
import requests
register_heif_opener()

app = Flask(__name__)

# Configuration for temp directory
CONFIG = {
    'temp_dir': '/tmp'
}

def calculate_peaks(channel_data, sample_rate, num_peaks=300):
    # Calculate peaks for visualization in WaveSurfer
    # num_peaks is the fixed number of peaks we want to calculate
    window_size = len(channel_data) // num_peaks
    peaks = []
    for i in range(num_peaks):
        window_start = i * window_size
        window_end = window_start + window_size
        window = channel_data[window_start:window_end]
        if len(window) == 0:
            break
        peak = np.max(np.abs(window)) / 32767.0  # Normalize to range [-1, 1]
        peaks.append(peak)
        
    return peaks

@app.route('/add_watermark', methods=['POST'])
def add_watermark():
    if 'file' not in request.files:
        return 'No file part', 400

    image_file = request.files['file']
    main_image = Image.open(image_file).convert("RGBA")
    
    # Crop to square if image is vertical
    width, height = main_image.size
    if height > width:
        # Calculate the crop box for center square
        crop_size = width
        top = (height - width) // 2
        bottom = top + crop_size
        crop_box = (0, top, width, bottom)
        main_image = main_image.crop(crop_box)

    watermark = Image.open('ropewalk-watermark.png').convert("RGBA")

    # Calculate new watermark size (15% of image width)
    new_height = int(main_image.width * 0.15)
    aspect_ratio = watermark.width / watermark.height
    new_width = int(new_height * aspect_ratio)

    watermark = watermark.resize((new_width, new_height), Image.LANCZOS)

    position = (0, main_image.height - watermark.height)

    watermarked = Image.new('RGBA', main_image.size)

    watermarked.paste(main_image, (0, 0))

    watermarked.paste(watermark, position, mask=watermark)

    watermarked = watermarked.convert("RGB")

    img_io = io.BytesIO()
    watermarked.save(img_io, 'WEBP', quality=95)  # Adjust quality as needed

    img_io.seek(0)

    return send_file(
        img_io,
        mimetype='image/webp',
        as_attachment=True,
        download_name='watermarked_image.webp'
    )

@app.route('/convert', methods=['POST'])
def convert_audio():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        # Save the original file
        original_filename = str(uuid.uuid4())
        input_path = f'/tmp/{original_filename}'
        file.save(input_path)

        # Set the output path
        output_filename = f'{original_filename}.mp3'
        output_path = f'/tmp/{output_filename}'

        # Convert the file
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, audio_bitrate='320k')
        ffmpeg.run(stream)

        # Send the converted file
        return send_file(output_path, as_attachment=True)

@app.route('/track-meta', methods=['POST'])
def get_track_meta():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        # Save the uploaded file
        original_filename = str(uuid.uuid4())
        input_path = f'/tmp/{original_filename}'
        output_wav_path = f'{input_path}.wav'
        file.save(input_path)

        # Convert the file to WAV format for processing
        ffmpeg.input(input_path).output(output_wav_path).run()

        # Read the WAV file data
        with wave.open(output_wav_path, 'r') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()  # Get the sample rate from the file
            num_frames = wav_file.getnframes()  # Get the number of frames from the file
            duration = num_frames / float(sample_rate)  # Calculate the duration of the track

            # Convert frames to numpy array
            frame_data = np.frombuffer(frames, dtype=np.int16)
            if channels == 1:
                # Duplicate the mono channel data for stereo processing
                left_channel = right_channel = frame_data
            elif channels == 2:
                # Split stereo into left and right channels
                left_channel = frame_data[::2]
                right_channel = frame_data[1::2]
            else:
                return 'Audio file has more than 2 channels', 400

        # Find peaks for WaveSurfer
        left_peaks = calculate_peaks(left_channel, sample_rate)
        right_peaks = calculate_peaks(right_channel, sample_rate)

        # Clean up the temporary files
        os.remove(input_path)
        os.remove(output_wav_path)

        return jsonify({'left_peaks': left_peaks, 'right_peaks': right_peaks, 'duration': duration})

@app.route('/convert-image', methods=['POST'])
def convert_image():
    # Check if the post request has the file part and required parameters
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Check for output format in the request
    output_format = request.form.get('outputFormat', 'webp').lower()  # Default to webp if not specified
    try:
        desired_width = int(request.form.get('width', 0))
        desired_height = int(request.form.get('height', 0))
        watermark = request.form.get('watermark', None)  # This example does not implement watermarking
    except ValueError:
        return 'Invalid parameters', 400

    if file:
        # Open the image
        img = Image.open(file.stream)

        # If no desired dimensions are provided, skip resizing and cropping
        if desired_width > 0 and desired_height > 0:
            # Calculate the desired aspect ratio
            desired_aspect_ratio = desired_width / desired_height
            original_width, original_height = img.size
            original_aspect_ratio = original_width / original_height

            if original_aspect_ratio > desired_aspect_ratio:
                # The image is wider than the desired aspect ratio, so crop the sides
                new_height = original_height
                new_width = int(desired_aspect_ratio * new_height)
                left = (original_width - new_width) / 2
                top = 0
                right = (original_width + new_width) / 2
                bottom = original_height
            else:
                # The image is taller than the desired aspect ratio, so crop the top and bottom
                new_width = original_width
                new_height = int(new_width / desired_aspect_ratio)
                left = 0
                top = (original_height - new_height) / 2
                right = original_width
                bottom = (original_height + new_height) / 2

            img = img.crop((left, top, right, bottom))
            img = img.resize((desired_width, desired_height), Image.LANCZOS)
        # Else, proceed without resizing or cropping

        # Convert to the desired format
        img_io = io.BytesIO()
        img.save(img_io, format=output_format.upper(), quality=95)  # Adjust quality as needed
        img_io.seek(0)

        return send_file(img_io, mimetype=f'image/{output_format}')

    return 'File processing error', 500

def find_closest_dimension(original_width, original_height, allowed_dimensions):
    """
    Finds the closest dimension from the allowed dimensions based on the closest aspect ratio.
    """
    original_aspect_ratio = original_width / original_height
    closest_dimension = allowed_dimensions[0]
    smallest_diff = float('inf')
    for width, height in allowed_dimensions:
        current_diff = abs((width / height) - original_aspect_ratio)
        if current_diff < smallest_diff:
            closest_dimension = (width, height)
            smallest_diff = current_diff
    return closest_dimension

def resize_and_crop(img, target_width, target_height):
    """
    Resize and crop the PIL Image to the target dimensions.
    """
    original_width, original_height = img.size
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    if original_aspect_ratio > target_aspect_ratio:
        # The image is wider than the target aspect ratio, resize based on height
        resize_height = target_height
        resize_width = int(resize_height * original_aspect_ratio)
    else:
        # The image is taller than the target aspect ratio, resize based on width
        resize_width = target_width
        resize_height = int(resize_width / original_aspect_ratio)

    img_resized = img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)

    # Calculate cropping box
    x_center = resize_width / 2
    y_center = resize_height / 2
    x0 = int(x_center - target_width / 2)
    y0 = int(y_center - target_height / 2)
    x1 = int(x_center + target_width / 2)
    y1 = int(y_center + target_height / 2)

    img_cropped = img_resized.crop((x0, y0, x1, y1))
    return img_cropped

@app.route('/transform-reference', methods=['POST'])
def transform_image():
    if 'file' not in request.files or 'dimensions' not in request.form:
        return 'Missing file or dimensions', 400
    file = request.files['file']
    dimensions_str = request.form['dimensions']
    output_format = request.form.get('output_format', 'webp').lower()  # Default to webp if not specified
    
    if file.filename == '':
        return 'No selected file', 400

    # Parse dimensions from the requestfileType
    allowed_dimensions = parse_dimensions(dimensions_str)
    if not allowed_dimensions:
        return 'Invalid dimensions format', 400

    # Load the image
    img = Image.open(file.stream)

    # Find the closest dimension based on the aspect ratio
    closest_dimension = find_closest_dimension(*img.size, allowed_dimensions)

    # Resize and crop the image to exactly match the closest dimension
    img_transformed = resize_and_crop(img, *closest_dimension)

    output = io.BytesIO()
    img_format = img.format if img.format is not None else 'JPEG'  # Default to JPEG if format is not detected
    img_transformed.save(output, format=output_format)
    output.seek(0)

    # Send the transformed file
    return send_file(output, mimetype=f'image/{img_format.lower()}', as_attachment=True, download_name=f'transformed.{img_format.lower()}')

def parse_dimensions(dimensions_str):
    """
    Parses the dimensions from a string format (e.g., "1024x1024,1152x896") into a list of tuples.
    """
    dimensions = []
    for dim_str in dimensions_str.split(','):
        try:
            width, height = map(int, dim_str.split('x'))
            dimensions.append((width, height))
        except ValueError:
            continue  # Skip invalid formats
    return dimensions

def ensure_dirs():
    """Ensure necessary directories exist"""
    os.makedirs(CONFIG['temp_dir'], exist_ok=True)

def validate_ffmpeg():
    """Validate that ffmpeg is working properly"""
    try:
        # Test ffmpeg availability by running a simple command
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("FFmpeg validation successful")
            return True
        else:
            print(f"FFmpeg validation failed with return code: {result.returncode}")
            return False
    except FileNotFoundError:
        print("FFmpeg not found in PATH")
        return False
    except Exception as e:
        print(f"FFmpeg validation failed: {e}")
        return False

def download_remote_file_if_needed(url_or_path, temp_dir, file_type):
    """
    Download a remote file if URL is provided, otherwise return the local path.
    Returns the local path to the file.
    """
    if url_or_path.startswith(('http://', 'https://')):
        # Download remote file
        try:
            print(f"Starting download of {file_type} from: {url_or_path}")
            response = requests.get(url_or_path, stream=True, timeout=60)
            response.raise_for_status()

            # Check content length
            content_length = response.headers.get('content-length')
            if content_length:
                content_length = int(content_length)
                if content_length < 1000:
                    raise Exception(f"Remote file is too small ({content_length} bytes)")
                print(f"Expected file size: {content_length} bytes")

            # Determine file extension from content-type or URL
            content_type = response.headers.get('content-type', '')
            if 'video' in content_type:
                ext = '.mp4'
            elif 'audio' in content_type:
                ext = '.mp3'
            else:
                # Extract extension from URL
                from urllib.parse import urlparse
                parsed_url = urlparse(url_or_path)
                ext = os.path.splitext(parsed_url.path)[1] or '.mp4'

            temp_filename = f"{file_type}_{uuid.uuid4()}{ext}"
            temp_path = os.path.join(temp_dir, temp_filename)

            downloaded_size = 0
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

            # Verify download was successful
            if content_length and downloaded_size != content_length:
                raise Exception(f"Download incomplete: expected {content_length} bytes, got {downloaded_size} bytes")

            if downloaded_size < 1000:
                raise Exception(f"Downloaded file is too small ({downloaded_size} bytes)")

            print(f"Successfully downloaded {file_type}: {temp_path} ({downloaded_size} bytes)")
            return temp_path

        except requests.exceptions.Timeout:
            raise Exception(f"Download timeout for {file_type} from {url_or_path}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error downloading {file_type} from {url_or_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to download {file_type} from {url_or_path}: {str(e)}")
    else:
        # Local file path - validate it exists
        if not os.path.exists(url_or_path):
            raise Exception(f"Local {file_type} file does not exist: {url_or_path}")

        file_size = os.path.getsize(url_or_path)
        if file_size < 1000:
            raise Exception(f"Local {file_type} file is too small ({file_size} bytes): {url_or_path}")

        print(f"Using local {file_type} file: {url_or_path} ({file_size} bytes)")
        return url_or_path

def add_audio_to_video(video_path, audio_path, output_path, audio_delay_ms=0):
    """
    Merge audio with video using ffmpeg, with optional audio delay.
    """
    try:
        # Validate input files exist and are readable
        if not os.path.exists(video_path):
            raise Exception(f"Video file does not exist: {video_path}")
        if not os.path.exists(audio_path):
            raise Exception(f"Audio file does not exist: {audio_path}")

        # Check file sizes to ensure they're not empty
        video_size = os.path.getsize(video_path)
        audio_size = os.path.getsize(audio_path)

        if video_size < 100:
            raise Exception(f"Video file is too small or empty: {video_size} bytes")
        if audio_size < 100:
            raise Exception(f"Audio file is too small or empty: {audio_size} bytes")

        print(f"Merging video ({video_size} bytes) with audio ({audio_size} bytes)")

        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(audio_path)

        # Apply audio delay if specified
        if audio_delay_ms != 0:
            delay_seconds = audio_delay_ms / 1000.0
            audio_input = audio_input.filter('adelay', f"{delay_seconds}s|{delay_seconds}s")

        # Combine video and audio
        output = ffmpeg.output(
            video_input.video,
            audio_input.audio,
            output_path,
            vcodec='libx264',
            acodec='aac',
            audio_bitrate='128k',
            video_bitrate='2M',
            movflags='faststart'
        )

        # Run ffmpeg without quiet mode to see any errors
        ffmpeg.run(output, overwrite_output=True)

        # Verify the output file was created and has content
        if not os.path.exists(output_path):
            raise Exception(f"FFmpeg failed to create output file: {output_path}")

        output_size = os.path.getsize(output_path)
        if output_size < 1000:
            raise Exception(f"Output file is too small ({output_size} bytes), likely corrupted")

        print(f"Successfully created merged video: {output_path} ({output_size} bytes)")

    except ffmpeg.Error as e:
        # Handle FFmpeg specific errors
        error_msg = "FFmpeg error"
        if hasattr(e, 'stderr') and e.stderr:
            error_msg += f": {e.stderr.decode('utf-8', errors='ignore')}"
        elif str(e):
            error_msg += f": {str(e)}"
        raise Exception(f"Failed to merge audio with video: {error_msg}")
    except Exception as e:
        raise Exception(f"Failed to merge audio with video: {str(e)}")

@app.route('/extract-first-frame', methods=['POST'])
def extract_first_frame():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Save the uploaded video file temporarily
    temp_video_path = f'/tmp/{str(uuid.uuid4())}.mp4'
    file.save(temp_video_path)

    try:
        # Open the video file
        cap = cv2.VideoCapture(temp_video_path)
        
        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            return 'Could not extract frame from video', 400

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        img = Image.fromarray(frame_rgb)
        
        # Save to bytes buffer
        img_io = io.BytesIO()
        img.save(img_io, 'WEBP', quality=95)
        img_io.seek(0)

        # Clean up
        cap.release()
        os.remove(temp_video_path)

        return send_file(
            img_io,
            mimetype='image/webp',
            as_attachment=True,
            download_name='first_frame.webp'
        )

    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return f'Error processing video: {str(e)}', 500


@app.route('/merge-audio', methods=['POST'])
def merge_audio_with_video_endpoint():
    """Merge provided audio with a video and return the merged video stream."""
    temp_dir = None
    temp_output_path = None
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        video_url = data.get('videoUrl')
        audio_url = data.get('audioUrl')
        options = data.get('options', {}) or {}
        audio_delay_ms = int(options.get('audioDelayMs', 0) or 0)

        if not video_url:
            return jsonify({'error': 'videoUrl is required'}), 400
        if not audio_url:
            return jsonify({'error': 'audioUrl is required'}), 400

        ensure_dirs()

        # Validate ffmpeg availability
        if not validate_ffmpeg():
            return jsonify({'error': 'FFmpeg is not available or not working properly'}), 500

        temp_dir = tempfile.mkdtemp(dir=CONFIG['temp_dir'])

        # Resolve local paths (supports remote URLs)
        print(f"Downloading video from: {video_url}")
        local_video_path = download_remote_file_if_needed(video_url, temp_dir, 'video')
        print(f"Downloading audio from: {audio_url}")
        local_audio_path = download_remote_file_if_needed(audio_url, temp_dir, 'audio')

        if not local_video_path or not os.path.exists(local_video_path):
            return jsonify({'error': 'Video file not found after resolution'}), 404
        if not local_audio_path or not os.path.exists(local_audio_path):
            return jsonify({'error': 'Audio file not found after resolution'}), 404

        # Validate downloaded files
        try:
            video_size = os.path.getsize(local_video_path)
            audio_size = os.path.getsize(local_audio_path)
            print(f"Downloaded video size: {video_size} bytes")
            print(f"Downloaded audio size: {audio_size} bytes")

            if video_size < 1000:
                return jsonify({'error': f'Video file is too small ({video_size} bytes)'}), 400
            if audio_size < 1000:
                return jsonify({'error': f'Audio file is too small ({audio_size} bytes)'}), 400

        except OSError as e:
            return jsonify({'error': f'Error validating downloaded files: {str(e)}'}), 500

        # Prepare output filename
        timestamp = int(time.time())
        output_filename = f"merged_with_audio_{timestamp}.mp4"
        temp_output_path = os.path.join(temp_dir, output_filename)

        # Combine (replace video audio with provided audio track)
        add_audio_to_video(local_video_path, local_audio_path, temp_output_path, audio_delay_ms)

        # Double-check the output file exists and is valid (prevent race condition)
        if not os.path.exists(temp_output_path):
            return jsonify({'error': 'Failed to create merged video file'}), 500

        try:
            output_size = os.path.getsize(temp_output_path)
            if output_size < 1000:
                return jsonify({'error': f'Merged video file is too small ({output_size} bytes)'}), 500
            print(f"Merged video file size: {output_size} bytes")
        except OSError as e:
            return jsonify({'error': f'Error validating merged video file: {str(e)}'}), 500

        def generate():
            try:
                # Check file exists right before opening to prevent race condition
                if not os.path.exists(temp_output_path):
                    raise Exception("Output file was deleted before streaming could begin")

                with open(temp_output_path, 'rb') as video_file:
                    while True:
                        chunk = video_file.read(8192)
                        if not chunk:
                            break
                        yield chunk
            except Exception as stream_error:
                print(f"Error during streaming: {stream_error}")
                raise
            finally:
                try:
                    if os.path.exists(temp_output_path):
                        os.unlink(temp_output_path)
                        print(f"Cleaned up temp file: {temp_output_path}")
                except Exception as cleanup_error:
                    print(f"Failed to clean up temp file: {cleanup_error}")

        response = app.response_class(
            generate(),
            mimetype='video/mp4',
            headers={
                'Content-Disposition': f'attachment; filename="{output_filename}"',
                'Content-Type': 'video/mp4',
                'X-Filename': output_filename
            }
        )
        return response
    except Exception as e:
        print(f"Error in merge_audio_with_video_endpoint: {e}")
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.unlink(temp_output_path)
            except Exception:
                pass
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                print(f"Failed to clean up temp directory {temp_dir}: {cleanup_error}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the service is running properly"""
    try:
        # Check if ffmpeg is available
        ffmpeg_available = validate_ffmpeg()

        # Check if temp directory is writable
        temp_dir_writable = os.access(CONFIG['temp_dir'], os.W_OK)

        return jsonify({
            'status': 'healthy' if ffmpeg_available and temp_dir_writable else 'unhealthy',
            'ffmpeg_available': ffmpeg_available,
            'temp_dir_writable': temp_dir_writable,
            'temp_dir': CONFIG['temp_dir']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/extract-last-frame', methods=['POST'])
def extract_last_frame():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    temp_video_path = f'/tmp/{str(uuid.uuid4())}.mp4'
    file.save(temp_video_path)
    cap = None
    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return 'Could not open video', 400

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame = None

        # Try to jump to the last frame
        if frame_count and frame_count > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ret, frame = cap.read()
            if not ret or frame is None:
                # Fallback: step from a few frames before the end
                start = max(0, frame_count - 5)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                last = None
                while True:
                    ret, f = cap.read()
                    if not ret:
                        break
                    last = f
                frame = last
        else:
            # Unknown frame count, iterate to the end
            last = None
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                last = f
            frame = last

        if frame is None:
            return 'Could not extract last frame from video', 400

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_io = io.BytesIO()
        img.save(img_io, 'WEBP', quality=95)
        img_io.seek(0)

        return send_file(
            img_io,
            mimetype='image/webp',
            as_attachment=True,
            download_name='last_frame.webp'
        )
    except Exception as e:
        return f'Error processing video: {str(e)}', 500
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass


@app.route('/concat', methods=['POST'])
def concat_videos():
    if 'file1' not in request.files or 'file2' not in request.files:
        return 'Missing files', 400

    f1 = request.files['file1']
    f2 = request.files['file2']
    if f1.filename == '' or f2.filename == '':
        return 'No selected file', 400

    # Generate unique filenames
    temp_id = str(uuid.uuid4())
    in1 = f"/tmp/{temp_id}_1.mp4"
    in2 = f"/tmp/{temp_id}_2.mp4"
    normalized1 = f"/tmp/{temp_id}_norm1.mp4"
    normalized2 = f"/tmp/{temp_id}_norm2.mp4"
    out = f"/tmp/{temp_id}_output.mp4"
    
    temp_files = [in1, in2, normalized1, normalized2, out]

    try:
        # Save uploaded files
        f1.save(in1)
        f2.save(in2)

        def get_video_info(path):
            """Get comprehensive video information including codecs, resolution, fps"""
            try:
                probe = ffmpeg.probe(path)
                video_stream = None
                audio_stream = None
                
                for stream in probe.get('streams', []):
                    if stream.get('codec_type') == 'video' and not video_stream:
                        video_stream = stream
                    elif stream.get('codec_type') == 'audio' and not audio_stream:
                        audio_stream = stream
                
                return {
                    'video': video_stream,
                    'audio': audio_stream,
                    'duration': float(probe.get('format', {}).get('duration', 0))
                }
            except Exception as e:
                raise Exception(f"Failed to probe video {path}: {str(e)}")

        def detect_hardware_acceleration():
            """Detect available hardware acceleration"""
            try:
                # Check for NVIDIA GPU support
                ffmpeg.run(ffmpeg.input('color=c=black:s=64x64:d=0.1', f='lavfi')
                          .output('/tmp/test_nvenc.mp4', vcodec='h264_nvenc', t=0.1, loglevel='quiet'), 
                          overwrite_output=True)
                os.remove('/tmp/test_nvenc.mp4')
                return 'h264_nvenc'
            except:
                try:
                    # Check for Intel Quick Sync
                    ffmpeg.run(ffmpeg.input('color=c=black:s=64x64:d=0.1', f='lavfi')
                              .output('/tmp/test_qsv.mp4', vcodec='h264_qsv', t=0.1, loglevel='quiet'), 
                              overwrite_output=True)
                    os.remove('/tmp/test_qsv.mp4')
                    return 'h264_qsv'
                except:
                    # Fallback to software encoding
                    return 'libx264'

        def get_optimal_specs(info1, info2):
            """Determine optimal video specifications for concatenation"""
            # Get resolutions
            res1 = (info1['video'].get('width', 1920), info1['video'].get('height', 1080))
            res2 = (info2['video'].get('width', 1920), info2['video'].get('height', 1080))
            
            # Use the lower resolution to avoid upscaling
            target_width = min(res1[0], res2[0])
            target_height = min(res1[1], res2[1])
            
            # Ensure dimensions are even (required for H.264)
            target_width = target_width - (target_width % 2)
            target_height = target_height - (target_height % 2)
            
            # Cap resolution to prevent H.264 level issues
            max_pixels = 1920 * 1080  # 1080p max
            current_pixels = target_width * target_height
            if current_pixels > max_pixels:
                ratio = (max_pixels / current_pixels) ** 0.5
                target_width = int(target_width * ratio) - (int(target_width * ratio) % 2)
                target_height = int(target_height * ratio) - (int(target_height * ratio) % 2)
            
            # Get frame rates and cap to reasonable maximum
            def parse_fps(fps_str):
                try:
                    if '/' in str(fps_str):
                        num, den = map(float, str(fps_str).split('/'))
                        return num / den if den != 0 else 30.0
                    return float(fps_str)
                except:
                    return 30.0
            
            fps1 = parse_fps(info1['video'].get('r_frame_rate', '30/1'))
            fps2 = parse_fps(info2['video'].get('r_frame_rate', '30/1'))
            
            # Use the lower fps to avoid issues, but cap at 60fps
            target_fps = min(max(fps1, fps2), 60.0)
            
            # Check audio compatibility
            has_audio = info1['audio'] is not None and info2['audio'] is not None
            
            return {
                'width': target_width,
                'height': target_height,
                'fps': target_fps,
                'has_audio': has_audio
            }

        def normalize_video(input_path, output_path, target_specs):
            """Normalize video to consistent format for concatenation"""
            try:
                input_stream = ffmpeg.input(input_path)
                
                # Use hardware-accelerated encoding if available
                vcodec = detect_hardware_acceleration()
                
                # Calculate bitrate based on resolution and fps to stay within H.264 limits
                pixels_per_second = target_specs['width'] * target_specs['height'] * target_specs['fps']
                
                # Conservative bitrate calculation to avoid level limits
                if pixels_per_second > 1920 * 1080 * 30:
                    video_bitrate = '8M'  # High resolution/fps
                elif pixels_per_second > 1280 * 720 * 30:
                    video_bitrate = '5M'  # 720p
                else:
                    video_bitrate = '3M'  # Lower resolution
                
                # Normalize to common settings with H.264 level constraints
                video_args = {
                    'vcodec': vcodec,
                    'pix_fmt': 'yuv420p',
                    's': f"{target_specs['width']}x{target_specs['height']}",
                    'r': target_specs['fps'],
                    'movflags': 'faststart',
                    'video_bitrate': video_bitrate,
                }
                
                # Add codec-specific settings
                if vcodec == 'libx264':
                    video_args.update({
                        'preset': 'medium',
                        'level': '4.1',  # Ensures compatibility with most devices
                        'maxrate': video_bitrate,
                        'bufsize': f"{int(video_bitrate[:-1]) * 2}M"  # 2x bitrate for buffer
                    })
                elif vcodec in ['h264_nvenc', 'h264_qsv']:
                    video_args.update({
                        'preset': 'medium',
                        'level': '4.1'
                    })
                
                if target_specs['has_audio']:
                    # Include audio normalization
                    audio_args = {
                        'acodec': 'aac',
                        'audio_bitrate': '128k',
                        'ar': 44100
                    }
                    stream = ffmpeg.output(input_stream, output_path, **video_args, **audio_args)
                else:
                    # Video only
                    stream = ffmpeg.output(input_stream.video, output_path, **video_args)
                
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
            except Exception as e:
                raise Exception(f"Failed to normalize video {input_path}: {str(e)}")

        # Get video information
        info1 = get_video_info(in1)
        info2 = get_video_info(in2)
        
        # Validate video specs and warn about potential issues
        def validate_video_specs(info):
            issues = []
            fps_str = info['video'].get('r_frame_rate', '30/1')
            
            def parse_fps(fps_str):
                try:
                    if '/' in str(fps_str):
                        num, den = map(float, str(fps_str).split('/'))
                        return num / den if den != 0 else 30.0
                    return float(fps_str)
                except:
                    return 30.0
            
            fps = parse_fps(fps_str)
            width = int(info['video'].get('width', 0))
            height = int(info['video'].get('height', 0))
            
            if fps > 120:
                issues.append(f"Extremely high frame rate ({fps:.1f} fps) detected")
            if width * height > 4096 * 2160:  # 4K
                issues.append(f"Ultra-high resolution ({width}x{height}) may cause issues")
            
            return issues
        
        validation_issues = []
        validation_issues.extend(validate_video_specs(info1))
        validation_issues.extend(validate_video_specs(info2))
        
        # Log validation issues but continue processing with normalization
        if validation_issues:
            print(f"Video validation warnings: {validation_issues}")
        
        # Determine optimal specifications
        target_specs = get_optimal_specs(info1, info2)
        
        # Normalize both videos to ensure compatibility
        normalize_video(in1, normalized1, target_specs)
        normalize_video(in2, normalized2, target_specs)

        # Optimized concatenation using normalized inputs
        input1 = ffmpeg.input(normalized1)
        input2 = ffmpeg.input(normalized2)
        
        # Detect best encoder for final output
        vcodec = detect_hardware_acceleration()
        
        # Calculate appropriate bitrate for final output
        pixels_per_second = target_specs['width'] * target_specs['height'] * target_specs['fps']
        
        if pixels_per_second > 1920 * 1080 * 30:
            final_bitrate = '10M'  # High quality for high resolution
        elif pixels_per_second > 1280 * 720 * 30:
            final_bitrate = '6M'   # 720p quality
        else:
            final_bitrate = '4M'   # Standard quality
        
        # Configure optimal encoding settings with H.264 constraints
        encode_args = {
            'vcodec': vcodec,
            'movflags': 'faststart',
            'video_bitrate': final_bitrate,
        }
        
        # Add codec-specific settings for final output
        if vcodec == 'libx264':
            encode_args.update({
                'preset': 'slow',      # Better quality for final output
                'level': '4.1',        # H.264 level constraint
                'maxrate': final_bitrate,
                'bufsize': f"{int(final_bitrate[:-1]) * 2}M",
                'profile': 'high'      # Better compression
            })
        elif vcodec in ['h264_nvenc', 'h264_qsv']:
            encode_args.update({
                'preset': 'slow',
                'level': '4.1',
                'profile': 'high'
            })

        if target_specs['has_audio']:
            # Concatenate both video and audio
            v1, a1 = input1.video, input1.audio
            v2, a2 = input2.video, input2.audio
            
            joined_video, joined_audio = ffmpeg.concat(v1, a1, v2, a2, v=1, a=1).node
            
            stream = ffmpeg.output(
                joined_video, joined_audio, out,
                acodec='aac',
                audio_bitrate='192k',  # Higher quality audio
                **encode_args
            )
        else:
            # Video only concatenation
            v1, v2 = input1.video, input2.video
            joined_video = ffmpeg.concat(v1, v2, v=1, a=0).node[0]
            
            stream = ffmpeg.output(joined_video, out, **encode_args)

        # Run with progress monitoring
        ffmpeg.run(stream, overwrite_output=True)

        return send_file(out, mimetype='video/mp4', as_attachment=True, download_name='concatenated.mp4')
        
    except Exception as e:
        return jsonify({'error': f'Failed to concatenate videos: {str(e)}'}), 500
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    """Analyze video file and return comprehensive information"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    temp_path = f"/tmp/{uuid.uuid4()}.mp4"
    
    try:
        file.save(temp_path)
        
        # Get detailed video information
        probe = ffmpeg.probe(temp_path)
        
        video_stream = None
        audio_stream = None
        
        for stream in probe.get('streams', []):
            if stream.get('codec_type') == 'video' and not video_stream:
                video_stream = stream
            elif stream.get('codec_type') == 'audio' and not audio_stream:
                audio_stream = stream
        
        if not video_stream:
            return jsonify({'error': 'No video stream found'}), 400
        
        # Parse frame rate
        def parse_fps(fps_str):
            try:
                if '/' in str(fps_str):
                    num, den = map(float, str(fps_str).split('/'))
                    return num / den if den != 0 else 30.0
                return float(fps_str)
            except:
                return 30.0
        
        fps = parse_fps(video_stream.get('r_frame_rate', '30/1'))
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        duration = float(probe.get('format', {}).get('duration', 0))
        
        # Calculate specs and potential issues
        pixels_per_second = width * height * fps
        total_pixels = width * height
        
        # Check for potential issues
        issues = []
        warnings = []
        
        if fps > 60:
            issues.append(f"Very high frame rate ({fps:.1f} fps) may cause encoding issues")
        elif fps > 30:
            warnings.append(f"High frame rate ({fps:.1f} fps) detected")
        
        if total_pixels > 1920 * 1080:
            issues.append(f"Resolution ({width}x{height}) exceeds 1080p, may be downscaled")
        
        if pixels_per_second > 1920 * 1080 * 60:
            issues.append("Video complexity may exceed H.264 level limits")
        
        # Determine recommended settings
        recommended_fps = min(fps, 60.0)
        recommended_width = min(width, 1920)
        recommended_height = min(height, 1080)
        
        # Ensure even dimensions
        recommended_width = recommended_width - (recommended_width % 2)
        recommended_height = recommended_height - (recommended_height % 2)
        
        response_data = {
            'original_specs': {
                'width': width,
                'height': height,
                'fps': round(fps, 2),
                'duration': round(duration, 2),
                'codec': video_stream.get('codec_name'),
                'bitrate': video_stream.get('bit_rate'),
                'has_audio': audio_stream is not None,
                'audio_codec': audio_stream.get('codec_name') if audio_stream else None
            },
            'recommended_specs': {
                'width': recommended_width,
                'height': recommended_height,
                'fps': recommended_fps,
                'max_bitrate': '10M' if pixels_per_second > 1920 * 1080 * 30 else '6M'
            },
            'analysis': {
                'pixels_per_second': int(pixels_per_second),
                'h264_level_compatible': pixels_per_second <= 1920 * 1080 * 60,
                'issues': issues,
                'warnings': warnings,
                'concatenation_ready': len(issues) == 0
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Failed to analyze video: {str(e)}'}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

if __name__ == '__main__':
    # Only run the development server if explicitly requested
    import os
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'

    if debug_mode:
        print("Running in development mode - Use FLASK_DEBUG=true only for development!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Production mode detected. Use a WSGI server like Gunicorn.")
        print("Example: gunicorn --bind 0.0.0.0:5000 app:app")
        print("For Docker deployment, the app will be served by gunicorn automatically.")
        # In Docker/containerized environments, don't exit - let the WSGI server handle it
        if os.getenv('DOCKER_CONTAINER', '').lower() in ('true', '1', 'yes'):
            print("Running in Docker container - app object is available for WSGI server")
        else:
            print("Not running in Docker - exiting. Use gunicorn for production deployment.")
            exit(1)
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
register_heif_opener()

app = Flask(__name__)

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

    in1 = f"/tmp/{uuid.uuid4()}.mp4"
    in2 = f"/tmp/{uuid.uuid4()}.mp4"
    out = f"/tmp/{uuid.uuid4()}.mp4"

    try:
        f1.save(in1)
        f2.save(in2)

        def has_audio(path):
            # Use ffprobe to check if the file has an audio stream
            try:
                probe = ffmpeg.probe(path)
                streams = probe.get('streams', [])
                for s in streams:
                    if s.get('codec_type') == 'audio':
                        return True
                return False
            except Exception:
                return False

        audio1 = has_audio(in1)
        audio2 = has_audio(in2)

        i1 = ffmpeg.input(in1)
        i2 = ffmpeg.input(in2)
        v1 = i1.video
        v2 = i2.video

        if audio1 and audio2:
            a1 = i1.audio
            a2 = i2.audio
            vcat, acat = ffmpeg.concat(v1, a1, v2, a2, v=1, a=1).node
            stream = ffmpeg.output(
                vcat, acat, out, vcodec='libx264', acodec='aac', movflags='faststart'
            )
            ffmpeg.run(stream, overwrite_output=True)
        else:
            # If either video has no audio, concat video only
            vcat = ffmpeg.concat(v1, v2, v=1, a=0).node[0]
            stream = ffmpeg.output(
                vcat, out, vcodec='libx264', movflags='faststart'
            )
            ffmpeg.run(stream, overwrite_output=True)

        return send_file(out, mimetype='video/mp4', as_attachment=True, download_name='concat.mp4')
    except Exception as e:
        return f'Error concatenating videos: {str(e)}', 500
    finally:
        for p in [in1, in2, out]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
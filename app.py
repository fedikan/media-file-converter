from flask import Flask, request, jsonify, send_file
import ffmpeg
import os
import uuid
from scipy.io import wavfile
from scipy.signal import find_peaks
import numpy as np
import os
import uuid
import wave
from PIL import Image

app = Flask(__name__)

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
        width = int(request.form.get('width', 0))
        height = int(request.form.get('height', 0))
    except ValueError:
        return 'Invalid width or height', 400

    if file:
        # Save the original file
        original_filename = str(uuid.uuid4())
        input_path = f'/tmp/{original_filename}'
        file.save(input_path)

        # Open the image file
        with Image.open(input_path) as img:
            # If dimensions are provided, resize the image
            if width > 0 and height > 0:
                img = img.resize((width, height), Image.ANTIALIAS)
            
            output_filename = f'{original_filename}.{output_format}'
            output_path = f'/tmp/{output_filename}'

            # Convert and save the image in the specified format with optimization
            if output_format == 'webp':
                img.save(output_path, format='WEBP', quality=80, method=6)  # High quality and compression for web
            else:
                # For other formats, adjust quality and parameters as needed
                img.save(output_path, format=output_format.upper())

            # Send the converted file
            return send_file(output_path, as_attachment=True)

    return 'Unsupported file type', 400


def resize_and_pad(img, desired_dimensions):
    img_ratio = img.width / img.height
    closest_fit = min(desired_dimensions, key=lambda x: abs((x[0]/x[1]) - img_ratio))
    
    # Resize image to maintain aspect ratio
    img = img.resize((closest_fit[0], int(closest_fit[0] / img_ratio)), Image.ANTIALIAS)
    
    # Create a new image with desired dimensions and paste the resized image
    new_img = Image.new("RGB", closest_fit, (255, 255, 255))
    new_img.paste(img, ((closest_fit[0] - img.width) // 2, (closest_fit[1] - img.height) // 2))
    
    return new_img

@app.route('/convert-reference', methods=['POST'])
def convert_reference():
    data = request.json
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400
    
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        desired_dimensions = [
            (1024, 1024), (1152, 896), (1216, 832), (1344, 768), (1536, 640),
            (640, 1536), (768, 1344), (832, 1216), (896, 1152)
        ]
        converted_img = resize_and_pad(img, desired_dimensions)
        
        temp_path = "optimized_reference.png"
        converted_img.save(temp_path)
        
        return send_file(temp_path, as_attachment=True, attachment_filename='optimized_reference.png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
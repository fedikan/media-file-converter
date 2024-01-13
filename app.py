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

@app.route('/peaks', methods=['POST'])
def get_peaks():
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
            channels = wav_file.getnchannels()
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

        # Return the peaks as JSON
        return jsonify({'left_peaks': left_peaks, 'right_peaks': right_peaks})

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
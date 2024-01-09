from flask import Flask, request, send_file
import ffmpeg
import os
import uuid
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

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
        # Save the original file
        original_filename = str(uuid.uuid4())
        input_path = f'/tmp/{original_filename}.wav'
        file.save(input_path)

        # Read the WAV file
        sample_rate, audio_data = wavfile.read(input_path)
        
        # Calculate the number of peaks per second
        duration_in_seconds = len(audio_data) / sample_rate
        peaks_per_second = 20  # Adjust this value as needed
        total_peaks = int(duration_in_seconds * peaks_per_second)
        
        # Find peaks
        peaks, _ = find_peaks(audio_data, distance=sample_rate/peaks_per_second)
        
        # Select the first 'total_peaks' peakse
        selected_peaks = peaks[:total_peaks]
        
        # Convert selected peaks to time
        peak_times = selected_peaks / sample_rate
        
        # Clean up the temporary file
        os.remove(input_path)
        
        # Return the peak times as a JSON response
        return {'peaks': peak_times.tolist()}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
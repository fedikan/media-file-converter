from flask import Flask, request, send_file
import ffmpeg
import os
import uuid

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
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream)

        # Send the converted file
        return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
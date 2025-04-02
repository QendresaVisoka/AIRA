import os
import pydicom
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io

# Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Set the folder where files will be uploaded
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload-dicom', methods=['POST'])
def upload_dicom():
    if 'dicomFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['dicomFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Save the uploaded DICOM file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Convert DICOM to PNG image and return the image
        try:
            # Read the DICOM file with pydicom
            dicom_data = pydicom.dcmread(file_path)

            # Get the pixel data (assumes the DICOM contains image data)
            pixel_data = dicom_data.pixel_array

            # Normalize the pixel data to the range [0, 255] (8-bit)
            if pixel_data.dtype != np.uint8:
                pixel_data = np.uint8((pixel_data - np.min(pixel_data)) / (np.max(pixel_data) - np.min(pixel_data)) * 255)

            # Convert the pixel data to an image (PIL Image)
            image = Image.fromarray(pixel_data)

            # Save the image as a PNG in memory
            img_io = io.BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)

            # Serve the PNG image as a response
            return send_file(img_io, mimetype='image/png')

        except Exception as e:
            return jsonify({'error': f'Error processing DICOM file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type, only DICOM (.dcm) files are allowed'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'dcm'

if __name__ == '__main__':
    app.run(debug=True)

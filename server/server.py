import os
import cv2
import logging
import numpy as np
import pydicom
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io

# Initialize Flask and CORS
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'dcm'

@app.route('/upload-dicom', methods=['POST'])
def upload_dicom():
    if 'dicomFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['dicomFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Save uploaded file as 'latest.dcm' always
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'latest.dcm')
        file.save(file_path)

        try:
            dicom_data = pydicom.dcmread(file_path)
            pixel_data = dicom_data.pixel_array

            if pixel_data.dtype != np.uint8:
                pixel_data = np.uint8((pixel_data - np.min(pixel_data)) / (np.max(pixel_data) - np.min(pixel_data)) * 255)

            image = Image.fromarray(pixel_data)
            img_io = io.BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')
        except Exception as e:
            return jsonify({'error': f'Error processing DICOM file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type, only DICOM (.dcm) files are allowed'}), 400

@app.route('/preprocess-dicom', methods=['POST'])
def preprocess_dicom():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'latest.dcm')

    try:
        dicom_data = pydicom.dcmread(file_path)
        pixel_data = dicom_data.pixel_array

        preprocessed_image = resize_and_pad(crop_image(pixel_data))

        img_io = io.BytesIO()
        is_gray = len(preprocessed_image.shape) == 2

        if is_gray:
            preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)

        success, encoded_image = cv2.imencode('.png', preprocessed_image)
        if not success:
            raise Exception("Could not encode image.")

        img_io.write(encoded_image.tobytes())
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': f'Error processing DICOM file: {str(e)}'}), 500

def crop_image(image):
    try:
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 1000
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        if len(filtered_contours) > 0:
            # Find the largest contour
            largest_contour = max(filtered_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return image[y:y+h, x:x+w]
        return image
    except Exception as e:
        logging.error(f"Error cropping image: {str(e)}")
        return image

def resize_and_pad(image, target_size=(224, 224)):
    try:
        if image.dtype != np.uint8:
            image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)

        old_size = image.shape[:2]
        ratio = min(target_size[1]/old_size[1], target_size[0]/old_size[0])
        new_size = tuple([int(x * ratio) for x in old_size])

        resized = cv2.resize(image, (new_size[1], new_size[0]))
        delta_w = int(target_size[1] - new_size[1])
        delta_h = int(target_size[0] - new_size[0])

        top, bottom = int(delta_h // 2), int(delta_h - (delta_h // 2))
        left, right = int(delta_w // 2), int(delta_w - (delta_w // 2))

        if len(image.shape) == 2:
            resized_padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        else:
            resized_padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return resized_padded_image
    except Exception as e:
        logging.error(f"Error resizing and padding image: {str(e)}")
        return image

if __name__ == '__main__':
    app.run(debug=True)

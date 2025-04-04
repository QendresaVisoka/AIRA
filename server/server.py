from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pydicom
import logging
import numpy as np
import os
import cv2
import io
from PIL import Image
import time

app = Flask(__name__)
CORS(app)

# Directory configurations
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_images'
PREDICTIONS_FOLDER = 'predictions'
MODEL_PATH = 'models\EfficientNet_UNet_25Layers.keras'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

# Dice Coefficient
def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Dice Loss
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Weighted Cross-Entropy Loss
def weighted_binary_crossentropy(y_true, y_pred, pos_weight=3.0, neg_weight=1.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_true = tf.cast(y_true, tf.float32)
    loss = -(
        pos_weight * y_true * tf.math.log(y_pred) +
        neg_weight * (1 - y_true) * tf.math.log(1 - y_pred)
    )

    return tf.reduce_mean(loss)

# Combined Loss Function
def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + weighted_binary_crossentropy(y_true, y_pred)

# Load the model
model = load_model(MODEL_PATH, custom_objects={'combined_loss': combined_loss, 'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss, 'weighted_binary_crossentropy': weighted_binary_crossentropy})


# Allowed file validation
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

        preprocessed_image, padding, bbox, cropped_shape = preprocess_image(pixel_data)

        # Save the preprocessed image in the processed_images folder
        preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'preprocessed.png')
        cv2.imwrite(preprocessed_path, preprocessed_image)

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
    

import time  # Import this at the beginning of your file

@app.route('/predict-mask', methods=['POST'])
def predict_mask():
    try:
        preprocessed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'preprocessed.png')
        print(f"Reading image from: {preprocessed_path}")

        # Load the preprocessed image
        preprocessed_image = cv2.imread(preprocessed_path, cv2.IMREAD_GRAYSCALE)
        if preprocessed_image is None:
            raise ValueError("No preprocessed image found or image could not be read.")

        print("Image loaded. Shape:", preprocessed_image.shape)

        print("Predicting mask...")

        # Predict the full mask using sliding window approach
        predicted_mask = predict_full_mask_from_patches(
            model,
            preprocessed_image,
            patch_size=(128, 128),
            stride=64,
            padding=32
        )

        predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

        # Save to memory
        img_io = io.BytesIO()
        image = Image.fromarray(predicted_mask)
        image.save(img_io, 'PNG')
        img_io.seek(0)

        # Save to disk
        timestamp = int(time.time())
        prediction_path = os.path.join(app.config['PREDICTIONS_FOLDER'], f'prediction_{timestamp}.png')
        image.save(prediction_path)

        print("Prediction complete.")
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        print("ERROR during prediction:", str(e))
        return jsonify({'error': f'Error predicting mask: {str(e)}'}), 500



def predict_full_mask_from_patches(model, image, patch_size=(128, 128), stride=64, padding=32):
    ndim = image.ndim
    if ndim == 2:
        image_padded = np.pad(image, ((padding, padding), (padding, padding)), mode='reflect')
    elif ndim == 3:
        image_padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    else:
        raise ValueError("Image must be 2D or 3D.")

    H_pad, W_pad = image_padded.shape[:2]
    full_mask = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight_map = np.zeros((H_pad, W_pad), dtype=np.float32)

    for y in range(0, H_pad - patch_size[0] + 1, stride):
        for x in range(0, W_pad - patch_size[1] + 1, stride):
            patch = image_padded[y:y+patch_size[0], x:x+patch_size[1]]
            patch_input = np.expand_dims(patch, axis=0).astype(np.float32)
            if patch_input.shape[-1] != 3:
                patch_input = np.repeat(patch_input[..., np.newaxis], 3, axis=-1)

            pred = model.predict(patch_input, verbose=0)[0, :, :, 0]
            full_mask[y:y+patch_size[0], x:x+patch_size[1]] += pred
            weight_map[y:y+patch_size[0], x:x+patch_size[1]] += 1.0

    weight_map[weight_map == 0] = 1.0
    averaged_mask = full_mask / weight_map

    return averaged_mask[padding:-padding, padding:-padding]



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
            largest_contour = max(filtered_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped = image[y:y+h, x:x+w]
            return cropped, (x, y, w, h), image.shape, cropped.shape
        else:
            return image, None, image.shape, image.shape
    except Exception as e:
        logging.error(f"Error cropping image: {str(e)}")
        return image, None, image.shape, image.shape

def resize_image(image, target_size=(224, 224)):
    try:
        if image.dtype != np.uint8:
            image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        old_size = image.shape[:2]
        ratio = min(target_size[1]/old_size[1], target_size[0]/old_size[0])
        new_size = tuple([int(x * ratio) for x in old_size])
        resized_image = cv2.resize(image, (new_size[1], new_size[0]))
        return resized_image, new_size, ratio
    except Exception as e:
        logging.error(f"Error resizing image: {str(e)}")
        return image, old_size, 1

def pad_image(resized_image, target_size=(224, 224)):
    try:
        resized_height, resized_width = resized_image.shape[:2]
        delta_w = int(target_size[1] - resized_width)
        delta_h = int(target_size[0] - resized_height)
        top, bottom = int(delta_h // 2), int(delta_h - (delta_h // 2))
        left, right = int(delta_w // 2), int(delta_w - (delta_w // 2))
        padding = (top, bottom, left, right)
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return padded_image, padding
    except Exception as e:
        logging.error(f"Error padding image: {str(e)}")
        return resized_image, (0, 0, 0, 0)

def preprocess_image(image, target_size=(224, 224)):
    try:
        cropped_image, bbox, original_size, cropped_shape = crop_image(image)
        if bbox is None:
            raise ValueError("No valid contours found; using original image for further processing.")
        resized_image, new_size, ratio = resize_image(cropped_image, target_size)
        padded_image, padding = pad_image(resized_image, target_size)
        return padded_image, padding, bbox, cropped_shape
    except Exception as e:
        logging.error(f"Error preprocessing image: {str(e)}")
        return image, (0, 0, 0, 0), None, image.shape

if __name__ == '__main__':
    app.run(debug=True)

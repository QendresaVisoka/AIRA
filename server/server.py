# ----------Imports----------
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import pydicom
import logging
import numpy as np
import os
import cv2
import io
from PIL import Image
import time
from scipy.ndimage import label
import base64
from flask import send_from_directory
import base64
import json

# ----------App Initialization----------
app = Flask(__name__)
CORS(app)

# ----------Path Configurations----------
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_images'
PREDICTIONS_FOLDER = 'predictions'
MODEL_PATH = 'models/Fine-Tuned_Model.keras'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

# ============Functions============
# ----Custom Losses and Metrics----
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

# =========Helper Functions==========
# Function to extract bounding boxes from a binary mask
def bbox_from_mask(mask):
    mask = mask.squeeze()
    components, num_features = label(mask)
    bboxes = []
    for i in range(1, num_features + 1):
        rows, cols = np.where(components == i)
        if rows.size > 0 and cols.size > 0:
            x_min, x_max = cols.min(), cols.max()
            y_min, y_max = rows.min(), rows.max()
            bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes

# -----------Preprocessing Functions-----------
# Function to crop the image based on contours
def crop_image(image):
    try:
        # Convert to grayscale if the image is RGB
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Thresholding to create a binary image
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours based on area
        min_contour_area = 1000
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # Check if any contours rtemain after filtering
        if len(filtered_contours) > 0:
            # Find the largest contour and crop the image
            largest_contour = max(filtered_contours, key=cv2.contourArea)
            # Get the bounding rectangle of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Crop the image using the bounding rectangle
            cropped = image[y:y+h, x:x+w]
            return cropped, (x, y, w, h), image.shape, cropped.shape
        else:
            return image, None, image.shape, image.shape
    except Exception as e:
        logging.error(f"Error cropping image: {str(e)}")
        return image, None, image.shape, image.shape
    
# Function to resize the image
def resize_image(image, target_size=(224, 224)):
    try:
        # Normalize to uint8 if needed
        if image.dtype != np.uint8:
            image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        # Get the original size and calculate the new size
        old_size = image.shape[:2]
        ratio = min(target_size[1]/old_size[1], target_size[0]/old_size[0])
        new_size = tuple([int(x * ratio) for x in old_size])
        # Resize the image
        resized_image = cv2.resize(image, (new_size[1], new_size[0]))
        return resized_image, new_size, ratio
    except Exception as e:
        logging.error(f"Error resizing image: {str(e)}")
        return image, old_size, 1

# Function to pad the image
def pad_image(resized_image, target_size=(224, 224)):
    try:
        # Get the resized image dimensions
        resized_height, resized_width = resized_image.shape[:2]
        # Calculate the difference between the target size and the resized image
        delta_w = int(target_size[1] - resized_width)
        delta_h = int(target_size[0] - resized_height)
        # Calculate padding values
        top, bottom = int(delta_h // 2), int(delta_h - (delta_h // 2))
        left, right = int(delta_w // 2), int(delta_w - (delta_w // 2))
        padding = (top, bottom, left, right)
        # Pad the image
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return padded_image, padding
    except Exception as e:
        logging.error(f"Error padding image: {str(e)}")
        return resized_image, (0, 0, 0, 0)

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    try:
        # Crop the image based on contours
        cropped_image, bbox, original_size, cropped_shape = crop_image(image)
        if bbox is None:
            raise ValueError("No valid contours found; using original image for further processing.")
        # Resize the cropped image
        resized_image, new_size, ratio = resize_image(cropped_image, target_size)
        # Pad the resized image
        padded_image, padding = pad_image(resized_image, target_size)
        return padded_image, padding, bbox, cropped_shape
    except Exception as e:
        logging.error(f"Error preprocessing image: {str(e)}")
        return image, (0, 0, 0, 0), None, image.shape

# -----------Prediction Function-----------
# Function to predict the full mask from patches
def predict_full_mask_from_patches(model, image, patch_size=(128, 128), stride=64, padding=32):
    ndim = image.ndim
    # Pad the image to handle edge cases
    if ndim == 2:
        image_padded = np.pad(image, ((padding, padding), (padding, padding)), mode='reflect')
    elif ndim == 3:
        image_padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    else:
        raise ValueError("Image must be 2D or 3D.")

    # Get the padded image dimensions
    H_pad, W_pad = image_padded.shape[:2]
    # Initialize the full mask and weight map
    full_mask = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight_map = np.zeros((H_pad, W_pad), dtype=np.float32)

    # Iterate over the image in patches
    for y in range(0, H_pad - patch_size[0] + 1, stride):
        for x in range(0, W_pad - patch_size[1] + 1, stride):
            # Extract the patch
            patch = image_padded[y:y+patch_size[0], x:x+patch_size[1]]
            # Ensure the patch is in the correct format for the model
            patch_input = np.expand_dims(patch, axis=0).astype(np.float32)
            if patch_input.shape[-1] != 3:
                patch_input = np.repeat(patch_input[..., np.newaxis], 3, axis=-1)

            # Predict the mask for the patch
            pred = model.predict(patch_input, verbose=0)[0, :, :, 0]
            full_mask[y:y+patch_size[0], x:x+patch_size[1]] += pred
            weight_map[y:y+patch_size[0], x:x+patch_size[1]] += 1.0

    # Normalize the full mask by the weight map
    # Avoid division by zero
    weight_map[weight_map == 0] = 1.0
    averaged_mask = full_mask / weight_map

    return averaged_mask[padding:-padding, padding:-padding]

# -----------Postprocessing Function-----------
# Function to postprocess the mask to fit the original image
def postprocess_mask(padded_image, padding, original_size, bbox, cropped_shape, target_size=(224, 224)):
   
    # Extract the padding values (top, bottom, left, right)
    top, bottom, left, right = padding
    original_height, original_width = original_size

    # Remove the padding from the image
    image_without_padding = padded_image[top:target_size[0] - bottom, left:target_size[1] - right]

    # Resize the image based on the input aspect ratio to match the cropped shape
    image_height, image_width = image_without_padding.shape[:2]
    cropped_height, cropped_width = cropped_shape

    # Resize the image to fit within the cropped shape (keeping aspect ratio)
    resized_image = cv2.resize(image_without_padding, (cropped_width, cropped_height), interpolation=cv2.INTER_LINEAR)

    # Place the resized image into the expanded canvas based on the bounding box
    x, y, w, h = bbox

    # Create an empty canvas for the restored image (of the original size)
    expanded_image = np.zeros((original_height, original_width), dtype=np.uint8)

    # Ensure that the resized cropped region is placed back into the correct location in the original image
    expanded_image[y:y+h, x:x+w] = resized_image[:h, :w]  

    return image_without_padding, resized_image, expanded_image

# ------------File Validation Function-------
# Function to check if the file is a DICOM file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'dcm'
  

# ----------Load the model----------
model = load_model(MODEL_PATH, custom_objects={'combined_loss': combined_loss, 'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss, 'weighted_binary_crossentropy': weighted_binary_crossentropy})


# ----------Flask Routes----------
# Route to upload DICOM file
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

            # Normalize to uint8 if needed
            if pixel_data.dtype != np.uint8:
                pixel_data = np.uint8((pixel_data - np.min(pixel_data)) / (np.max(pixel_data) - np.min(pixel_data)) * 255)

            # Convert image to PNG and then base64
            image = Image.fromarray(pixel_data)
            img_io = io.BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Extract metadata
            patient_id = str(dicom_data.get("PatientID", "Unknown"))
            patient_sex = str(dicom_data.get("PatientSex", "Unknown"))
            patient_age = str(dicom_data.get("PatientAge", "Unknown"))

            return jsonify({
                'image': f"data:image/png;base64,{img_base64}",
                'patient': {
                    'id': patient_id,
                    'sex': patient_sex,
                    'age': patient_age
                }
            })

        except Exception as e:
            return jsonify({'error': f'Error processing DICOM file: {str(e)}'}), 500

    else:
        return jsonify({'error': 'Invalid file type, only DICOM (.dcm) files are allowed'}), 400

# Route to preprocess DICOM file
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
    

# Route to predict mask
@app.route('/predict-mask', methods=['POST'])
def predict_mask():
    try:
        preprocessed_path = os.path.join(PROCESSED_FOLDER, 'preprocessed.png')
        original_dicom_path = os.path.join(UPLOAD_FOLDER, 'latest.dcm')

        if not os.path.exists(preprocessed_path) or not os.path.exists(original_dicom_path):
            raise ValueError("Required files for prediction not found.")

        # Load DICOM and pixel data safely
        dicom_data = pydicom.dcmread(original_dicom_path)

        pixel_data = dicom_data.pixel_array
        if pixel_data.dtype != np.uint8:
            pixel_data = np.uint8((pixel_data - np.min(pixel_data)) / (np.max(pixel_data) - np.min(pixel_data)) * 255)

        # Load preprocessed image
        preprocessed_image = cv2.imread(preprocessed_path, cv2.IMREAD_GRAYSCALE)
        if preprocessed_image is None:
            raise ValueError("Preprocessed image could not be read.")

        # Predict mask
        pred_mask = predict_full_mask_from_patches(model, preprocessed_image)
        pred_mask = (pred_mask).astype(np.uint8) * 255

        # Postprocess
        processed, padding, bbox, cropped_shape = preprocess_image(pixel_data)
        _, _, restored_mask = postprocess_mask(
            pred_mask, padding, original_size=pixel_data.shape,
            bbox=bbox, cropped_shape=cropped_shape
        )

        if np.sum(restored_mask) == 0:
            print("No Tumor detected")
        else:
            print("Tumor detected")

        # Save mask
        np.save(os.path.join(PREDICTIONS_FOLDER, 'restored_mask.npy'), restored_mask)

        # Save bounding boxes
        bounding_boxes = bbox_from_mask(restored_mask)
        bounding_boxes = [[int(x) for x in box] for box in bounding_boxes]
        
        # Save pixel spacing for calculating mm measurements
        pixel_spacing = dicom_data.get("PixelSpacing", [1.0, 1.0])
        pixel_spacing = [float(x) for x in pixel_spacing]

        with open(os.path.join(PREDICTIONS_FOLDER, 'bounding_boxes.json'), 'w') as f:
            json.dump({'boxes': bounding_boxes, 'pixel_spacing': pixel_spacing}, f)

        # Create overlay
        # Map intensity values to JET colormap
        heatmap = cv2.applyColorMap(restored_mask, cv2.COLORMAP_JET)
        # Generate a 3-channel transparency mask to isolate tumor regions
        transparency_mask = restored_mask > 0
        transparency_mask = np.stack([transparency_mask] * 3, axis=-1)
        # Convert grayscale image to BGR, needed for heatmap overlay
        original_image = cv2.cvtColor(pixel_data, cv2.COLOR_GRAY2BGR)
        # Smooth heatmap for better visualization
        heatmap = cv2.GaussianBlur(heatmap, (17, 17), 20)
        # Remove heatmap color outside the tumor regions
        heatmap[~transparency_mask] = 0
        # Overlay heatmap on the original image
        overlay = cv2.addWeighted(original_image, 1, heatmap, 1, 0)
        cv2.imwrite(os.path.join(PREDICTIONS_FOLDER, 'overlay.png'), overlay)

        img_io = io.BytesIO()
        Image.fromarray(original_image).save(img_io, 'PNG')
        img_io.seek(0)

        response_data = {
            'tumorFound': bool(np.sum(restored_mask) > 0)
        }

        # Include image as base64 to send along with the response (optional)
        buffered = io.BytesIO()
        Image.fromarray(original_image).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response_data['image'] = f"data:image/png;base64,{img_str}"
        return jsonify(response_data)
    except Exception as e:
        print("Postprocessing Error:", e)
        return jsonify({'error': str(e)}), 500


# Route to get the overlay image
@app.route('/get-overlay', methods=['GET'])
def get_overlay():
    try:
        # Serve the overlay image from the PREDICTIONS_FOLDER directory
        overlay_path = os.path.join(PREDICTIONS_FOLDER, 'overlay.png')
        if not os.path.exists(overlay_path):
            return jsonify({'error': 'No overlay image found'}), 404
        return send_from_directory(PREDICTIONS_FOLDER, 'overlay.png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Route to get the image with the bounidng box
@app.route('/get-bounding-boxes', methods=['GET'])
def get_bounding_boxes():
    try:
        path = os.path.join(PREDICTIONS_FOLDER, 'bounding_boxes.json')
        if not os.path.exists(path):
            return jsonify({'error': 'No prediction data found'}), 404
        with open(path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Route to get the image with the heatmap
@app.route('/get-heatmap-legend', methods=['GET'])
def get_heatmap_legend():
    width = request.args.get('width', default=120, type=int)   # Total legend width
    height = request.args.get('height', default=20, type=int)  # Height of the bar
    label_area_height = 20  # Space for labels below the heatmap

    # Create horizontal gradient: Blue (low) → Red (high)
    gradient = np.linspace(0, 255, num=width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply JET colormap and alpha
    heatmap = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    alpha_channel = np.full((height, width), int(255 * 0.5), dtype=np.uint8)
    heatmap = np.dstack((heatmap, alpha_channel)) 

    # Blend with a light grey background
    # Create background (light grey)
    background = np.full((height, width, 3), 220, dtype=np.uint8)

    # Extract alpha mask and normalize
    alpha = heatmap[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    # Foreground (heatmap color only)
    heatmap_rgb = heatmap[:, :, :3]

    # Alpha blending
    blended = (alpha * heatmap_rgb + (1 - alpha) * background).astype(np.uint8)

    # Create canvas for the legend
    total_height = height + label_area_height
    canvas = np.zeros((total_height, width, 4), dtype=np.uint8)

    # Place blended heatmap onto canvas
    canvas[:height, :, :3] = blended
    canvas[:height, :, 3] = 255 

    # Add labels below the gradient
    label_values = [0.0, 0.5, 1.0]
    label_positions = [0, width // 2 - 10, width - 30]

    for value, x in zip(label_values, label_positions):
        cv2.putText(
            canvas,
            f'{value:.2f}',
            (x, height + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255, 255),  # White
            1,
            cv2.LINE_AA
        )

    # Convert to PNG
    _, buffer = cv2.imencode('.png', canvas)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'
    return response

# -----------Start App----------
if __name__ == '__main__':
    app.run(debug=True)

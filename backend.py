import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from torchvision.transforms import functional as F, transforms
from PIL import Image
import io
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import easyocr
import os
from io import BytesIO
from matplotlib import pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your models
model_path_1 = "models/vehicles_best_model.pt"
model_1 = torch.load(model_path_1, map_location=torch.device("cpu"))
model_1.eval()

model_path_2 = "models/my_model.h5"
model_2 = load_model(model_path_2)

# Transformations for the first model
transform_1 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Transformations for the second model
def transform_image(image):
    image = image.resize((224, 224))  # Resize image for model input
    image = np.array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def get_plate_coordinates(image, original_width, original_height):
    input_image = transform_image(image)
    predictions = model_2.predict(input_image)

    # Assuming the coordinates are normalized (0-1 range)
    ny = predictions[0]  # Denormalize the predictions

    # Rescale to the original dimensions
    x1, y1, x2, y2 = (
        int(ny[0] * original_width),
        int(ny[1] * original_height),
        int(ny[2] * original_width),
        int(ny[3] * original_height),
    )

    return x1, y1, x2, y2


def read_license_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error: Unable to load image. Please check the file path."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        return None, "License plate could not be detected."

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1 : x2 + 1, y1 : y2 + 1]

    reader = easyocr.Reader(["en"])
    result = reader.readtext(cropped_image)

    if len(result) == 0:
        return None, "No text detected on the license plate."

    text = result[0][-2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, 1, 2)
    text_offset_x = location[0][0][0]
    text_offset_y = location[0][0][1] - 15

    res = cv2.putText(
        img,
        text=text,
        org=(text_offset_x, text_offset_y),
        fontFace=font,
        fontScale=1,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    res = cv2.rectangle(
        img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3
    )

    is_success, buffer = cv2.imencode(".png", res)
    if not is_success:
        return None, "Error: Failed to encode image."

    return buffer, None


@app.route("/detect_car", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    # Prepare the image for the model input
    image_tensor = transform_1(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model_1(image_tensor)

    # Load the original image
    file.stream.seek(0)  # Reset the file pointer
    original_image = np.array(Image.open(file.stream).convert("RGB"))
    img = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # Process and draw predictions
    for boxes, scores, labels in zip(
        predictions[0]["boxes"], predictions[0]["scores"], predictions[0]["labels"]
    ):
        if scores.item() > 0.5:
            x1, y1, x2, y2 = [round(b.item()) for b in boxes]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the result as a file
    is_success, buffer = cv2.imencode(".png", img)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype="image/png")


@app.route("/read_plate", methods=["POST"])
def read_plate():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        buffer, error = read_license_plate(temp_file_path)
        os.remove(temp_file_path)
        if error:
            return jsonify({"error": error}), 400

        return send_file(
            BytesIO(buffer),
            mimetype="image/png",
            as_attachment=True,
            download_name="result.png",
        )


if __name__ == "__main__":
    app.run(debug=True)

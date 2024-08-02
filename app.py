from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load the trained YOLO model
model_path = 'best.pt'  # Replace with the path to your model
trained_model = YOLO(model_path)

def detect_and_annotate(image, trained_model, device='cpu'):
    # Predict using the trained model
    pred_img = trained_model.predict(image, device=device)

    # Convert image to RGB
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    for result in pred_img:
        objects = result.names
        for box in result.boxes:
            _, _, _, _, conf, obj_key = map(float, box.data[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(image, objects[int(obj_key)], (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(image, f'{conf:.2f}%', (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return image

@app.route('/')
def index():
    return render_template('index.html')

import base64

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('file')
    results = []

    for file in files:
        if file.filename == '':
            continue

        image = Image.open(file.stream)
        annotated_image = detect_and_annotate(image, trained_model)

        # Save annotated image to a bytes buffer
        buf = io.BytesIO()
        annotated_image_pil = Image.fromarray(annotated_image)
        annotated_image_pil.save(buf, format='PNG')
        buf.seek(0)

        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')

        results.append({
            'filename': file.filename,
            'image': encoded_image
        })

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)



import sys
from pathlib import Path

# Ensure yolov5 repo is on the Python path
YOLOV5_PATH = Path(__file__).resolve().parent / "yolov5"
sys.path.insert(0, str(YOLOV5_PATH))

import torch
from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np

from yolov5.utils.augmentations import letterbox 
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device


app = Flask(__name__)

# Load model
weights = 'yolov5s.pt'
device = select_device('cpu')
model = DetectMultiBackend(weights, device=device, dnn=False)
model.eval()

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_bytes = request.files['file'].read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Resize and pad image to 640x640
    img_np = np.array(img)
    img_resized = letterbox(img_np, new_shape=640)[0]

    # Convert to tensor
    img_resized = img_resized.transpose((2, 0, 1))  # HWC to CHW
    img_resized = np.ascontiguousarray(img_resized)

    img_tensor = torch.from_numpy(img_resized).to(device)
    img_tensor = img_tensor.float() / 255.0  # Normalize to [0,1]
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dim

    # Inference
    with torch.no_grad():
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    names = model.names
    detections = []
    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                label = names[int(cls)]
                detections.append({'label': label, 'confidence': float(conf)})

    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

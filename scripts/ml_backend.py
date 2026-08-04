"""
LabelStudio ML backend — YOLOv8 bbox predictions (danio_rerio + reflection).
Implements the LS ML backend API directly, no label-studio-ml dependency.

Run:   make ml-backend
Then:  LabelStudio → project Settings → Machine Learning → http://localhost:9090
Click: Predict All
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import os
import tempfile
from pathlib import Path

import cv2
import requests
import yaml
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from scripts.model_registry import load_yolo   # path OR models:/...@champion → native YOLO

CONFIG_PATH = 'config.yaml'
CLASSES     = {0: 'danio_rerio', 1: 'reflection'}

load_dotenv(Path(__file__).parent.parent / ".env")

LS_URL   = os.getenv('LABEL_STUDIO_URL', 'http://localhost:8080')
LS_TOKEN = os.getenv('LABEL_STUDIO_API_KEY', '')

app   = Flask(__name__)
model = None


def get_ls_headers():
    resp = requests.post(f"{LS_URL}/api/token/refresh", json={"refresh": LS_TOKEN})
    resp.raise_for_status()
    access = resp.json()["access"]
    return {"Authorization": f"Bearer {access}"}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_model():
    global model
    if model is None:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        # model for self-labeling lives in its own ml_backend block; fall back to
        # the tracker's model if ml_backend.model_path isn't set, so this never crashes.
        path = cfg.get('ml_backend', {}).get('model_path') or cfg['tracker']['model_path']
        model = load_yolo(path)
        print(f"  Model loaded: {path}")
    return model


def fetch_image(image_url):
    """Download image from LabelStudio and return local temp path."""
    if image_url.startswith('/data/'):
        image_url = LS_URL + image_url
    resp = requests.get(image_url, headers=get_ls_headers())
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


def predict_image(image_url):
    local_path = fetch_image(image_url)
    try:
        img    = cv2.imread(local_path)
        h, w   = img.shape[:2]
        yolo   = get_model()
        result = yolo(local_path, verbose=False)[0]

        annotations = []
        for box, cls_id, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            label = CLASSES.get(int(cls_id))
            if label is None:
                continue
            x1, y1, x2, y2 = box
            annotations.append({
                'from_name': 'label',
                'to_name':   'image',
                'type':      'rectanglelabels',
                'value': {
                    'x':               float(x1 / w * 100),
                    'y':               float(y1 / h * 100),
                    'width':           float((x2 - x1) / w * 100),
                    'height':          float((y2 - y1) / h * 100),
                    'rectanglelabels': [label],
                },
                'score': float(conf),
            })
        return annotations
    finally:
        os.unlink(local_path)


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get('/')
@app.get('/health')
def health():
    return jsonify({'status': 'UP'})


@app.route('/setup', methods=['GET', 'POST'])
def setup():
    return jsonify({'model_version': '1.0.0', 'status': 'UP'})


@app.post('/predict')
def predict():
    tasks   = request.json.get('tasks', [])
    results = []
    for task in tasks:
        annotations = predict_image(task['data']['image'])
        avg_score   = sum(a['score'] for a in annotations) / len(annotations) if annotations else 0
        results.append({'result': annotations, 'score': avg_score})
    return jsonify({'results': results})


@app.post('/fit')
def fit():
    return jsonify({'status': 'not implemented'})


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    get_model()
    print(f"  ML backend running at http://localhost:9090")
    app.run(host='0.0.0.0', port=9090, debug=False)

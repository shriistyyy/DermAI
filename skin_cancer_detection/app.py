"""
Skin Cancer Detection — Flask Web Application
============================================
Run: python app.py
Then open: http://localhost:5000
"""

import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/skin_cancer_model.h5"
IMG_SIZE    = 224
MAX_FILE_MB = 10

CLASS_INFO = {
    'nv': {
        'name': 'Melanocytic Nevi',
        'risk': 'Low',
        'color': '#27ae60',
        'description': 'Common moles. Usually benign but monitor for changes in size, shape, or color.',
        'advice': 'Regular self-examination recommended. See a dermatologist annually.'
    },
    'mel': {
        'name': 'Melanoma',
        'risk': 'High',
        'color': '#e74c3c',
        'description': 'Most dangerous skin cancer. Develops from melanocytes. Early detection is crucial.',
        'advice': '⚠️ URGENT: Consult a dermatologist immediately for biopsy and diagnosis.'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'risk': 'Low',
        'color': '#2ecc71',
        'description': 'Non-cancerous skin growth including seborrheic keratoses and solar lentigines.',
        'advice': 'Generally harmless. Removal possible for cosmetic reasons or if irritated.'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'risk': 'Medium',
        'color': '#e67e22',
        'description': 'Most common form of skin cancer. Rarely spreads but can be locally destructive.',
        'advice': 'Schedule appointment with dermatologist within 2-4 weeks for evaluation.'
    },
    'akiec': {
        'name': 'Actinic Keratoses',
        'risk': 'Medium',
        'color': '#f39c12',
        'description': 'Precancerous rough patches caused by UV damage. Can progress to squamous cell carcinoma.',
        'advice': 'Dermatologist visit recommended. Treatable with cryotherapy or topical medications.'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'risk': 'Low',
        'color': '#3498db',
        'description': 'Includes angiomas, angiokeratomas, and pyogenic granulomas. Usually benign.',
        'advice': 'Generally benign. Consult doctor if bleeding or rapidly growing.'
    },
    'df': {
        'name': 'Dermatofibroma',
        'risk': 'Low',
        'color': '#9b59b6',
        'description': 'Benign fibrous skin nodule, usually on legs. Firm and painless.',
        'advice': 'Benign and usually harmless. Removal possible if bothersome.'
    }
}

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print("⚠️  No trained model found. Using demo mode with random predictions.")
        print("    Run train_model.py first to train the real model.")

# ─── PREDICTION ─────────────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image_array: np.ndarray) -> dict:
    class_keys = list(CLASS_INFO.keys())

    if model is None:
        # Demo mode: simulate predictions
        probs = np.random.dirichlet(np.ones(7) * 0.5)
    else:
        probs = model.predict(image_array, verbose=0)[0]

    pred_idx  = int(np.argmax(probs))
    pred_key  = class_keys[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    top3 = sorted(
        [{'key': class_keys[i], 'name': CLASS_INFO[class_keys[i]]['name'],
          'prob': round(float(p) * 100, 1)}
         for i, p in enumerate(probs)],
        key=lambda x: x['prob'], reverse=True
    )[:3]

    return {
        'predicted_class': pred_key,
        'class_info': CLASS_INFO[pred_key],
        'confidence': round(confidence, 1),
        'top3': top3,
        'all_probs': {class_keys[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)}
    }


# ─── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Unsupported file type. Allowed: {allowed}'}), 400

    try:
        img_bytes = file.read()
        if len(img_bytes) > MAX_FILE_MB * 1024 * 1024:
            return jsonify({'error': f'File too large (max {MAX_FILE_MB}MB)'}), 400

        img = Image.open(io.BytesIO(img_bytes))
        img_array = preprocess_image(img)
        result = predict(img_array)

        # Encode thumbnail for display
        img_thumb = img.convert('RGB')
        img_thumb.thumbnail((300, 300))
        buf = io.BytesIO()
        img_thumb.save(buf, format='JPEG', quality=85)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()

        result['thumbnail'] = f"data:image/jpeg;base64,{thumb_b64}"
        result['demo_mode'] = model is None
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/classes')
def get_classes():
    return jsonify(CLASS_INFO)


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    print("\n🚀 Starting Skin Cancer Detection Web App")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

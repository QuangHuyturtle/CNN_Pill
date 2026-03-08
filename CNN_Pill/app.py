"""
Flask Web App for Pill Image Classification
Simple local web interface for demo/thesis purposes
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from inference import PillClassifier, EnsemblePillClassifier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
# Single model checkpoint path (optional, use ENSEMBLE_PATHS for ensemble)
app.config['CHECKPOINT_PATH'] = 'checkpoints/run_20260308_083313/best_fold0.pth'
# Ensemble checkpoint paths (comma-separated)
app.config['ENSEMBLE_PATHS'] = None
app.config['ENCODER_PATH'] = 'data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/label_encoder.pickle'

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once at startup
print("Loading model...")

# Use ensemble if paths specified, otherwise use single model
if app.config['ENSEMBLE_PATHS']:
    checkpoint_paths = [p.strip() for p in app.config['ENSEMBLE_PATHS'].split(',')]
    print(f"ENSEMBLE MODE: Loading {len(checkpoint_paths)} models")
    for path in checkpoint_paths:
        print(f"  - {path}")
    classifier = EnsemblePillClassifier(
        checkpoint_paths=checkpoint_paths,
        label_encoder_path=app.config['ENCODER_PATH']
    )
else:
    print(f"SINGLE MODEL MODE: Loading {app.config['CHECKPOINT_PATH']}")
    classifier = PillClassifier(
        checkpoint_path=app.config['CHECKPOINT_PATH'],
        label_encoder_path=app.config['ENCODER_PATH']
    )
print("Model loaded successfully!")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tif', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, gif, tif, webp'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        predictions = classifier.predict(filepath, top_k=5)

        # Format results
        results = {
            'filename': filename,
            'predictions': [
                {
                    'rank': pred['rank'],
                    'label': pred['label'],
                    'confidence': f"{pred['confidence']:.2f}%"
                }
                for pred in predictions
            ]
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PILL CLASSIFICATION WEB APP")
    print("="*60)
    print(f"Open browser at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)

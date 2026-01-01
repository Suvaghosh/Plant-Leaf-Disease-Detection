"""
Plant Leaf Disease Detection - Flask Web Application
====================================================
This is the main Flask application that provides a web interface
for uploading leaf images and getting disease predictions.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from predict import PlantDiseasePredictor
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize predictor (lazy loading - will load when first prediction is made)
predictor = None

def get_predictor():
    """Get or initialize the predictor (singleton pattern)."""
    global predictor
    if predictor is None:
        try:
            predictor = PlantDiseasePredictor()
        except Exception as e:
            print(f"Error loading predictor: {str(e)}")
            return None
    return predictor

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and prediction.
    """
    # Check if file is present in request
    if 'file' not in request.files:
        flash('No file selected. Please choose an image file.')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        flash('No file selected. Please choose an image file.')
        return redirect(url_for('index'))
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, BMP).')
        return redirect(url_for('index'))
    
    try:
        # Read file content
        file_content = file.read()
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            flash('File is too large. Maximum size is 10 MB.')
            return redirect(url_for('index'))
        
        # Get predictor
        pred = get_predictor()
        if pred is None:
            flash('Model not available. Please ensure the model is trained first.')
            return redirect(url_for('index'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save file
        with open(filepath, 'wb') as f:
            f.write(file_content)
        
        # Make prediction
        file_stream = io.BytesIO(file_content)
        result = pred.predict(file_stream, top_k=3)
        
        # Prepare response data
        response_data = {
            'success': True,
            'predicted_class': result['predicted_class'],
            'confidence': round(result['confidence'] * 100, 2),
            'image_url': f'/static/uploads/{filename}',
            'all_predictions': result['all_predictions']
        }
        
        # Render result page
        return render_template('result.html', **response_data)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions (returns JSON).
    Useful for integration with other applications.
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    try:
        # Read file content
        file_content = file.read()
        
        if len(file_content) > MAX_FILE_SIZE:
            return jsonify({'success': False, 'error': 'File too large'}), 400
        
        # Get predictor
        pred = get_predictor()
        if pred is None:
            return jsonify({'success': False, 'error': 'Model not available'}), 500
        
        # Make prediction
        file_stream = io.BytesIO(file_content)
        result = pred.predict(file_stream, top_k=3)
        
        return jsonify({
            'success': True,
            'predicted_class': result['predicted_class'],
            'confidence': round(result['confidence'] * 100, 2),
            'all_predictions': result['all_predictions']
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    pred = get_predictor()
    if pred is None:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 503
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Plant Leaf Disease Detection - Web Application")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)


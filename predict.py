"""
Plant Leaf Disease Detection - Prediction Utility
==================================================
This script provides utility functions for making predictions on leaf images.
Used by both the training script and the Flask web application.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PIL import Image
import io

# Configuration
MODEL_PATH = "model/plant_disease_model.h5"
CLASS_NAMES_PATH = "model/class_names.txt"
IMG_SIZE = (224, 224)

class PlantDiseasePredictor:
    """
    Class to handle plant disease prediction from images.
    """
    
    def __init__(self, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        """
        Initialize the predictor by loading the model and class names.
        
        Args:
            model_path: Path to the saved model file
            class_names_path: Path to the class names text file
        """
        self.model = None
        self.class_names = []
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.load_model()
        self.load_class_names()
    
    def load_model(self):
        """Load the trained model from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Please train the model first using train.py"
            )
        
        print(f"Loading model from {self.model_path}...")
        self.model = load_model(self.model_path)
        print("Model loaded successfully!")
    
    def load_class_names(self):
        """Load class names from file."""
        if not os.path.exists(self.class_names_path):
            raise FileNotFoundError(
                f"Class names file not found: {self.class_names_path}\n"
                "Please train the model first using train.py"
            )
        
        with open(self.class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.class_names)} classes")
    
    def preprocess_image(self, image_input):
        """
        Preprocess image for model prediction.
        
        Args:
            image_input: Can be:
                - File path (string)
                - PIL Image object
                - BytesIO object
                - numpy array
        
        Returns:
            Preprocessed image array ready for model prediction
        """
        # Handle different input types
        if isinstance(image_input, str):
            # File path
            img = image.load_img(image_input, target_size=IMG_SIZE)
        elif isinstance(image_input, Image.Image):
            # PIL Image
            img = image_input.resize(IMG_SIZE)
        elif isinstance(image_input, io.BytesIO):
            # BytesIO (from file upload)
            img = Image.open(image_input)
            img = img.resize(IMG_SIZE)
        elif isinstance(image_input, bytes):
            # Raw bytes
            img = Image.open(io.BytesIO(image_input))
            img = img.resize(IMG_SIZE)
        else:
            raise ValueError("Unsupported image input type")
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        return img_array
    
    def predict(self, image_input, top_k=3):
        """
        Predict disease from leaf image.
        
        Args:
            image_input: Image input (path, PIL Image, BytesIO, or bytes)
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary containing:
                - predicted_class: Name of the predicted class
                - confidence: Confidence score (0-1)
                - all_predictions: List of top_k predictions with scores
        """
        # Preprocess image
        processed_image = self.preprocess_image(image_input)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        predicted_class = self.class_names[predicted_index]
        
        # Get top k predictions
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        all_predictions = [
            {
                'class': self.class_names[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_indices
        ]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        }
    
    def predict_from_file(self, file_path, top_k=3):
        """
        Convenience method to predict from a file path.
        
        Args:
            file_path: Path to the image file
            top_k: Number of top predictions to return
        
        Returns:
            Prediction dictionary
        """
        return self.predict(file_path, top_k)

# Example usage
if __name__ == "__main__":
    # Example: Predict from a test image
    predictor = PlantDiseasePredictor()
    
    # You can test with an image path
    # test_image_path = "path/to/test/image.jpg"
    # result = predictor.predict_from_file(test_image_path)
    # print(f"Predicted: {result['predicted_class']}")
    # print(f"Confidence: {result['confidence']:.2%}")
    
    print("Predictor initialized successfully!")
    print("Use this class in your Flask application for predictions.")


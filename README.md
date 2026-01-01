# Plant Leaf Disease Detection and Classification Using Deep Learning

A complete end-to-end deep learning system for detecting and classifying plant leaf diseases from images. This project uses Convolutional Neural Networks (CNN) with transfer learning to provide accurate disease classification through a user-friendly web interface.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [System Architecture](#system-architecture)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Running the Web Application](#running-the-web-application)
- [API Endpoints](#api-endpoints)
- [Results and Evaluation](#results-and-evaluation)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🌿 Project Overview

This project implements a deep learning-based solution for automated plant leaf disease detection. The system can:

- **Classify plant leaf diseases** from uploaded images
- **Provide confidence scores** for predictions
- **Display top predictions** for better understanding
- **Offer a simple web interface** for easy use

The model uses **MobileNetV2** (a lightweight CNN architecture) with transfer learning, making it both accurate and efficient for deployment.

## 🎯 Problem Statement

Plant diseases are a major threat to agricultural productivity worldwide. Early detection and accurate diagnosis of plant diseases are crucial for:

- **Preventing crop loss**: Early detection allows for timely treatment
- **Reducing pesticide use**: Targeted treatment instead of broad-spectrum applications
- **Improving yield**: Healthy plants produce better yields
- **Cost savings**: Preventing disease spread saves money

Manual disease detection requires expert knowledge and is time-consuming. This project automates the process using deep learning, making disease detection accessible to farmers and agricultural workers.

## 📊 Dataset Description

The dataset should be organized in the following structure:

```
Dataset/
├── class_name_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_name_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

**Dataset Location**: `C:\Users\subha\OneDrive\Desktop\Dataset`

**Key Points**:
- Each folder represents a disease class (or healthy class)
- Images should be in common formats (JPG, PNG, etc.)
- The training script automatically handles train/validation split (80/20)
- Data augmentation is applied during training to improve generalization

## 🏗️ System Architecture

The system consists of three main components:

### 1. **Data Processing Pipeline**
   - Image loading and preprocessing
   - Data augmentation (rotation, flipping, zooming, etc.)
   - Train/validation split
   - Normalization (pixel values scaled to [0, 1])

### 2. **Deep Learning Model**
   - Base: MobileNetV2 (pre-trained on ImageNet)
   - Custom classification head
   - Transfer learning approach
   - Optimized for accuracy and speed

### 3. **Web Application**
   - Flask backend for serving predictions
   - HTML/CSS frontend for user interaction
   - Image upload and display
   - Results visualization

```
User Uploads Image
       ↓
Flask Web App (app.py)
       ↓
Preprocessing (predict.py)
       ↓
Trained CNN Model
       ↓
Prediction Results
       ↓
Display to User
```

## 🧠 Model Architecture

The model uses **MobileNetV2** as the base architecture with the following structure:

```
Input Layer (224x224x3)
       ↓
MobileNetV2 Base (Pre-trained, frozen)
       ↓
Global Average Pooling 2D
       ↓
Dropout (0.3)
       ↓
Dense Layer (512 units, ReLU)
       ↓
Dropout (0.3)
       ↓
Output Layer (N classes, Softmax)
```

**Why MobileNetV2?**
- Lightweight and efficient
- Fast inference time
- Good accuracy for image classification
- Suitable for web deployment
- Pre-trained on ImageNet (transfer learning)

**Training Details**:
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)
- **Image Size**: 224x224 pixels
- **Data Augmentation**: Yes (rotation, shift, zoom, flip)

## 📁 Project Structure

```
minor project/
│
├── model/                          # Model storage directory
│   ├── plant_disease_model.h5      # Trained model (generated after training)
│   ├── class_names.txt             # Class names (generated after training)
│   ├── confusion_matrix.png        # Confusion matrix plot (generated)
│   └── training_history.png        # Training curves (generated)
│
├── static/                         # Static files (CSS, uploaded images)
│   ├── style.css                   # Stylesheet for web UI
│   └── uploads/                    # Uploaded images (created automatically)
│
├── templates/                      # HTML templates
│   ├── index.html                  # Main upload page
│   └── result.html                 # Results display page
│
├── app.py                          # Flask web application
├── train.py                        # Model training script
├── predict.py                     # Prediction utility module
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Dataset folder with images organized by class

### Step 1: Clone or Download the Project

Navigate to your project directory:
```bash
cd "C:\Users\subha\OneDrive\Desktop\minor project"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (for deep learning)
- Flask (for web framework)
- Pillow (for image processing)
- NumPy, Matplotlib, Seaborn (for data processing and visualization)
- scikit-learn (for evaluation metrics)

### Step 4: Verify Dataset Path

Ensure your dataset is located at:
```
C:\Users\subha\OneDrive\Desktop\Dataset
```

If your dataset is in a different location, update the `DATASET_PATH` variable in `train.py`.

## 📖 Usage

### Training the Model

1. **Prepare your dataset** in the required folder structure (see Dataset Description)

2. **Run the training script**:
   ```bash
   python train.py
   ```

3. **Training process**:
   - The script will automatically:
     - Load and preprocess images
     - Split data into training/validation sets (80/20)
     - Build the model architecture
     - Train the model with data augmentation
     - Save the best model to `model/plant_disease_model.h5`
     - Generate evaluation metrics and plots

4. **Training outputs**:
   - `model/plant_disease_model.h5`: Trained model file
   - `model/class_names.txt`: List of class names
   - `model/confusion_matrix.png`: Confusion matrix visualization
   - `model/training_history.png`: Training accuracy and loss curves

5. **Monitor training**:
   - Watch the console for training progress
   - Early stopping will prevent overfitting
   - Best model is saved based on validation accuracy

### Running the Web Application

1. **Ensure the model is trained** (see Training the Model above)

2. **Start the Flask server**:
   ```bash
   python app.py
   ```

3. **Open your web browser** and navigate to:
   ```
   http://127.0.0.1:5000
   ```

4. **Use the application**:
   - Click "Select Image" or drag and drop a leaf image
   - Click "Analyze Leaf Image"
   - View the prediction results with confidence scores

5. **Stop the server**: Press `CTRL+C` in the terminal

## 🔌 API Endpoints

The Flask application provides the following endpoints:

### 1. **GET /** 
   - Main page with image upload form
   - Returns: HTML page

### 2. **POST /predict**
   - Upload image and get prediction
   - Accepts: multipart/form-data with 'file' field
   - Returns: HTML page with results

### 3. **POST /api/predict**
   - API endpoint for programmatic access
   - Accepts: multipart/form-data with 'file' field
   - Returns: JSON response
   ```json
   {
     "success": true,
     "predicted_class": "Disease Name",
     "confidence": 95.23,
     "all_predictions": [
       {"class": "Disease 1", "confidence": 95.23},
       {"class": "Disease 2", "confidence": 3.45},
       {"class": "Disease 3", "confidence": 1.32}
     ]
   }
   ```

### 4. **GET /health**
   - Health check endpoint
   - Returns: JSON with system status
   ```json
   {
     "status": "healthy",
     "model_loaded": true
   }
   ```

## 📈 Results and Evaluation

After training, the model generates:

1. **Classification Report**: Precision, recall, and F1-score for each class
2. **Confusion Matrix**: Visual representation of prediction accuracy
3. **Training Curves**: Accuracy and loss over epochs

**Metrics to Monitor**:
- **Training Accuracy**: Should increase over epochs
- **Validation Accuracy**: Should track training accuracy (watch for overfitting)
- **Validation Loss**: Should decrease (watch for divergence from training loss)

**Tips for Better Results**:
- Use a balanced dataset (similar number of images per class)
- Ensure high-quality, clear images
- Include diverse lighting conditions and angles
- Train for more epochs if validation accuracy is still improving
- Consider unfreezing base model layers for fine-tuning

## 🔮 Future Improvements

1. **Model Enhancements**:
   - Fine-tuning (unfreeze base model layers)
   - Experiment with other architectures (ResNet, EfficientNet)
   - Ensemble methods for better accuracy
   - Multi-label classification (multiple diseases)

2. **Dataset Improvements**:
   - Data collection from multiple sources
   - Synthetic data generation (GANs)
   - Better class balance
   - More diverse environmental conditions

3. **Application Features**:
   - User authentication and history
   - Batch image processing
   - Treatment recommendations
   - Disease information database
   - Mobile app version
   - Real-time camera integration

4. **Deployment**:
   - Cloud deployment (AWS, Google Cloud, Azure)
   - Docker containerization
   - RESTful API for integration
   - Model versioning and A/B testing

5. **Performance**:
   - Model quantization for faster inference
   - Edge device deployment (Raspberry Pi, mobile)
   - Caching mechanisms
   - Load balancing for high traffic

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for contribution:

- Model improvements
- UI/UX enhancements
- Documentation updates
- Bug fixes
- Feature additions

## 📝 License

This project is open source and available for educational and research purposes.

## 👨‍💻 Author

Created as a Minor Project for academic purposes.

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- Flask team for the web framework
- MobileNetV2 architecture by Google
- All contributors to open-source libraries used in this project

---

**Note**: This is an educational project. For production use in agriculture, please consult with agricultural experts and validate results with field testing.


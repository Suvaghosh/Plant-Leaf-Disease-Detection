# System Architecture Overview

## Plant Leaf Disease Detection System - Architecture Documentation

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Web Browser (HTML/CSS/JavaScript)                   │   │
│  │  - Image Upload Interface                            │   │
│  │  - Results Display                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP/HTTPS
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Flask Web Server (app.py)                           │   │
│  │  - Route Handling                                    │   │
│  │  - File Upload Management                            │   │
│  │  - Request/Response Processing                       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                    Prediction Layer                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Prediction Module (predict.py)                      │   │
│  │  - Image Preprocessing                               │   │
│  │  - Model Loading                                     │   │
│  │  - Inference                                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                    Deep Learning Layer                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Trained CNN Model (MobileNetV2)                     │   │
│  │  - Feature Extraction                                │   │
│  │  - Classification                                    │   │
│  │  - Output: Disease Class + Confidence               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Dataset (Training Images)                           │   │
│  │  - Organized by Class                                │   │
│  │  - Preprocessed & Augmented                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. **Training Pipeline** (`train.py`)

**Purpose**: Train the deep learning model on the dataset

**Workflow**:
```
Dataset Loading
    ↓
Data Preprocessing & Augmentation
    ↓
Train/Validation Split (80/20)
    ↓
Model Architecture Building (MobileNetV2 + Custom Head)
    ↓
Model Training with Callbacks
    ↓
Model Evaluation & Metrics
    ↓
Model Saving (.h5 format)
```

**Key Features**:
- Automatic data loading from class-based folders
- Data augmentation (rotation, flip, zoom, shift)
- Transfer learning with MobileNetV2
- Early stopping to prevent overfitting
- Model checkpointing (saves best model)
- Comprehensive evaluation metrics

#### 2. **Prediction Module** (`predict.py`)

**Purpose**: Provide prediction functionality for uploaded images

**Class**: `PlantDiseasePredictor`

**Methods**:
- `load_model()`: Load trained model from disk
- `load_class_names()`: Load class labels
- `preprocess_image()`: Prepare image for model input
- `predict()`: Make prediction on image

**Image Preprocessing Steps**:
1. Load image (supports multiple formats)
2. Resize to 224x224 pixels
3. Convert to RGB if needed
4. Normalize pixel values to [0, 1]
5. Add batch dimension

#### 3. **Web Application** (`app.py`)

**Purpose**: Provide web interface for users

**Routes**:
- `GET /`: Main upload page
- `POST /predict`: Handle image upload and return results page
- `POST /api/predict`: JSON API endpoint
- `GET /health`: Health check endpoint

**Features**:
- File upload handling
- File validation (type, size)
- Error handling and user feedback
- Image storage in `static/uploads/`

#### 4. **Frontend** (`templates/` + `static/`)

**Components**:
- `index.html`: Upload interface
- `result.html`: Results display
- `style.css`: Responsive styling

**Features**:
- Drag-and-drop file upload
- Image preview
- Responsive design
- Clean, modern UI
- Confidence visualization

### Data Flow

#### Training Phase:
```
Raw Images → Preprocessing → Augmentation → Model Training → Saved Model
```

#### Inference Phase:
```
User Upload → Flask App → Preprocessing → Model Inference → Results → User
```

### Model Architecture Details

**Base Model**: MobileNetV2
- Pre-trained on ImageNet
- Input: 224x224x3 RGB images
- Output: Feature maps

**Custom Classification Head**:
```
Global Average Pooling 2D
    ↓
Dropout (0.3)
    ↓
Dense (512 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (N classes, Softmax)
```

**Why This Architecture?**:
1. **MobileNetV2**: Lightweight, efficient, good accuracy
2. **Transfer Learning**: Leverages pre-trained features
3. **Global Average Pooling**: Reduces parameters, prevents overfitting
4. **Dropout Layers**: Regularization
5. **Softmax Output**: Probability distribution over classes

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Deep Learning | TensorFlow/Keras | Model training and inference |
| Base Model | MobileNetV2 | Feature extraction |
| Web Framework | Flask | Backend server |
| Frontend | HTML/CSS/JavaScript | User interface |
| Image Processing | Pillow | Image manipulation |
| Data Science | NumPy, Matplotlib | Data handling and visualization |

### File Organization

```
Project Root/
├── train.py              # Training pipeline
├── predict.py            # Prediction utilities
├── app.py                # Flask web application
├── verify_setup.py      # Setup verification
├── requirements.txt      # Dependencies
├── README.md            # Main documentation
├── ARCHITECTURE.md      # This file
├── .gitignore          # Git ignore rules
│
├── model/              # Model storage
│   ├── plant_disease_model.h5
│   ├── class_names.txt
│   └── [generated plots]
│
├── static/            # Static files
│   ├── style.css
│   └── uploads/       # User uploads
│
└── templates/         # HTML templates
    ├── index.html
    └── result.html
```

### Security Considerations

1. **File Upload Validation**:
   - File type checking
   - File size limits (10MB)
   - Secure filename handling

2. **Error Handling**:
   - Try-catch blocks
   - User-friendly error messages
   - Graceful degradation

3. **Model Loading**:
   - Lazy loading (only when needed)
   - Error handling for missing model

### Performance Optimizations

1. **Model**:
   - Lightweight MobileNetV2 architecture
   - Efficient inference
   - Batch processing capability

2. **Web Application**:
   - Singleton predictor instance
   - Image preprocessing optimization
   - Static file serving

3. **Training**:
   - Data augmentation for better generalization
   - Early stopping to save time
   - Model checkpointing

### Scalability Considerations

**Current Limitations**:
- Single-threaded Flask server (development mode)
- No database for storing predictions
- No user authentication

**Future Scalability Options**:
- Deploy with Gunicorn/uWSGI (production server)
- Add database (PostgreSQL, MongoDB)
- Implement caching (Redis)
- Load balancing for multiple instances
- Containerization (Docker)
- Cloud deployment (AWS, GCP, Azure)

### Deployment Architecture (Future)

```
┌─────────────┐
│   CDN       │ (Static files)
└─────────────┘
      ↓
┌─────────────┐
│ Load Balancer│
└─────────────┘
      ↓
┌─────────────┐     ┌─────────────┐
│ Flask App 1 │ ... │ Flask App N │
└─────────────┘     └─────────────┘
      ↓                   ↓
┌─────────────┐     ┌─────────────┐
│   Redis     │     │  Database   │
│  (Cache)    │     │ (PostgreSQL)│
└─────────────┘     └─────────────┘
```

---

**Last Updated**: 2025
**Version**: 1.0


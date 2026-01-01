"""
Plant Leaf Disease Detection - Model Training Script
====================================================
This script handles:
1. Data loading and preprocessing
2. Model architecture (using MobileNetV2 transfer learning)
3. Model training
4. Model evaluation and saving
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATASET_PATH = r"C:\Users\subha\OneDrive\Desktop\Dataset"
MODEL_SAVE_PATH = "model/plant_disease_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.0001

def load_and_preprocess_data():
    """
    Load images from dataset directory and create train/validation generators.
    Assumes dataset structure: Dataset/class_name/image.jpg
    """
    print("=" * 60)
    print("Loading and Preprocessing Data")
    print("=" * 60)
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")
    
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
        validation_split=VALIDATION_SPLIT,
        rotation_range=20,  # Random rotation
        width_shift_range=0.2,  # Random horizontal shift
        height_shift_range=0.2,  # Random vertical shift
        shear_range=0.2,  # Random shear
        zoom_range=0.2,  # Random zoom
        horizontal_flip=True,  # Random horizontal flip
        fill_mode='nearest'  # Fill strategy for transformations
    )
    
    # Only rescaling for validation set (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=VALIDATION_SPLIT
    )
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Create validation generator
    val_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get class names and number of classes
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    
    print(f"\nFound {num_classes} classes:")
    for i, class_name in enumerate(class_names):
        print(f"  {i}: {class_name}")
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Save class names to file for later use
    with open("model/class_names.txt", "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    return train_generator, val_generator, num_classes, class_names

def build_model(num_classes):
    """
    Build CNN model using MobileNetV2 transfer learning.
    MobileNetV2 is lightweight and efficient for mobile/web deployment.
    """
    print("\n" + "=" * 60)
    print("Building Model Architecture")
    print("=" * 60)
    
    # Load pre-trained MobileNetV2 model (without top layer)
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially (optional - can unfreeze later)
    base_model.trainable = False
    
    # Build the complete model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Global average pooling
        layers.Dropout(0.3),  # Dropout for regularization
        layers.Dense(512, activation='relu'),  # Dense layer
        layers.Dropout(0.3),  # Another dropout layer
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    return model

def train_model(model, train_generator, val_generator):
    """
    Train the model with callbacks for best model saving and early stopping.
    """
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Callbacks
    callbacks = [
        # Save the best model based on validation accuracy
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, val_generator, class_names):
    """
    Evaluate the model and generate classification report and confusion matrix.
    """
    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)
    
    # Get predictions
    predictions = model.predict(val_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved to model/confusion_matrix.png")
    
    # Calculate accuracy
    accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

def plot_training_history(history):
    """
    Plot training history (accuracy and loss curves).
    """
    print("\n" + "=" * 60)
    print("Plotting Training History")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved to model/training_history.png")

def main():
    """
    Main function to orchestrate the training process.
    """
    print("\n" + "=" * 60)
    print("PLANT LEAF DISEASE DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Create model directory at the start to ensure it exists
    os.makedirs("model", exist_ok=True)
    
    try:
        # Step 1: Load and preprocess data
        train_generator, val_generator, num_classes, class_names = load_and_preprocess_data()
        
        # Step 2: Build model
        model = build_model(num_classes)
        
        # Step 3: Train model
        history = train_model(model, train_generator, val_generator)
        
        # Step 4: Evaluate model
        evaluate_model(model, val_generator, class_names)
        
        # Step 5: Plot training history
        plot_training_history(history)
        
        print("\n" + "=" * 60)
        print("Training Completed Successfully!")
        print(f"Model saved to: {MODEL_SAVE_PATH}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


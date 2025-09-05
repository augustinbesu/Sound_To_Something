# -*- coding: utf-8 -*-
"""
Music Genre Classifier Trainer

This script trains a neural network to classify audio files
using the GTZAN dataset with 10 musical genres.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import pickle
import time
import shutil
import warnings

# Suppress specific librosa warnings
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", message="Deprecated as of librosa version")

# Configuration
SAMPLE_RATE = 22050  # Standard sampling rate
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "early_stopping_patience": 15,
    "early_stopping_enabled": True,  # New parameter
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "min_lr": 0.00001,
    "dropout_rate": 0.3,
    "validation_split": 0.2
}

def get_hyperparameters():
    """
    Allows user to configure hyperparameters or use defaults.
    
    Returns:
        dict: Hyperparameters configuration
    """
    print("\n=== Model Hyperparameters Configuration ===")
    print("Press Enter to use default values, or enter custom values:")
    
    hyperparams = DEFAULT_HYPERPARAMS.copy()
    
    # Training parameters
    print(f"\nTraining Parameters:")
    epochs = input(f"Epochs [{DEFAULT_HYPERPARAMS['epochs']}]: ").strip()
    if epochs:
        try:
            hyperparams["epochs"] = int(epochs)
        except ValueError:
            print("Invalid input, using default.")
    
    batch_size = input(f"Batch size [{DEFAULT_HYPERPARAMS['batch_size']}]: ").strip()
    if batch_size:
        try:
            hyperparams["batch_size"] = int(batch_size)
        except ValueError:
            print("Invalid input, using default.")
    
    learning_rate = input(f"Learning rate [{DEFAULT_HYPERPARAMS['learning_rate']}]: ").strip()
    if learning_rate:
        try:
            hyperparams["learning_rate"] = float(learning_rate)
        except ValueError:
            print("Invalid input, using default.")
    
    validation_split = input(f"Validation split [{DEFAULT_HYPERPARAMS['validation_split']}]: ").strip()
    if validation_split:
        try:
            val_split = float(validation_split)
            if 0.1 <= val_split <= 0.5:
                hyperparams["validation_split"] = val_split
            else:
                print("Validation split should be between 0.1 and 0.5, using default.")
        except ValueError:
            print("Invalid input, using default.")
    
    # Advanced parameters
    print(f"\nAdvanced Parameters:")
    advanced = input("Configure advanced parameters? (y/n) [n]: ").strip().lower()
    
    if advanced in ['y', 'yes', 's', 'si', 'sí']:
        dropout_rate = input(f"Dropout rate [{DEFAULT_HYPERPARAMS['dropout_rate']}]: ").strip()
        if dropout_rate:
            try:
                rate = float(dropout_rate)
                if 0.0 <= rate <= 0.8:
                    hyperparams["dropout_rate"] = rate
                else:
                    print("Dropout rate should be between 0.0 and 0.8, using default.")
            except ValueError:
                print("Invalid input, using default.")
        
        # Early stopping configuration
        early_stopping = input(f"Enable early stopping? (y/n) [y]: ").strip().lower() or "y"
        if early_stopping in ['n', 'no']:
            hyperparams["early_stopping_enabled"] = False
            print("Early stopping disabled - will train for all epochs.")
        else:
            hyperparams["early_stopping_enabled"] = True
            early_stopping_patience = input(f"Early stopping patience [{DEFAULT_HYPERPARAMS['early_stopping_patience']}]: ").strip()
            if early_stopping_patience:
                try:
                    hyperparams["early_stopping_patience"] = int(early_stopping_patience)
                except ValueError:
                    print("Invalid input, using default.")
        
        reduce_lr_patience = input(f"Reduce LR patience [{DEFAULT_HYPERPARAMS['reduce_lr_patience']}]: ").strip()
        if reduce_lr_patience:
            try:
                hyperparams["reduce_lr_patience"] = int(reduce_lr_patience)
            except ValueError:
                print("Invalid input, using default.")
        
        reduce_lr_factor = input(f"Reduce LR factor [{DEFAULT_HYPERPARAMS['reduce_lr_factor']}]: ").strip()
        if reduce_lr_factor:
            try:
                factor = float(reduce_lr_factor)
                if 0.1 <= factor <= 0.9:
                    hyperparams["reduce_lr_factor"] = factor
                else:
                    print("Factor should be between 0.1 and 0.9, using default.")
            except ValueError:
                print("Invalid input, using default.")
    
    # Show final configuration
    print(f"\nFinal Hyperparameters Configuration:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    confirm = input(f"\nContinue with these parameters? (y/n) [y]: ").strip().lower() or "y"
    if confirm not in ['y', 'yes', 's', 'si', 'sí']:
        print("Configuration cancelled.")
        return None
    
    return hyperparams

def create_data_directory():
    """Creates only the necessary directory structure"""
    os.makedirs("models", exist_ok=True)
    print("Models directory created.")

def use_existing_dataset(source_dir):
    """
    Uses an existing dataset like GTZAN.
    
    Args:
        source_dir (str): Directory where the GTZAN dataset is located
    """
    print(f"Using existing dataset at: {source_dir}")
    
    # Check if directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Directory {source_dir} does not exist.")
        return False
    
    # Check if it contains expected genres
    missing_genres = []
    for genre in GENRES:
        genre_dir = os.path.join(source_dir, genre)
        if not os.path.exists(genre_dir):
            missing_genres.append(genre)
    
    if missing_genres:
        print(f"Error: Missing the following genres in the dataset: {', '.join(missing_genres)}")
        return False
    
    # Create models directory (the only one we actually need)
    os.makedirs("models", exist_ok=True)
    print("Models directory created.")
    
    return True

def extract_features(file_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extracts MFCC features from a complete audio file.
    
    Args:
        file_path (str): Path to the audio file
        num_mfcc (int): Number of MFCC coefficients to extract
        n_fft (int): Length of the FFT window
        hop_length (int): Number of samples between successive frames
        
    Returns:
        numpy.ndarray: Averaged MFCC features
    """
    try:
        # Load complete audio file with warnings suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Extract MFCC from entire song
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=num_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Calculate statistics to get a fixed feature vector
        # We use mean and standard deviation of each MFCC coefficient
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Concatenate to form the feature vector
        features = np.concatenate((mfcc_mean, mfcc_std))
        
        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def prepare_dataset(dataset_dir):
    """
    Prepares the dataset by extracting features from audio files.
    
    Args:
        dataset_dir (str): Directory where the dataset is located
        
    Returns:
        tuple: (X, y) with features and labels
    """
    # Check if there are files in the folders
    total_files = 0
    for genre in GENRES:
        genre_path = os.path.join(dataset_dir, genre)
        if os.path.exists(genre_path):
            files = [f for f in os.listdir(genre_path) if f.endswith(('.wav', '.mp3', '.au'))]
            total_files += len(files)
            print(f"Found {len(files)} files in {genre}")
    
    if total_files == 0:
        print("No audio files found. Check the dataset.")
        return None, None
    
    # Extract features
    print(f"\nExtracting features from {total_files} complete files...")
    
    features = []
    labels = []
    
    for i, genre in enumerate(GENRES):
        genre_path = os.path.join(dataset_dir, genre)
        if not os.path.exists(genre_path):
            continue
            
        print(f"Processing genre: {genre}")
        
        genre_files = [f for f in os.listdir(genre_path) if f.endswith(('.wav', '.mp3', '.au'))]
        
        for j, file_name in enumerate(genre_files):
            file_path = os.path.join(genre_path, file_name)
            
            # Show progress
            print(f"  Processing {j+1}/{len(genre_files)}: {file_name}", end="\r")
            
            # Extract features
            song_features = extract_features(file_path)
            
            if song_features is not None:
                # Add features and labels
                features.append(song_features)
                labels.append(i)
        
        print()  # New line after each genre
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y

def build_model(input_shape, num_classes, hyperparams):
    """
    Builds the model for audio classification.
    
    Args:
        input_shape (tuple): Shape of input data
        num_classes (int): Number of classes to classify
        hyperparams (dict): Hyperparameters configuration
        
    Returns:
        model: Keras model
    """
    # Create sequential model
    model = models.Sequential()
    
    # Dense layers for statistical features
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hyperparams["dropout_rate"]))
    
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hyperparams["dropout_rate"]))
    
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hyperparams["dropout_rate"]))
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hyperparams["dropout_rate"]))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile model with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X, y, hyperparams, model_name):
    """
    Trains the model with the provided data.
    
    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray): Labels
        hyperparams (dict): Hyperparameters configuration
        model_name (str): Name of the model (for saving plots)
        
    Returns:
        tuple: (model, history, encoder, test_accuracy) - Trained model, history, encoder and final accuracy
    """
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, 
        test_size=hyperparams["validation_split"], 
        random_state=42
    )
    
    input_shape = (X_train.shape[1],)
    model = build_model(input_shape, len(GENRES), hyperparams)
    
    # Model summary
    model.summary()
    
    # Create model directory for saving plots
    model_dir = f"models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Train model
    print(f"\nTraining model with hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print()
    
    # Build callbacks list conditionally
    callbacks = []
    
    # Always add ReduceLROnPlateau
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=hyperparams["reduce_lr_factor"],
            patience=hyperparams["reduce_lr_patience"],
            min_lr=hyperparams["min_lr"],
            verbose=1
        )
    )
    
    # Add Early Stopping only if enabled
    if hyperparams.get("early_stopping_enabled", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=hyperparams["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1
            )
        )
        print(f"Early stopping enabled with patience: {hyperparams['early_stopping_patience']}")
    else:
        print("Early stopping disabled - training for all epochs")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAccuracy on test set: {test_acc:.4f}")
    
    # Predictions for classification report
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=GENRES))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(GENRES))
    plt.xticks(tick_marks, GENRES, rotation=45)
    plt.yticks(tick_marks, GENRES)
    
    # Add values to matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix in model directory
    confusion_matrix_path = os.path.join(model_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Training plots - ONLY accuracy and loss
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title(f'Model Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title(f'Model Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save training history in model directory
    training_history_path = os.path.join(model_dir, 'training_history.png')
    plt.savefig(training_history_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, history, encoder, test_acc 

def save_model(model, encoder, hyperparams, model_name=None, test_accuracy=None):
    """
    Saves the trained model and label encoder with versioning support.
    
    Args:
        model: Trained model
        encoder: Label encoder
        hyperparams (dict): Hyperparameters used for training
        model_name (str): Optional custom name for the model
        test_accuracy (float): Final test accuracy to save
    """
    # Generate model name if not provided
    if model_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = f"model_{timestamp}"
    
    # Create model-specific directory
    model_dir = f"models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = f"{model_dir}/music_genre_classifier.h5"
    model.save(model_path)
    
    # Save encoder
    encoder_path = f"{model_dir}/label_encoder.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    
    # Save preprocessing parameters (including hyperparameters)
    params = {
        "sample_rate": SAMPLE_RATE,
        "num_mfcc": 13,
        "n_fft": 2048,
        "hop_length": 512,
        "genres": GENRES,
        "model_name": model_name,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": hyperparams
    }
    
    params_path = f"{model_dir}/preprocessing_params.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    
    # Save model info as JSON for easy reading (including hyperparameters and accuracy)
    import json
    info = {
        "model_name": model_name,
        "training_date": params["training_date"],
        "genres": GENRES,
        "sample_rate": SAMPLE_RATE,
        "architecture": "Dense Neural Network",
        "input_features": "MFCC statistics (mean + std)",
        "hyperparameters": hyperparams,
        "test_accuracy": round(test_accuracy, 4) if test_accuracy else None 
    }
    
    info_path = f"{model_dir}/model_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\nModel saved to '{model_dir}/'")
    print(f"  - Model: {model_path}")
    print(f"  - Encoder: {encoder_path}")
    print(f"  - Parameters: {params_path}")
    print(f"  - Info: {info_path}")
    print(f"  - Confusion Matrix: {model_dir}/confusion_matrix.png")
    print(f"  - Training History: {model_dir}/training_history.png")
    
    return model_dir

def list_available_models():
    """Lists all available trained models"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found.")
        return []
    
    model_folders = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            model_file = os.path.join(item_path, "music_genre_classifier.h5")
            if os.path.exists(model_file):
                model_folders.append(item)
    
    if model_folders:
        print("\nAvailable models:")
        for i, model_name in enumerate(model_folders, 1):
            # Try to load model info
            info_path = os.path.join(models_dir, model_name, "model_info.json")
            if os.path.exists(info_path):
                try:
                    import json
                    with open(info_path, "r") as f:
                        info = json.load(f)
                    hyperparams = info.get('hyperparameters', {})
                    epochs = hyperparams.get('epochs', 'N/A')
                    accuracy = info.get('test_accuracy', 'N/A')
                    accuracy_str = f"{accuracy:.1%}" if accuracy != 'N/A' else 'N/A' 
                    print(f"  {i}. {model_name} (trained: {info.get('training_date', 'unknown')}) - Epochs: {epochs}, Accuracy: {accuracy_str}")
                except:
                    print(f"  {i}. {model_name}")
            else:
                print(f"  {i}. {model_name}")
    else:
        print("No trained models found.")
    
    return model_folders

def select_model_for_training():
    """Allows user to select training options"""
    print("\nTraining options:")
    print("1. Train new model")
    print("2. List existing models")
    print("3. Cancel")
    
    choice = input("\nSelect an option (1-3) [1]: ").strip() or "1"
    
    if choice == "1":
        # New model
        model_name = input("\nEnter a name for the new model (leave empty for timestamp): ").strip()
        if not model_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{timestamp}"
        return "new", model_name
    
    elif choice == "2":
        # List models
        list_available_models()
        return select_model_for_training()  # Ask again
    
    else:
        return "cancel", None

def main():
    """Main function to train the music genre classifier"""
    print("=== Music Genre Classifier Trainer (GTZAN Dataset) ===")
    
    # Model selection for training
    action, model_name = select_model_for_training()
    
    if action == "cancel":
        print("Training cancelled.")
        return
    
    # Always training a new model now
    print(f"\nTraining new model: {model_name}")
    
    # Get hyperparameters configuration
    hyperparams = get_hyperparameters()
    if hyperparams is None:
        print("Training cancelled.")
        return
    
    # Request GTZAN dataset location
    print("\nThe GTZAN dataset will be used to train the model.")
    dataset_dir = input("Path to GTZAN dataset directory (default: 'genres_original'): ").strip() or "genres_original"
    
    # Verify and use existing dataset
    if not use_existing_dataset(dataset_dir):
        print("Could not use the dataset. Check the path and directory structure.")
        return
    
    # Prepare dataset
    X, y = prepare_dataset(dataset_dir)
    
    if X is None or len(X) == 0:
        print("Could not extract features. Check the audio files.")
        return
    
    # Train model
    model, history, encoder, test_accuracy = train_model(X, y, hyperparams, model_name) 
    
    # Save model with the specified name, hyperparameters and accuracy
    saved_model_dir = save_model(model, encoder, hyperparams, model_name, test_accuracy) 
    
    print(f"\nTraining completed! Model saved as '{model_name}'")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Model directory: {saved_model_dir}")
    print("You can now use 'audio_classifier.py' to classify new audio files.")

if __name__ == "__main__":
    # Check if necessary libraries are installed
    try:
        import librosa
        import tensorflow as tf
    except ImportError:
        print("Installing necessary libraries...")
        import subprocess
        subprocess.check_call(["pip", "install", "librosa", "tensorflow", "scikit-learn", "matplotlib", "pandas", "numpy"])
        print("Libraries installed correctly.")
    
    main()
# -*- coding: utf-8 -*-
"""
Music Genre Classifier

This script classifies audio files into ten musical genres
using a model trained with the GTZAN dataset.
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time

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
    
    return model_folders

def select_model():
    """Allows user to select which model to use"""
    available_models = list_available_models()
    
    if not available_models:
        print("No trained models found. Run 'audio_classifier_trainer.py' first.")
        return None
    
    if len(available_models) == 1:
        print(f"Using model: {available_models[0]}")
        return available_models[0]
    
    print("\nAvailable models:")
    for i, model_name in enumerate(available_models, 1):
        # Try to load model info
        info_path = os.path.join("models", model_name, "model_info.json")
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
    
    while True:
        choice = input(f"\nSelect model (1-{len(available_models)}) [1]: ").strip() or "1"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                selected_model = available_models[idx]
                print(f"Selected model: {selected_model}")
                return selected_model
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")

def load_model_and_params(model_name=None):
    """
    Loads the trained model and preprocessing parameters.
    
    Args:
        model_name (str): Name of the model to load (optional)
    
    Returns:
        tuple: (model, encoder, params) - Model, encoder and parameters
    """
    # Select model if not specified
    if model_name is None:
        model_name = select_model()
        if model_name is None:
            return None, None, None
    
    model_dir = f"models/{model_name}"
    
    # Check if necessary files exist
    model_path = f"{model_dir}/music_genre_classifier.h5"
    encoder_path = f"{model_dir}/label_encoder.pkl"
    params_path = f"{model_dir}/preprocessing_params.pkl"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None, None, None
    
    try:
        # Load model
        model = load_model(model_path)
        
        # Load encoder
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        
        # Load parameters
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        
        return model, encoder, params
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

def extract_features(file_path, params):
    """
    Extracts MFCC features from a complete audio file.
    
    Args:
        file_path (str): Path to the audio file
        params (dict): Preprocessing parameters
        
    Returns:
        numpy.ndarray: Averaged MFCC features
    """
    try:
        # Load complete audio file
        signal, sr = librosa.load(file_path, sr=params["sample_rate"])
        
        # Ensure minimum duration (3 seconds)
        if len(signal) < 3 * params["sample_rate"]:
            print(f"File {file_path} is too short. At least 3 seconds needed.")
            return None
        
        # Extract MFCC from entire song
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=params["num_mfcc"],
            n_fft=params["n_fft"],
            hop_length=params["hop_length"]
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

def classify_audio(file_path, model, encoder, params):
    """
    Classifies an audio file into one of the musical genres.
    
    Args:
        file_path (str): Path to the audio file
        model: Trained model
        encoder: Label encoder
        params (dict): Preprocessing parameters
        
    Returns:
        tuple: (genre, probabilities, prediction_time) - Predicted genre, probabilities and prediction time
    """
    # Extract features
    features = extract_features(file_path, params)
    
    if features is None:
        return None, None, None
    
    # Prepare for prediction
    X = np.array([features])
    
    # Predict
    start_time = time.time()
    predictions = model.predict(X)
    prediction_time = time.time() - start_time
    
    # Get probabilities
    probabilities = predictions[0]
    
    # Get genre with highest probability
    genre_idx = np.argmax(probabilities)
    genre = params["genres"][genre_idx]
    
    return genre, probabilities, prediction_time

def visualize_prediction(probabilities, genres):
    """
    Visualizes prediction probabilities.
    
    Args:
        probabilities (numpy.ndarray): Probabilities for each genre
        genres (list): List of genres
    """
    # Sort genres by probability for better visualization
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_genres = [genres[i] for i in sorted_indices]
    sorted_probs = probabilities[sorted_indices]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_genres, sorted_probs * 100)
    
    # Color the bar of the genre with highest probability
    bars[0].set_color('green')
    
    plt.xlabel('Musical Genre')
    plt.ylabel('Probability (%)')
    plt.title('Music Genre Prediction')
    plt.xticks(rotation=45)
    
    # Add labels with percentages
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', rotation=0)
    
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

def visualize_audio(file_path):
    """
    Visualizes the waveform and spectrogram of an audio file.
    
    Args:
        file_path (str): Path to the audio file
    """
    # Load file
    y, sr = librosa.load(file_path, sr=22050)
    
    plt.figure(figsize=(12, 8))
    
    # Waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    
    # Spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to classify audio files"""
    print("=== Music Genre Classifier (GTZAN) ===")
    
    # Load model and parameters (with model selection)
    model, encoder, params = load_model_and_params()
    
    if model is None:
        return
    
    print(f"\nModel loaded. Supported genres: {', '.join(params['genres'])}")
    
    while True:
        print("\n1. Classify an audio file")
        print("2. Classify a directory of files")
        print("3. Visualize an audio file")
        print("4. Switch model")
        print("5. Exit")
        
        choice = input("\nSelect an option: ")
        
        if choice == "1":
            # Classify a single file
            file_path = input("\nPath to audio file: ")
            
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                continue
            
            if not file_path.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a', '.au')):
                print("File must be an audio file (mp3, wav, ogg, m4a, au).")
                continue
            
            print(f"\nAnalyzing {os.path.basename(file_path)}...")
            
            # Classify audio
            genre, probabilities, prediction_time = classify_audio(file_path, model, encoder, params)
            
            if genre is None:
                print("Could not classify the file.")
                continue
            
            # Show results
            print(f"\nPredicted genre: {genre.upper()}")
            print(f"Prediction time: {prediction_time:.2f} seconds")
            print("\nProbabilities:")
            
            # Show sorted probabilities
            sorted_indices = np.argsort(probabilities)[::-1]
            for i in sorted_indices:
                print(f"  {params['genres'][i]}: {probabilities[i]*100:.2f}%")
            
            # Visualize prediction
            visualize_prediction(probabilities, params["genres"])
            
        elif choice == "2":
            # Classify a directory
            dir_path = input("\nPath to directory with audio files: ")
            
            if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
                print(f"Directory {dir_path} does not exist.")
                continue
            
            # Get audio files
            audio_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a', '.au'))]
            
            if not audio_files:
                print("No audio files found in the directory.")
                continue
            
            print(f"\nFound {len(audio_files)} audio files.")
            
            # Results
            results = []
            
            for i, file_name in enumerate(audio_files):
                file_path = os.path.join(dir_path, file_name)
                print(f"\nAnalyzing {i+1}/{len(audio_files)}: {file_name}")
                
                # Classify audio
                genre, probabilities, prediction_time = classify_audio(file_path, model, encoder, params)
                
                if genre is not None:
                    print(f"  Predicted genre: {genre.upper()}")
                    results.append((file_name, genre, probabilities))
                else:
                    print("  Could not classify the file.")
            
            # Show summary
            print("\n=== Classification Summary ===")
            for file_name, genre, _ in results:
                print(f"{file_name}: {genre.upper()}")
            
            # Save results to CSV
            save_option = input("\nDo you want to save the results to a CSV file? (y/n) [y]: ").strip().lower() or "y"
            if save_option in ['y', 'yes', 's', 'si', 'sí']:
                import csv
                csv_file = os.path.join(dir_path, "genre_classification.csv")
                
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["File", "Genre"] + params["genres"])
                    
                    for file_name, genre, probs in results:
                        row = [file_name, genre] + [f"{p*100:.2f}%" for p in probs]
                        writer.writerow(row)
                
                print(f"Results saved to {csv_file}")
            
        elif choice == "3":
            # Visualize an audio file
            file_path = input("\nPath to audio file: ")
            
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                continue
            
            if not file_path.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a', '.au')):
                print("File must be an audio file (mp3, wav, ogg, m4a, au).")
                continue
            
            visualize_audio(file_path)
            
        elif choice == "4":
            # Switch model
            print("\nSwitching model...")
            new_model, new_encoder, new_params = load_model_and_params()
            if new_model is not None:
                model, encoder, params = new_model, new_encoder, new_params
                print(f"Model switched. Supported genres: {', '.join(params['genres'])}")
            else:
                print("Model not changed.")
            
        elif choice == "5":
            break
            
        else:
            print("Invalid option.")
    
    print("\nThanks for using the Music Genre Classifier!")

if __name__ == "__main__":
    # Check if necessary libraries are installed
    try:
        import librosa
        import tensorflow as tf
        import matplotlib.pyplot as plt
    except ImportError:
        print("Installing necessary libraries...")
        import subprocess
        subprocess.check_call(["pip", "install", "librosa", "tensorflow", "matplotlib", "numpy"])
        print("Libraries installed correctly.")
    
    main()
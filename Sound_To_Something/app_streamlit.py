import os

# Usar variables de entorno para configurar Streamlit
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

# Ahora importa streamlit
import streamlit as st

# Importar el resto de bibliotecas
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image
from io import BytesIO
import imageio
import tempfile
import shutil
import base64

# Import functionality from your existing modules
from audio_classifier import list_available_models, select_model, load_model_and_params, classify_audio
from text_to_image_generator import (
    setup_model, generate_image, generate_video, 
    music_genre_to_prompt_static, song_to_prompt,
    extract_song_features_enhanced
)

# Create necessary directories
os.makedirs("temp", exist_ok=True)

# --------- FUNCIONES AUXILIARES ---------

# Function to save uploaded file temporarily
def save_uploaded_file(uploaded_file):
    try:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Para Streamlit, uploaded_file es un objeto UploadedFile
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Guardar el archivo
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Function to visualize audio waveform and spectrogram
def visualize_audio(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=22050)
    
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    
    # Waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Forma de onda')
    
    # Spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma')
    
    plt.tight_layout()
    
    # Save figure to temporary file
    temp_file = os.path.join("temp", f"spectrogram_{int(time.time())}.png")
    fig.savefig(temp_file)
    plt.close(fig)
    
    return temp_file

# Load all models
@st.cache_resource
def load_models(selected_model_name=None):
    try:
        # Show loading message
        with st.spinner("Loading AI models..."):
            try:
                # Load classification model with user selection
                available_models = list_available_models()
                if not available_models:
                    st.warning("No trained audio classification models found.")
                    classifier_model, encoder, params = None, None, None
                else:
                    # Use selected model or default to first
                    if selected_model_name and selected_model_name in available_models:
                        model_name = selected_model_name
                    else:
                        model_name = available_models[0]
                    
                    st.info(f"Using audio classification model: **{model_name}**")
                    classifier_model, encoder, params = load_model_and_params(model_name)
                
                # Check GPU availability
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                image_pipe, is_cpu = setup_model("image")
                
                # Only load video model if GPU is available
                video_pipe = None
                if device == "cuda":
                    video_pipe, _ = setup_model("video")
                
                models = {
                    "classifier": (classifier_model, encoder, params),
                    "image_generator": image_pipe,
                    "video_generator": video_pipe,
                    "device": device,
                    "is_cpu": is_cpu,
                    "current_model": model_name if available_models else None
                }
                
                return models
            except Exception as e:
                st.error(f"Error during model loading: {e}")
                import traceback
                st.error(traceback.format_exc())
                return None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Create a GPU info message
def get_gpu_info():
    if torch.cuda.is_available():
        device = "CUDA (GPU)"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"🖥️ Using {device}: {gpu_name} ({gpu_mem_total:.1f} GB)"
    else:
        return "⚠️ Using CPU - Generation will be very slow. GPU is recommended."

# Audio analysis function
def analyze_audio(audio_file):
    if audio_file is None:
        return None, None, None, None, "Please upload an audio file"
    
    st.info("Starting audio analysis...")
    
    # Get selected model from sidebar
    selected_model = st.session_state.get('selected_model', None)
    
    # Call load_models with selected model
    models_data = load_models(selected_model)
    if models_data is None:
        return None, None, None, None, "Failed to load models"
    
    # Get classifier model
    classifier_model, encoder, params = models_data["classifier"]
    
    # Check if classifier is available
    if classifier_model is None:
        return None, None, None, None, "Audio classifier not available. Please train a model first."
    
    # Show which model is being used
    current_model = models_data.get("current_model", "Unknown")
    st.info(f"Analyzing with model: **{current_model}**")
    
    try:
        # Save uploaded audio to temporary file
        st.info("Saving audio file...")
        file_path = save_uploaded_file(audio_file)
        
        if not file_path:
            return None, None, None, None, "Error saving uploaded file"
        
        # Generate spectrogram and save its path
        st.info("Generating spectrogram...")
        spectrogram_path = visualize_audio(file_path)
        
        # Classify genre
        st.info("Classifying musical genre...")
        genre, probabilities, prediction_time = classify_audio(file_path, classifier_model, encoder, params)
        
        if genre is None:
            return None, None, None, None, "Could not classify the audio. Check the file format and try again."
        
        # Show prediction confidence
        max_prob = np.max(probabilities)
        st.success(f"Detected genre: **{genre.upper()}** (confidence: {max_prob:.1%})")
        
        # Extract features
        st.info("Extracting audio features...")
        features = extract_song_features_enhanced(file_path)
        
        # Generate prompt for image creation
        st.info("Generating prompt for image creation...")
        prompt = song_to_prompt(file_path, genre)
        
        # Create genre probability chart
        genres = params["genres"]
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_genres = [genres[i] for i in sorted_indices]
        sorted_probs = probabilities[sorted_indices]
        
        # Create a NEW figure for the genre chart
        fig_genres, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(sorted_genres[:5], sorted_probs[:5] * 100)
        bars[0].set_color('#1E88E5')
        ax.set_ylabel('Probability (%)')
        ax.set_title('Top 5 Most Likely Genres')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', rotation=0)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save genre chart with a different name
        timestamp = int(time.time())
        genre_probs_filename = f"genre_probs_{timestamp}.png"
        probs_chart_path = os.path.join("temp", genre_probs_filename)
        fig_genres.savefig(probs_chart_path)
        plt.close(fig_genres)
        
        # Verify that files were actually saved
        if not os.path.exists(spectrogram_path):
            return None, None, None, None, "Error saving spectrogram image"
        
        # Only verify that the probability file exists
        if not os.path.exists(probs_chart_path):
            return None, None, None, None, "Error saving genre probability chart"
        
        # Format features for display
        features_html = f"""<h3>Detected Genre: {genre.upper()}</h3>
        <p><strong>Model used:</strong> {current_model}</p>
        <p><strong>Prediction confidence:</strong> {max_prob:.1%}</p>
        <h4>Basic Features</h4>
        <ul>
            <li><b>Tempo:</b> {features['tempo']} BPM</li>
            <li><b>Energy:</b> {features['energy']}</li>
            <li><b>Key:</b> {features['key']}</li>
        </ul>"""

        if 'instruments' in features and features['instruments']:
            features_html += f"""<h4>Detected Instruments</h4>
        <p>{", ".join(features['instruments'])}</p>"""

        if 'emotions' in features and features['emotions']:
            features_html += f"""<h4>Emotional Qualities</h4>
        <p>{", ".join(features['emotions'])}</p>"""

        features_html += f"""<h4>Sound Characteristics</h4>
        <ul>
            <li><b>Vocals:</b> {"Present" if features.get('has_vocals', False) else "Not detected"}</li>
            <li><b>Brightness:</b> {features['brightness']}</li>
            <li><b>Bass presence:</b> {features['bass_presence']}</li>
            <li><b>Dynamism:</b> {features['dynamism']}</li>
        </ul>"""

        if 'structure' in features:
            features_html += f"<p><b>Structure:</b> {features['structure']}</p>"
        
        return file_path, spectrogram_path, probs_chart_path, prompt, features_html
    
    except Exception as e:
        st.error(f"Error analyzing audio: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, f"Error analyzing audio: {str(e)}"

# Image generation from prompt function
def generate_image_from_prompt(prompt, steps, width, height, num_images, negative_prompt="", guidance_scale=7.5, seed=-1):
    if not prompt:
        return None, "Please enter a prompt"
    
    # Load models
    models_data = load_models()
    if models_data is None:
        return None, "Failed to load models"
    
    # Get image generator
    image_pipe = models_data["image_generator"]
    is_cpu = models_data["is_cpu"]
    
    try:
        # Configure advanced parameters
        advanced_params = {"guidance_scale": guidance_scale}
        if seed != -1:
            advanced_params["generator"] = torch.Generator().manual_seed(seed)
        
        # Prepare parameters
        image_params = {
            'num_inference_steps': steps,
            'width': width,
            'height': height,
            'num_images_per_prompt': num_images,
            **advanced_params
        }
        
        if negative_prompt:
            image_params['negative_prompt'] = negative_prompt
        
        # Generate images
        with st.spinner("Generating images..."):
            images = generate_image(image_pipe, prompt, image_params, is_cpu)
        
        if not images:
            return None, "Failed to generate images"
        
        # Save images both as files and as base64 data
        image_data = []
        for i, img in enumerate(images):
            # Save to file
            timestamp = int(time.time())
            img_path = os.path.join("temp", f"generated_image_{timestamp}_{i}.png")
            img.save(img_path)
            
            # Also convert to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            image_data.append({
                "path": img_path,
                "base64": f"data:image/png;base64,{img_str}"
            })
        
        return image_data, None
    
    except Exception as e:
        return None, f"Error generating images: {str(e)}"

# Function to generate video from image
def generate_video_from_image(image_path, steps, motion_bucket, fps, duration):
    if image_path is None:
        return None, None, "Please upload or select an image"
    
    # Load models
    models_data = load_models()
    if models_data is None:
        return None, None, "Failed to load models"
    
    # Check if we have GPU
    if models_data["device"] != "cuda":
        return None, None, "Video generation requires a CUDA-compatible GPU"
    
    # Get video generator
    video_pipe = models_data["video_generator"]
    if video_pipe is None:
        return None, None, "Video generation model not loaded"
    
    try:
        # Load image
        image = Image.open(image_path)
        
        # Check image resolution
        img_width, img_height = image.size
        if img_width > 512 or img_height > 512:
            # Reduce resolution maintaining aspect ratio
            if img_width > img_height:
                new_width = 512
                new_height = int(img_height * (512 / img_width))
            else:
                new_height = 512
                new_width = int(img_width * (512 / img_height))
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Preparar parámetros
        video_params = {
            'num_inference_steps': steps,
            'motion_bucket_id': motion_bucket,
            'decode_chunk_size': 1
        }
        
        # Generar video
        with st.spinner("Generating video animation..."):
            st.info("It can take a while depending on your GPU...")
            frames = generate_video(video_pipe, image, video_params, False)
        
        if not frames:
            return None, None, "Failed to generate video frames"
        
        # Calculate fps to achieve desired duration
        adjusted_fps = len(frames) / duration
        
        # Save as GIF
        gif_path = os.path.join("temp", f"video_{int(time.time())}.gif")
        frames_pil = [Image.fromarray(np.uint8(frame)) for frame in frames]
        frames_pil[0].save(
            gif_path,
            save_all=True,
            append_images=frames_pil[1:],
            duration=1000//adjusted_fps,
            loop=0
        )
        
        # Create MP4
        mp4_path = None
        try:
            import moviepy.editor as mpy
            
            mp4_path = os.path.join("temp", f"video_{int(time.time())}.mp4")
            clip = mpy.ImageSequenceClip([np.array(frame) for frame in frames], fps=adjusted_fps)
            clip.write_videofile(mp4_path, codec='libx264', fps=adjusted_fps, verbose=False, logger=None)
        except Exception as e:
            st.warning(f"Could not create MP4: {e}")
        
        return gif_path, mp4_path, None
    
    except Exception as e:
        return None, None, f"Error generating video: {str(e)}"

def setup_sidebar():
    """
    Configura la barra lateral con opciones de selección de modelo.
    
    Returns:
        str: Nombre del modelo seleccionado
    """
    st.sidebar.markdown("## Model Configuration")
    
    # Obtener modelos disponibles
    try:
        available_models = list_available_models()
        
        if not available_models:
            st.sidebar.warning("No trained models found")
            st.sidebar.markdown("""
            **No audio classification models available.**
            
            To use audio classification:
            1. Train a model using `audio_classifier_trainer.py`
            2. Restart this application
            """)
            return None
        
        st.sidebar.success(f"Found {len(available_models)} model(s)")
        
        # Mostrar información de modelos disponibles
        st.sidebar.markdown("### Available Models:")
        
        model_info = {}
        for model in available_models:
            try:
                import json
                info_path = f"models/{model}/model_info.json"
                if os.path.exists(info_path):
                    with open(info_path, "r") as f:
                        info = json.load(f)
                    training_date = info.get('training_date', 'Unknown')
                    hyperparams = info.get('hyperparameters', {})
                    epochs = hyperparams.get('epochs', 'N/A')
                    accuracy = info.get('test_accuracy', 'N/A')
                    
                    model_info[model] = {
                        'date': training_date,
                        'epochs': epochs,
                        'accuracy': accuracy
                    }
                    
                    # Formatear fecha si es posible
                    try:
                        from datetime import datetime
                        if training_date != 'Unknown':
                            dt = datetime.strptime(training_date, '%Y-%m-%d %H:%M:%S')
                            formatted_date = dt.strftime('%Y-%m-%d %H:%M')
                        else:
                            formatted_date = 'Unknown'
                    except:
                        formatted_date = training_date
                    
                    st.sidebar.markdown(f"""
                    **{model}**
                    - Trained: {formatted_date}
                    - Epochs: {epochs}
                    - Accuracy: {accuracy if accuracy != 'N/A' else 'N/A'}
                    """)
                else:
                    model_info[model] = {
                        'date': 'Unknown',
                        'epochs': 'N/A',
                        'accuracy': 'N/A'
                    }
                    st.sidebar.markdown(f"""
                    **{model}**
                    - No detailed info available
                    """)
            except Exception as e:
                model_info[model] = {
                    'date': 'Error',
                    'epochs': 'Error',
                    'accuracy': 'Error'
                }
                st.sidebar.markdown(f"""
                **{model}**
                - Error loading info
                """)
        
        # Selector de modelo
        st.sidebar.markdown("### Select Model:")
        
        # Crear opciones con información
        model_options = []
        for model in available_models:
            info = model_info.get(model, {})
            if info.get('accuracy') != 'N/A' and info.get('accuracy') != 'Error':
                try:
                    acc_value = float(info['accuracy'])
                    model_display = f"{model} (Acc: {acc_value:.1%})"
                except:
                    model_display = f"{model}"
            else:
                model_display = f"{model}"
            model_options.append(model_display)
        
        # Obtener modelo seleccionado previo del estado de sesión
        current_selection = st.session_state.get('selected_model', available_models[0])
        
        # Encontrar el índice del modelo actualmente seleccionado
        try:
            current_index = available_models.index(current_selection)
        except ValueError:
            current_index = 0
        
        # Selector de modelo
        selected_index = st.sidebar.selectbox(
            "Choose classification model:",
            range(len(model_options)),
            index=current_index,
            format_func=lambda x: model_options[x],
            help="Select which trained model to use for audio classification",
            key="model_selector"
        )
        
        selected_model = available_models[selected_index]
        
        # Guardar en estado de sesión
        st.session_state.selected_model = selected_model
        
        # Mostrar modelo actualmente en uso
        st.sidebar.markdown(f"**Active Model:** `{selected_model}`")
        
        # Botón para refrescar modelos
        if st.sidebar.button("Refresh Models", help="Scan for new trained models"):
            st.cache_resource.clear()
            st.rerun()
        
        return selected_model
        
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
        return None


# --------- APLICACIÓN PRINCIPAL ---------

# Configuración de la página
st.set_page_config(
    page_title="Sound to Something",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Aplicar estilos personalizados
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        color: #424242;
        text-align: center;
        margin-bottom: 30px;
    }
    .warning-box {
        padding: 20px;
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        margin-bottom: 20px;
        color: #000000; /* Esta línea fuerza el texto negro */
    }
    .info-box {
        padding: 20px;
        background-color: #E3F2FD; /* Fondo azul claro */
        color: #000000 !important; /* Texto negro */
        border-left: 5px solid #2196F3; /* Borde azul */
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 28px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 15px;
        color: #1E88E5;
    }
    .subsection-title {
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #424242;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #e0e0e0;
        color: #9e9e9e;
    }
    .stButton button {
        width: 100%;
    }
    
    /* NUEVOS ESTILOS PARA MEJORAR LA LEGIBILIDAD DE LOS AVISOS */
    /* Estilo para mensajes de información */
    .stAlert.info {
        background-color: #E3F2FD;
        color: #000000  !important;
        border-color: #2196F3;
    }
    .stAlert.info p {
        color: #000000  !important;
        font-weight: 500;
    }
    
    /* Estilo para mensajes de advertencia */
    .stAlert.warning {
        background-color: #FFF8E1; /* Fondo claro */
        color: #000000 !important; /* Texto negro */
        border-color: #FFC107; /* Borde amarillo */
    }
    .stAlert.warning p {
        color: #000000 !important; /* Texto negro */
        font-weight: 500; /* Negrita para mejor legibilidad */
    }
    
    /* Estilo para mensajes de error */
    .stAlert.error {
        background-color: #FFEBEE;
        color: #000000 !important;
        border-color: #F44336;
    }
    .stAlert.error p {
        color: #000000  !important;
        font-weight: 500;
    }
    
    /* Estilo para mensajes de éxito */
    .stAlert.success {
        background-color: #E8F5E9;
        color: #000000  !important;
        border-color: #4CAF50;
    }
    .stAlert.success p {
        color: #000000  !important;
        font-weight: 500;
    }
    
    /* Aumentar el tamaño del texto en todos los mensajes */
    .stAlert p {
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar el estado de la sesión si no existe
if 'step' not in st.session_state:
    st.session_state.step = 'welcome'
    
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
    
if 'spectrogram' not in st.session_state:
    st.session_state.spectrogram = None
    
if 'genre_probs' not in st.session_state:
    st.session_state.genre_probs = None
    
if 'prompt' not in st.session_state:
    st.session_state.prompt = ""
    
if 'features_html' not in st.session_state:
    st.session_state.features_html = ""
    
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
    
if 'source_image' not in st.session_state:
    st.session_state.source_image = None
    
if 'gif_output' not in st.session_state:
    st.session_state.gif_output = None
    
if 'video_output' not in st.session_state:
    st.session_state.video_output = None
    
if 'error_msg' not in st.session_state:
    st.session_state.error_msg = ""

# Funciones de navegación
def go_to_step(step):
    st.session_state.step = step
    
def start_over():
    st.session_state.step = 'upload'
    st.session_state.audio_file = None
    st.session_state.spectrogram = None
    st.session_state.genre_probs = None
    st.session_state.prompt = ""
    st.session_state.features_html = ""
    st.session_state.error_msg = ""

def main_app():
    # Encabezado principal de la aplicación
    st.markdown('<h1 class="title">Sound to Something</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Convert music into images and videos with AI</p>', unsafe_allow_html=True)

    # Información del sistema GPU/CPU
    gpu_info = get_gpu_info()
    st.markdown(f'<div class="info-box">{gpu_info}</div>', unsafe_allow_html=True)

    # Setup sidebar
    selected_model = setup_sidebar()
    
    # Store selected model in session state
    if selected_model:
        st.session_state.selected_model = selected_model
    
    # PASO 1: Bienvenida
    if st.session_state.step == 'welcome':
        st.markdown('<h2 class="section-title">Transform Your Music into Visual Art</h2>', unsafe_allow_html=True)
        st.write("Upload an audio file and let AI transform it into stunning images or videos.")
        
        if st.button("Get Started", key="start_btn"):
            go_to_step('upload')

    # PASO 2: Subir audio
    elif st.session_state.step == 'upload':
        st.markdown('<h2 class="section-title">Step 1: Upload Your Audio</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav', 'ogg', 'flac'])
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("Analyze Audio", key="analyze_btn"):
                with st.spinner("Analyzing audio..."):
                    audio_file, spectrogram, genre_probs, prompt, features_html = analyze_audio(uploaded_file)
                    
                    if audio_file and not (isinstance(features_html, str) and "Error" in features_html):
                        st.success("Analysis completed successfully!")
                        
                        # Update state
                        st.session_state.audio_file = audio_file
                        st.session_state.spectrogram = spectrogram
                        st.session_state.genre_probs = genre_probs
                        st.session_state.prompt = prompt
                        st.session_state.features_html = features_html
                        
                        # Change step
                        st.session_state.step = 'analysis'
                        st.rerun()
                    else:
                        st.error(f"Analysis error: {features_html}")

    # PASO 3: Resultados del análisis
    elif st.session_state.step == 'analysis':
        st.markdown('<h2 class="section-title">Step 2: Audio Analysis Results</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="subsection-title">Waveform and Spectrogram</h3>', unsafe_allow_html=True)
            if st.session_state.spectrogram and os.path.exists(st.session_state.spectrogram):
                st.image(st.session_state.spectrogram)
            else:
                st.warning("Spectrogram image not available. It may have been deleted or moved.")
            
            st.markdown('<h3 class="subsection-title">Genre Classification</h3>', unsafe_allow_html=True)
            if st.session_state.genre_probs and os.path.exists(st.session_state.genre_probs):
                st.image(st.session_state.genre_probs)
            else:
                st.warning("Genre probability chart not available. It may have been deleted or moved.")
        
        with col2:
            st.markdown('<h3 class="subsection-title">Detailed Audio Features</h3>', unsafe_allow_html=True)
            if st.session_state.features_html:
                st.markdown(st.session_state.features_html, unsafe_allow_html=True)
            
            st.markdown('<h3 class="subsection-title">Generated Prompt for Creation</h3>', unsafe_allow_html=True)
            prompt = st.text_area("You can edit this prompt if you want:", value=st.session_state.prompt, height=100)
            st.session_state.prompt = prompt
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Upload", key="back_to_upload_btn"):
                go_to_step('upload')
        
        with col2:
            if st.button("Continue to Generation →", key="continue_btn", type="primary"):
                st.session_state.step = 'choose_generation'

    # PASO 4: Elegir tipo de generación
    elif st.session_state.step == 'choose_generation':
        st.markdown('<h2 class="section-title">Step 3: Choose What to Generate</h2>', unsafe_allow_html=True)
        
        generation_type = st.radio(
            "What would you like to create?",
            ["Generate Image", "Generate Video"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Analysis", key="back_to_analysis_btn"):
                go_to_step('analysis')
        
        with col2:
            if st.button("Proceed →", key="proceed_btn", type="primary"):
                if generation_type == "Generate Image":
                    go_to_step('image_params')
                else:
                    go_to_step('video_params')

    # PASO 5a: Parámetros de generación de imagen
    elif st.session_state.step == 'image_params':
        st.markdown('<h2 class="section-title">Step 4: Image Generation Parameters</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="subsection-title">Basic Parameters</h3>', unsafe_allow_html=True)
            
            steps = st.slider("Inference steps", min_value=1, max_value=150, value=50, step=1, 
                             help="More steps = better quality but slower generation")
            
            width = st.slider("Width", min_value=256, max_value=1024, value=512, step=64)
            height = st.slider("Height", min_value=256, max_value=1024, value=512, step=64)
            
            num_images = st.slider("Number of images", min_value=1, max_value=4, value=1, step=1)
        
        with col2:
            st.markdown('<h3 class="subsection-title">Advanced Parameters</h3>', unsafe_allow_html=True)
            
            negative_prompt = st.text_area("Negative prompt", height=80, 
                                          help="Things you don't want in the image")
            
            guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.1,
                                      help="How closely to follow the prompt (higher = more literal)")
            
            seed = st.number_input("Seed (-1 for random)", value=-1, step=1,
                                  help="Use the same seed to get reproducible results")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", key="back_to_choice_btn_img"):
                go_to_step('choose_generation')
        
        with col2:
            if st.button("Generate Image", key="generate_img_btn", type="primary"):
                with st.spinner("Generating images..."):
                    images, error = generate_image_from_prompt(
                        st.session_state.prompt, steps, width, height, 
                        num_images, negative_prompt, guidance_scale, seed
                    )
                    
                    if images and not error:
                        st.session_state.generated_images = images
                        go_to_step('image_results')
                    else:
                        st.error(error)

    # PASO 5b: Parámetros de generación de video
    elif st.session_state.step == 'video_params':
        st.markdown('<h2 class="section-title">Step 4: Video Generation Parameters</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Memory Warning</h4>
            <p>Video generation requires at least 8GB of VRAM to work properly.</p>
            <p>Use low resolutions and few inference steps to avoid memory errors.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # First we need to generate the source image
        st.markdown('<h3 class="subsection-title">First, let\'s generate an image to animate</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4>Image Parameters</h4>', unsafe_allow_html=True)
            
            img_steps = st.slider("Image inference steps", min_value=1, max_value=150, value=30, step=1)
            
            img_width = st.slider("Width", min_value=256, max_value=512, value=384, step=64,
                                 help="For video generation, keep resolution at 384x384 for best results")
            img_height = st.slider("Height", min_value=256, max_value=512, value=384, step=64)
        
        with col2:
            st.markdown('<h4>Advanced Image Parameters</h4>', unsafe_allow_html=True)
            
            img_negative_prompt = st.text_area("Negative prompt", height=80)
            img_guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.1)
            img_seed = st.number_input("Seed (-1 for random)", value=-1, step=1)
        
        # Video parameters
        st.markdown('<h3 class="subsection-title">Video Animation Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            video_steps = st.slider("Video inference steps", min_value=10, max_value=50, value=25, step=1)
            motion_bucket = st.slider("Motion intensity", min_value=1, max_value=255, value=127, step=1)
        
        with col2:
            fps = st.slider("Frames per second", min_value=1, max_value=30, value=8, step=1)
            duration = st.slider("Duration (seconds)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", key="back_to_choice_btn_vid"):
                go_to_step('choose_generation')
        
        with col2:
            if st.button("Generate Video", key="generate_vid_btn", type="primary"):
                # First generate the image
                with st.spinner("Generating source image..."):
                    images, error = generate_image_from_prompt(
                        st.session_state.prompt, img_steps, img_width, img_height, 
                        1, img_negative_prompt, img_guidance_scale, img_seed
                    )
                    
                    if images and not error:
                        # Store only the path from the dictionary
                        st.session_state.source_image = images[0]["path"]
                        
                        # Then generate the video
                        with st.spinner("Generating video animation..."):
                            gif_path, mp4_path, video_error = generate_video_from_image(
                                st.session_state.source_image,
                                video_steps, motion_bucket, fps, duration
                            )
                            
                            if gif_path and mp4_path and not video_error:
                                # Guardar en el estado de sesión y redirigir inmediatamente
                                st.session_state.gif_output = gif_path
                                st.session_state.video_output = mp4_path
                                go_to_step('video_results')
                                st.rerun()
                            else:
                                st.error(video_error)
                    else:
                        st.error(error)

    # PASO 6a: Resultados de la generación de imagen
    elif st.session_state.step == 'image_results':
        st.markdown('<h2 class="section-title">Results: Generated Images</h2>', unsafe_allow_html=True)
        
        # Contenedor principal para los resultados
        result_container = st.container()
        
        with result_container:
            # Mostrar galería de imágenes
            if st.session_state.generated_images:
                st.markdown('<h3 class="subsection-title">Generated Images</h3>', unsafe_allow_html=True)
                
                # Determinar número de columnas según cantidad de imágenes
                num_images = len(st.session_state.generated_images)
                cols = min(num_images, 2)  # Max 2 columns
                
                # Crear la cuadrícula
                image_cols = st.columns(cols)
                
                for i, img_data in enumerate(st.session_state.generated_images):
                    with image_cols[i % cols]:
                        st.markdown(f"#### Image {i+1}")
                        
                        try:
                            # Intentar primero con el archivo
                            if "path" in img_data and os.path.exists(img_data["path"]):
                                st.image(img_data["path"], use_container_width=True)
                            # Intentar con base64 si el archivo no existe
                            elif "base64" in img_data:
                                st.image(img_data["base64"], use_container_width=True)
                            else:
                                st.warning("Image not available")
                                continue
                                
                            # Botón de descarga
                            if "base64" in img_data:
                                b64data = img_data["base64"].split(",")[1]
                                st.download_button(
                                    f"Download Image",
                                    base64.b64decode(b64data),
                                    f"generated_image_{i+1}.png",
                                    mime="image/png"
                                )
                            
                            # Botón para animar esta imagen específica
                            if st.button(f"Animate This Image", key=f"animate_btn_{i}", type="secondary"):
                                st.session_state.source_image = img_data["path"] if "path" in img_data else None
                                go_to_step('video_params')
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
            else:
                st.warning("No images were generated or they were deleted")
        
        # Navegación en sección separada
        st.markdown("<hr>", unsafe_allow_html=True)
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        
        with nav_col1:
            if st.button("← Back to Parameters", key="back_to_img_params"):
                go_to_step('image_params')
        
        with nav_col2:
            if st.button("Generate More Images", key="more_images_btn"):
                go_to_step('image_params')
        
        with nav_col3:
            if st.button("Start New Audio", key="new_audio_btn_img", type="primary"):
                start_over()

    # PASO 6b: Resultados de la generación de video
    elif st.session_state.step == 'video_results':
        st.markdown('<h2 class="section-title">Results: Generated Video</h2>', unsafe_allow_html=True)
        
        # Contenedor para organizar visualmente los resultados
        result_container = st.container()
        
        with result_container:
            # Sección de imagen fuente
            st.markdown('<h3 class="subsection-title">Source Image</h3>', unsafe_allow_html=True)
            source_col1, source_col2 = st.columns([2, 1])
            
            with source_col1:
                if st.session_state.source_image and os.path.exists(st.session_state.source_image):
                    st.image(st.session_state.source_image, use_container_width=True)
            
            with source_col2:
                st.markdown("#### Details")
                st.write("This is the source image that was animated.")
                if st.session_state.source_image:
                    st.download_button(
                        "Download Source Image",
                        open(st.session_state.source_image, "rb").read(),
                        "source_image.png",
                        mime="image/png"
                    )
            
            # Línea divisoria
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Sección de resultados de animación
            st.markdown('<h3 class="subsection-title">Animation Results</h3>', unsafe_allow_html=True)
            
            anim_col1, anim_col2 = st.columns(2)
            
            with anim_col1:
                st.markdown("#### GIF Animation")
                if st.session_state.gif_output and os.path.exists(st.session_state.gif_output):
                    st.image(st.session_state.gif_output, use_container_width=True)
                    st.download_button(
                        "Download GIF",
                        open(st.session_state.gif_output, "rb").read(),
                        "animation.gif",
                        mime="image/gif"
                    )
                else:
                    st.warning("GIF file not available or was deleted")
            
            with anim_col2:
                st.markdown("#### MP4 Video")
                if st.session_state.video_output and os.path.exists(st.session_state.video_output):
                    st.video(st.session_state.video_output)
                    st.download_button(
                        "Download MP4",
                        open(st.session_state.video_output, "rb").read(),
                        "animation.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.warning("MP4 file not available or was deleted")
        
        # Botones de navegación en una sección separada
        st.markdown("<hr>", unsafe_allow_html=True)
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        
        with nav_col1:
            if st.button("← Back to Parameters", key="back_to_params_btn"):
                go_to_step('video_params')
        
        with nav_col2:
            if st.button("Edit Prompt", key="edit_prompt_btn"):
                go_to_step('analysis')
        
        with nav_col3:
            if st.button("Start New Audio", key="new_audio_btn_vid", type="primary"):
                start_over()

        st.markdown('<div class="footer">Sound to Something © 2025</div>', unsafe_allow_html=True)


# Limpiar archivos temporales antiguos al iniciar
if __name__ == "__main__":
    # Run main app
    main_app()
    
    # Clean temp files
    try:
        temp_files = os.listdir("temp")
        current_time = time.time()
        for file in temp_files:
            file_path = os.path.join("temp", file)
            # Delete files older than 1 hour
            if os.path.isfile(file_path) and current_time - os.path.getmtime(file_path) > 3600:
                os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning temp files: {e}")
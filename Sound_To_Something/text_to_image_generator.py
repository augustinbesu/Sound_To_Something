# -*- coding: utf-8 -*-
"""
Text-to-Image and Text-to-Video Generation using Diffusion Models
"""

import matplotlib.pyplot as plt
import torch
import os
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from PIL import Image
import numpy as np

# Librosa patch (integrated directly)
try:
    import librosa
    
    # Monkey patch to fix peak_pick compatibility issue
    original_peak_pick = librosa.util.peak_pick

    def fixed_peak_pick(x, *args, **kwargs):
        if len(args) > 0:
            # Compatible mode with old versions
            pre_max, post_max, pre_avg, post_avg, delta, wait = args
            return original_peak_pick(
                x, pre_max=pre_max, post_max=post_max, 
                pre_avg=pre_avg, post_avg=post_avg, 
                delta=delta, wait=wait
            )
        else:
            # Compatible mode with new versions
            return original_peak_pick(x, **kwargs)

    librosa.util.peak_pick = fixed_peak_pick
    print("Librosa patch applied successfully")
except ImportError:
    print("Librosa not available - some features may not work")

def setup_model(model_type="image"):
    """
    Sets up and loads the Stable Diffusion model.
    
    Args:
        model_type (str): Type of model ("image" or "video")
        
    Returns:
        Pipeline: Configured pipeline
    """
    # Fixed model for images
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    
    print(f"Loading {model_type} model: {model_id}")
    
    # Configure PyTorch for better memory management
    torch.cuda.empty_cache()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        print("Using GPU (CUDA)")
        
        # Show GPU memory information
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"Total GPU memory: {gpu_mem_total:.2f} GB")
        print(f"Reserved GPU memory: {gpu_mem_reserved:.2f} GB")
        print(f"Allocated GPU memory: {gpu_mem_allocated:.2f} GB")
        
        is_cpu = False
    else:
        device = "cpu"
        print("\nWARNING! CUDA not available, using CPU.")
        print("CPU generation can take A LOT of time (10-30 minutes per image/video).")
        print("A GPU with CUDA is recommended for this program.")
        
        print("Using CPU optimized configuration...")
        is_cpu = True
    
    # Load the appropriate model based on type
    if model_type == "image":
        if is_cpu:
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                use_safetensors=True
            )
        pipe = pipe.to(device)
    elif model_type == "video":
        # Configuration to optimize memory
        if is_cpu:
            print("WARNING! Video generation on CPU is extremely slow.")
            print("It may take hours to complete.")
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt"
            )
        else:
            # Load with memory optimizations
            print("Loading video model with memory optimizations...")
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", 
                torch_dtype=torch.float16,
                variant="fp16"
            )
            
            # Enable offloading to CPU for unused components
            pipe.enable_model_cpu_offload()
            
            # Enable sequential attention to save memory
            if hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload()
            
            # Enable attention slicing to reduce memory usage
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing(1)
            
            # Use vae in evaluation mode to save memory
            if hasattr(pipe, "vae"):
                pipe.vae.eval()
                
            print("Memory optimizations applied")
        
    return pipe, is_cpu

def generate_image(pipe, prompt, params=None, is_cpu=False):
    """
    Generates images from a prompt using the Stable Diffusion pipeline.
    
    Args:
        pipe: Stable Diffusion pipeline
        prompt (str): Text description of the image to generate
        params (dict): Additional parameters for generation
        is_cpu (bool): Whether running on CPU
        
    Returns:
        list: List of generated images
    """
    if params is None:
        params = {}
    
    print(f"[PROMPT]: {prompt}")
    print(f"Parameters: {params}")
    
    if is_cpu:
        print("\nGenerating image on CPU. This may take a very long time...")
        print("Please wait patiently. Do not close the program.")
        
        # Show time counter
        import time
        start_time = time.time()
        
        def callback_fn(step, timestep, latents):
            elapsed = time.time() - start_time
            print(f"Step {step}/{params.get('num_inference_steps', 50)} - Elapsed time: {elapsed:.1f} seconds", end="\r")
            return None
        
        # Add callback to show progress
        params['callback'] = callback_fn
    
    # Generate images
    images = pipe(prompt, **params).images
    
    if is_cpu:
        total_time = time.time() - start_time
        print(f"\nGeneration completed in {total_time:.1f} seconds")
    
    # Display images
    num_images = len(images)
    if num_images > 1:
        fig, ax = plt.subplots(nrows=1, ncols=num_images, figsize=(5*num_images, 5))
        for i in range(num_images):
            ax[i].imshow(images[i])
            ax[i].axis('off')
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(images[0])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return images

def generate_video(pipe, image, params=None, is_cpu=False):
    """
    Generates a video from an image using the Stable Video Diffusion pipeline.
    
    Args:
        pipe: Stable Video Diffusion pipeline
        image: Initial image for video generation
        params (dict): Additional parameters for generation
        is_cpu (bool): Whether running on CPU
        
    Returns:
        list: List of video frames
    """
    if params is None:
        params = {}
    
    print(f"Generating video from image...")
    print(f"Parameters: {params}")
    
    # Free memory before generating
    torch.cuda.empty_cache()
    
    if is_cpu:
        print("\nGenerating video on CPU. This may take A VERY long time (possibly hours)...")
    else:
        # Show GPU memory information
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"GPU memory before generation - Total: {gpu_mem_total:.2f} GB, Reserved: {gpu_mem_reserved:.2f} GB, Allocated: {gpu_mem_allocated:.2f} GB")
    
    print("Please wait patiently. Do not close the program.")
    
    # Show time counter
    import time
    start_time = time.time()
    
    try:
        # Reduce image resolution if necessary to save memory
        max_size = 512  # Maximum size for input image
        orig_width, orig_height = image.size
        
        if orig_width > max_size or orig_height > max_size:
            print(f"Resizing image from {orig_width}x{orig_height} to maximum {max_size}px to save memory")
            if orig_width > orig_height:
                new_width = max_size
                new_height = int(orig_height * (max_size / orig_width))
            else:
                new_height = max_size
                new_width = int(orig_width * (max_size / orig_height))
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            print(f"New resolution: {new_width}x{new_height}")
        
        # Adjust parameters to reduce memory usage
        if not is_cpu:
            # Reduce inference steps if not specified
            if 'num_inference_steps' not in params or params['num_inference_steps'] > 30:
                params['num_inference_steps'] = 30
                print(f"Adjusting inference steps to {params['num_inference_steps']} to save memory")
            
            # Use decode_chunk_size to process in chunks
            params['decode_chunk_size'] = 1
        
        # Generate video with memory error handling
        frames = pipe(image, **params).frames[0]
        
    except torch.cuda.OutOfMemoryError as e:
        print("\nGPU memory error! Trying with lower memory configuration...")
        
        # Free all possible memory
        torch.cuda.empty_cache()
        
        # Reduce resolution even more
        max_size = 384
        image = image.resize((max_size, max_size), Image.LANCZOS)
        
        # Reduce parameters to minimize memory usage
        params['num_inference_steps'] = 20
        params['decode_chunk_size'] = 1
        
        try:
            # Second attempt with reduced parameters
            frames = pipe(image, **params).frames[0]
        except Exception as e2:
            print(f"\nError generating video even with reduced configuration: {str(e2)}")
            print("Recommendations:")
            print("1. Close other applications using GPU")
            print("2. Restart the program")
            print("3. Use a lower resolution for the initial image")
            print("4. Reduce the number of inference steps")
            return None
    
    total_time = time.time() - start_time
    print(f"\nVideo generation completed in {total_time:.1f} seconds")
    print(f"Frames generated: {len(frames)}")
    
    # Free memory after generating
    torch.cuda.empty_cache()
    
    # Show first and last frame
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(frames[0])
    ax1.set_title("First frame")
    ax1.axis('off')
    ax2.imshow(frames[-1])
    ax2.set_title("Last frame")
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
    
    return frames

def save_images(images, base_filename="generated_image"):
    """
    Saves generated images to files.
    
    Args:
        images (list): List of images to save
        base_filename (str): Base name for files
    """
    for i, img in enumerate(images):
        filename = f"{base_filename}_{i+1}.png"
        img.save(filename)
        print(f"Image saved as: {filename}")

def save_video(frames, base_filename="generated_video", fps=8, duration=None):
    """
    Saves generated frames as a video and GIF.
    
    Args:
        frames (list): List of video frames
        base_filename (str): Base name for files
        fps (int): Frames per second
        duration (float): Desired duration in seconds (will adjust fps)
    """
    num_frames = len(frames)
    
    # If duration is specified, adjust fps
    if duration is not None:
        # Calculate fps to achieve desired duration
        adjusted_fps = num_frames / duration
        print(f"Adjusting FPS to {adjusted_fps:.1f} to achieve {duration} seconds duration")
        fps = adjusted_fps
    else:
        # Calculate resulting duration
        duration = num_frames / fps
        print(f"Video duration: {duration:.1f} seconds at {fps} FPS")
    
    # Save as GIF
    gif_filename = f"{base_filename}.gif"
    frames_pil = [Image.fromarray(np.uint8(frame)) for frame in frames]
    frames_pil[0].save(
        gif_filename,
        save_all=True,
        append_images=frames_pil[1:],
        duration=1000//fps,  # Duration of each frame in milliseconds
        loop=0
    )
    print(f"GIF saved as: {gif_filename}")
    
    # Save as MP4 if moviepy is available
    try:
        import moviepy.editor as mpy
        
        mp4_filename = f"{base_filename}.mp4"
        clip = mpy.ImageSequenceClip([np.array(frame) for frame in frames], fps=fps)
        clip.write_videofile(mp4_filename, codec='libx264', fps=fps)
        print(f"MP4 video saved as: {mp4_filename}")
    except ImportError:
        print("To save as MP4, install moviepy: pip install moviepy")
        print("For now, only GIF has been saved.")

def music_genre_to_prompt_free(genre):
    """
    Generates a descriptive prompt for a musical genre using free local models.
    
    Args:
        genre (str): Musical genre
        
    Returns:
        str: Descriptive prompt for image generation
    """
    try:
        from transformers import pipeline
        
        # First run will download the model (only once)
        print(f"Generating description for genre: {genre}")
        
        # Use a small model that works on CPU
        generator = pipeline('text-generation', 
                           model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                           device="cpu")
        
        prompt = f"""
        Please create a detailed visual description for the musical genre '{genre}'.
        Include visual elements, atmosphere, instruments, clothing, scenarios and emotions.
        The description should be suitable for generating an image with AI.
        Use exactly 40-60 words and do not include introductions or conclusions.
        """
        
        # Generate response
        response = generator(prompt, max_length=300, num_return_sequences=1)
        
        # Extract only the relevant part of the response
        generated_text = response[0]['generated_text']
        # Remove original prompt and extract only the description
        description = generated_text.split(prompt)[-1].strip()
        
        # Limit to approximately 60 words
        words = description.split()
        if len(words) > 70:  # A bit more for margin
            description = ' '.join(words[:60]) + '...'
            
        return description
            
    except Exception as e:
        print(f"Error generating automatic description: {str(e)}")
        print("Using fallback description...")
        
        # If it fails, use static dictionary as fallback
        return music_genre_to_prompt_static(genre)

# Rename original function to use as fallback
def music_genre_to_prompt_static(genre):
    """
    Static version with predefined dictionary of musical genres.
    
    Args:
        genre (str): Musical genre
        
    Returns:
        str: Descriptive prompt for image generation
    """
    # Dictionary of musical genres and their visual descriptions
    genre_prompts = {
        "rock": "Electric guitars on stage with dramatic lighting, rock concert atmosphere, energetic band performing, amplifiers and speakers, passionate musicians, crowd with raised hands",
        
        "pop": "Colorful stage with pop star performing, vibrant lighting effects, modern dance choreography, stylish outfits, large audience, digital screens, upbeat atmosphere",
        
        "reggaeton": "Urban street scene with reggaeton artists, tropical vibes, colorful urban fashion, dancing people, Caribbean influence, gold chains, urban nightlife, rhythm visualization",
        
        "punk": "Mohawk hairstyles, leather jackets with studs, underground club scene, anarchist symbols, intense mosh pit, DIY aesthetic, rebellious attitude, raw energy, graffiti walls",
        
        "classical": "Symphony orchestra in an elegant concert hall, conductor with baton, string instruments, sheet music, formal attire, sophisticated audience, architectural details, warm lighting",
        
        "jazz": "Dimly lit jazz club, saxophone player, double bass, piano, trumpet, smoky atmosphere, intimate setting, vintage microphone, audience enjoying cocktails",
        
        "electronic": "DJ booth with turntables and controllers, laser light show, futuristic visuals, dancing crowd with raised hands, neon colors, digital waveforms, immersive atmosphere",
        
        "hip hop": "Urban street with graffiti art, breakdancers, DJ with turntables, gold chains, urban fashion, concrete backdrop, expressive hand gestures, authentic street culture",
        
        "metal": "Dark concert stage with pyrotechnics, long-haired musicians headbanging, electric guitars, leather and spikes, intense facial expressions, dramatic lighting, devoted fans",
        
        "country": "Rural landscape with acoustic guitars, cowboy hats and boots, rustic wooden stage, American flag, denim clothing, warm sunset lighting, authentic countryside atmosphere",
        
        "indie": "Intimate venue with indie band, vintage instruments, warm ambient lighting, artistic atmosphere, thoughtful expressions, vinyl records, authentic and raw performance",
        
        "r&b": "Smooth stage lighting, soulful singer with microphone, elegant attire, emotional expressions, intimate atmosphere, urban sophistication, passionate performance",
        
        "folk": "Acoustic guitars and banjos, rustic outdoor setting, campfire, traditional clothing, storytelling atmosphere, natural landscape, authentic instruments, community gathering",
        
        "blues": "Old blues club with vintage microphone, emotional guitarist, dim blue lighting, whiskey glass on piano, soulful expressions, intimate atmosphere, authentic feeling",
        
        "reggae": "Vibrant Caribbean colors, dreadlocks, Jamaican flag, tropical setting, relaxed atmosphere, message of unity, Rastafarian symbols, sunshine and positive vibes",
        
        "disco": "Glitter ball reflecting colorful lights, dance floor with geometric patterns, 70s fashion with sequins, dynamic dance poses, vibrant colors, retro aesthetic",
        
        "techno": "Futuristic warehouse party, minimal lighting with laser beams, industrial setting, DJ with headphones, crowd in trance-like state, technological elements, smoke effects",
        
        "ambient": "Abstract flowing colors, minimalist landscape, ethereal lighting, peaceful natural elements, cosmic imagery, meditative atmosphere, subtle textures and patterns",
        
        "trap": "Urban night scene with luxury cars, modern jewelry, atmospheric lighting, urban fashion, confident poses, city skyline, moody atmosphere, high contrast visuals",
        
        "soul": "Vintage microphone, emotional singer, warm stage lighting, elegant attire, passionate expressions, intimate venue, authentic feeling, soulful atmosphere"
    }
    
    # Normalize genre (lowercase and remove extra spaces)
    genre_normalized = genre.lower().strip()
    
    # If genre is in our dictionary, use its description
    if genre_normalized in genre_prompts:
        return genre_prompts[genre_normalized]
    else:
        # For unrecognized genres, create a generic prompt
        return f"Visual representation of {genre} music, musical instruments, concert atmosphere, artistic interpretation of {genre} sound and culture"

# Main function that decides which method to use
def music_genre_to_prompt(genre):
    """
    Converts a musical genre into a descriptive prompt for image generation.
    
    Args:
        genre (str): Musical genre
        
    Returns:
        str: Descriptive prompt for image generation
    """
    # Ask user which method to use
    use_ai = input("Generate description automatically with AI? (y/n) [n]: ").strip().lower() or "n"
    
    if use_ai in ['y', 'yes', 's', 'si', 'sí']:
        return music_genre_to_prompt_free(genre)
    else:
        return music_genre_to_prompt_static(genre)

def extract_song_features(file_path):
    """
    Extracts detailed musical features from a song.
    
    Args:
        file_path (str): Path to audio file
        
    Returns:
        dict: Dictionary with musical features
    """
    try:
        import librosa
        import numpy as np
        
        print(f"Analyzing song features: {os.path.basename(file_path)}")
        
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)
        
        # 1. Extract tempo (BPM)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # 2. Extract key (major/minor)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_chroma = np.sum(chroma, axis=1)
        key_index = np.argmax(key_chroma)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_index]
        
        # 3. Extract energy
        energy = np.mean(librosa.feature.rms(y=y)[0])
        energy_percentile = np.mean(librosa.feature.rms(y=y)[0]) / 0.1  # Normalized to approximate scale 0-1
        energy_level = "high" if energy_percentile > 0.6 else "medium" if energy_percentile > 0.3 else "low"
        
        # 4. Extract brightness (presence of high frequencies)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        brightness = spectral_centroid / 4000  # Normalized approximately 0-1
        brightness_level = "bright" if brightness > 0.6 else "warm" if brightness > 0.3 else "dark"
        
        # 5. Detect vocals
        # Simple approximation based on energy in human voice frequency ranges
        spec = np.abs(librosa.stft(y))
        spec_voice = spec[(librosa.fft_frequencies(sr=sr) >= 300) & (librosa.fft_frequencies(sr=sr) <= 3000)]
        voice_energy = np.mean(spec_voice)
        has_vocals = voice_energy > 0.1  # Simple threshold
        
        # 6. Evaluate variability/dynamism
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        onset_density = len(onset_frames) / (len(y)/sr)  # Onsets per second
        dynamism = "dynamic" if onset_density > 2 else "moderate" if onset_density > 1 else "constant"
        
        # 7. Evaluate frequency balance (bass/treble)
        spec_low = np.mean(spec[librosa.fft_frequencies(sr=sr) < 500])
        spec_high = np.mean(spec[librosa.fft_frequencies(sr=sr) > 2000])
        bass_presence = "strong" if spec_low > 0.2 else "moderate" if spec_low > 0.1 else "light"
        treble_presence = "bright" if spec_high > 0.1 else "moderate" if spec_high > 0.05 else "soft"
        
        # Collect all features
        features = {
            "tempo": int(tempo),
            "tempo_category": "fast" if tempo > 120 else "medium" if tempo > 90 else "slow",
            "key": key,
            "energy": energy_level,
            "brightness": brightness_level,
            "has_vocals": has_vocals,
            "dynamism": dynamism,
            "bass_presence": bass_presence,
            "treble_presence": treble_presence
        }
        
        # Display extracted features
        for feature, value in features.items():
            print(f"{feature}: {value}")
        
        return features
        
    except Exception as e:
        print(f"Error analyzing song features: {str(e)}")
        return None

def extract_song_features_enhanced(file_path):
    """
    Enhanced version that extracts more detailed features, including
    instruments, emotions, and lyrics analysis if available.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        dict: Dictionary with extended musical features
    """
    try:
        import librosa
        import numpy as np
        import scipy
        from sklearn.preprocessing import StandardScaler
        import warnings
        warnings.filterwarnings('ignore')
        
        print(f"Performing advanced analysis of: {os.path.basename(file_path)}")
        
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)
        
        # 1. Extract tempo (BPM)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # 2. Extract key (major/minor)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_chroma = np.sum(chroma, axis=1)
        key_index = np.argmax(key_chroma)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_index]
        
        # 3. Extract energy
        energy = np.mean(librosa.feature.rms(y=y)[0])
        energy_percentile = np.mean(librosa.feature.rms(y=y)[0]) / 0.1  # Normalized to approximate scale 0-1
        energy_level = "high" if energy_percentile > 0.6 else "medium" if energy_percentile > 0.3 else "low"
        
        # 4. Extract brightness (presence of high frequencies)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        brightness = spectral_centroid / 4000  # Normalized approximately 0-1
        brightness_level = "bright" if brightness > 0.6 else "warm" if brightness > 0.3 else "dark"
        
        # 5. Detect vocals
        # Simple approximation based on energy in human voice frequency ranges
        spec = np.abs(librosa.stft(y))
        spec_voice = spec[(librosa.fft_frequencies(sr=sr) >= 300) & (librosa.fft_frequencies(sr=sr) <= 3000)]
        voice_energy = np.mean(spec_voice)
        has_vocals = voice_energy > 0.1  # Simple threshold
        
        # 6. Evaluate variability/dynamism
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        onset_density = len(onset_frames) / (len(y)/sr)  # Onsets per second
        dynamism = "dynamic" if onset_density > 2 else "moderate" if onset_density > 1 else "constant"
        
        # 7. Evaluate frequency balance (bass/treble)
        spec_low = np.mean(spec[librosa.fft_frequencies(sr=sr) < 500])
        spec_high = np.mean(spec[librosa.fft_frequencies(sr=sr) > 2000])
        bass_presence = "strong" if spec_low > 0.2 else "moderate" if spec_low > 0.1 else "light"
        treble_presence = "bright" if spec_high > 0.1 else "moderate" if spec_high > 0.05 else "soft"
        
        # === INSTRUMENT DETECTION ===
        # Frequency band analysis for common instruments
        instruments = []
        
        # Extract spectral features
        freq_bands = librosa.fft_frequencies(sr=sr)
        
        # Approximate frequency bands for various instruments
        instrument_bands = {
            'bass': (40, 250),
            'kick drum': (50, 100),
            'electric guitar': (300, 4000),
            'piano': (27, 4200),
            'synthesizer': (80, 8000),
            'strings': (200, 3500),
            'trumpet': (160, 980),
            'saxophone': (100, 900),
            'drums': (200, 12000),
        }
        
        # Detect instrument presence by energy in their typical bands
        for instrument, (low_freq, high_freq) in instrument_bands.items():
            band_energy = np.mean(spec[(freq_bands >= low_freq) & (freq_bands <= high_freq)])
            instrument_energy = band_energy / np.mean(spec)  # Normalize
            if instrument_energy > 0.6:
                instruments.append(instrument)
        
        # === DETAILED EMOTIONAL ANALYSIS ===
        # Basic model of musical emotions based on acoustic features
        
        # Extract additional features for emotional analysis
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        
        # Calculate features for emotional mapping
        tempo_norm = (tempo - 60) / 120  # Normalize tempo (60-180 BPM)
        
        # Major/minor modes (approximation)
        major_minor_ratio = np.std(chroma) / np.mean(chroma)
        is_major = major_minor_ratio < 0.15
        
        # Presence of dissonance
        dissonance = np.mean(spectral_contrast[1:3])
        
        # Calculate emotional values
        emotions = {}
        
        # Valence (positive vs negative)
        emotions['valence'] = 0.6 if is_major else 0.3  # Based on major/minor mode
        emotions['valence'] += 0.2 * energy_percentile  # High energy → more positive
        emotions['valence'] -= 0.1 * dissonance  # Dissonance → more negative
        emotions['valence'] = max(0, min(1, emotions['valence']))  # Limit 0-1
        
        # Arousal (activation/intensity)
        emotions['arousal'] = 0.3 * energy_percentile + 0.3 * tempo_norm + 0.2 * dissonance
        emotions['arousal'] = max(0, min(1, emotions['arousal']))  # Limit 0-1
        
        # Tension
        emotions['tension'] = 0.4 * dissonance + 0.2 * (1 - energy_percentile) + 0.2 * tempo_norm
        emotions['tension'] = max(0, min(1, emotions['tension']))  # Limit 0-1
        
        # Classify emotions into categories
        emotion_categories = []
        
        # Emotional quadrants based on Russell's model
        if emotions['valence'] > 0.5 and emotions['arousal'] > 0.5:
            emotion_categories.append('joyful')
            emotion_categories.append('energetic')
        elif emotions['valence'] > 0.5 and emotions['arousal'] <= 0.5:
            emotion_categories.append('relaxed')
            emotion_categories.append('serene')
        elif emotions['valence'] <= 0.5 and emotions['arousal'] > 0.5:
            emotion_categories.append('tense')
            emotion_categories.append('restless')
        else:
            emotion_categories.append('melancholic')
            emotion_categories.append('gloomy')
        
        # Add specific characteristics
        if emotions['tension'] > 0.7:
            emotion_categories.append('dramatic')
        
        if tempo > 140 and energy_percentile > 0.7:
            emotion_categories.append('euphoric')
        
        if tempo < 80 and energy_percentile < 0.4:
            emotion_categories.append('introspective')
        
        # === SECTION DETECTION (structure) ===
        # Simple analysis of song structure
        
        # Segment the song based on changes in features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Calculate spectral novelty
        novelty = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(novelty, 3, 3, 3, 5, 0.5, 10)
        
        # Identify significant changes (possible sections)
        if len(peaks) > 2:
            has_structure_changes = True
            structure_complexity = "complex" if len(peaks) > 5 else "moderate"
        else:
            has_structure_changes = False
            structure_complexity = "simple"
        
        # === VOCAL CLARITY ANALYSIS ===
        vocal_clarity = "not detected"
        if has_vocals:
            # Estimate vocal clarity based on energy in vocal frequency bands
            vocal_bands = spec[(freq_bands >= 300) & (freq_bands <= 3000)]
            other_bands = spec[~((freq_bands >= 300) & (freq_bands <= 3000))]
            
            vocal_to_instrumental_ratio = np.mean(vocal_bands) / np.mean(other_bands)
            
            if vocal_to_instrumental_ratio > 1.5:
                vocal_clarity = "prominent"
            elif vocal_to_instrumental_ratio > 0.8:
                vocal_clarity = "clear"
            else:
                vocal_clarity = "fuzzy"
        
        # Collect all enhanced features
        features = {
            # Basic features
            "tempo": int(tempo),
            "tempo_category": "fast" if tempo > 120 else "medium" if tempo > 90 else "slow",
            "key": key,
            "energy": energy_level,
            "brightness": brightness_level,
            "has_vocals": has_vocals,
            "dynamism": dynamism,
            "bass_presence": bass_presence,
            "treble_presence": treble_presence,
            
            # New features
            "instruments": instruments,
            "emotions": emotion_categories,
            "structure": structure_complexity,
            "vocal_clarity": vocal_clarity,
            "emotion_values": emotions
        }
        
        # Display extracted advanced features
        for feature, value in features.items():
            print(f"{feature}: {value}")
        
        return features
        
    except Exception as e:
        print(f"Error in advanced analysis: {str(e)}")
        print("Falling back to basic analysis...")
        # Attempt basic analysis as a fallback
        return extract_song_features(file_path)

def song_to_prompt(file_path, genre):
    """
    Generates a descriptive prompt for a specific song, considering
    both the genre and the specific audio features.
    
    Args:
        file_path (str): Path to the audio file
        genre (str): Detected musical genre
        
    Returns:
        str: Descriptive prompt for image generation
    """
    # Extract specific features of the song
    features = extract_song_features_enhanced(file_path)
    
    if features is None:
        print("Could not extract specific features. Using only the genre.")
        return music_genre_to_prompt_static(genre)
    
    # Use the static method as a base to ensure a useful result
    base_prompt = music_genre_to_prompt_static(genre)
    
    # Build an enriched description based on the extracted features
    instruments_str = ", ".join(features.get("instruments", [])) if features.get("instruments") else "typical instruments"
    emotions_str = ", ".join(features.get("emotions", [])) if features.get("emotions") else "characteristic emotions"
    
    # Build an enhanced prompt manually (without relying on AI for this part)
    enhanced_prompt = f"{base_prompt}, with {features['tempo_category']} tempo ({features['tempo']} BPM), high energy"
    
    # Add details of detected instruments
    if features.get("instruments"):
        enhanced_prompt += f", highlighting {instruments_str}"
    
    # Add description of detected emotions
    if features.get("emotions"):
        enhanced_prompt += f", atmosphere {emotions_str}"
    
    # Add information about vocals
    if features.get("has_vocals", False):
        enhanced_prompt += f", with {features.get('vocal_clarity', 'prominent')} vocals"
    else:
        enhanced_prompt += ", instrumental composition"
    
    # Add sound characteristics
    enhanced_prompt += f", {features['brightness']} sound with {features['bass_presence']} bass presence"
    
    print(f"Enhanced prompt created with specific features of the song")
    return enhanced_prompt

def auto_detect_genre(file_path):
    """
    Auto-detects genre using the audio classifier with model selection.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        str or None: Detected genre or None if failed
    """
    try:
        from audio_classifier import load_model_and_params, classify_audio, list_available_models
        
        # Check if there are available models
        available_models = list_available_models()
        if not available_models:
            print("No trained models found for genre detection.")
            return None
        
        # Model selection logic (like in audio_classifier.py)
        model_name = None
        
        if len(available_models) == 1:
            model_name = available_models[0]
            print(f"Using the only available model: '{model_name}'")
        else:
            print("\nAvailable models for genre detection:")
            for i, model in enumerate(available_models, 1):
                # Try to show model info if available
                try:
                    import json
                    info_path = f"models/{model}/model_info.json"
                    if os.path.exists(info_path):
                        with open(info_path, "r") as f:
                            info = json.load(f)
                        training_date = info.get('training_date', 'unknown')
                        hyperparams = info.get('hyperparameters', {})
                        epochs = hyperparams.get('epochs', 'N/A')
                        print(f"  {i}. {model} (trained: {training_date}, epochs: {epochs})")
                    else:
                        print(f"  {i}. {model}")
                except:
                    print(f"  {i}. {model}")
            
            # Ask user to select model
            while True:
                choice = input(f"\nSelect model (1-{len(available_models)}) [1]: ").strip() or "1"
                
                try:
                    model_idx = int(choice) - 1
                    if 0 <= model_idx < len(available_models):
                        model_name = available_models[model_idx]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(available_models)}")
                except ValueError:
                    print("Please enter a valid number")
        
        if not model_name:
            print("No model selected.")
            return None
        
        print(f"Using model '{model_name}' for genre detection...")
        
        # Load the selected model
        model, encoder, params = load_model_and_params(model_name)
        if model is not None:
            detected_genre, probabilities, _ = classify_audio(file_path, model, encoder, params)
            if detected_genre:
                print(f"Detected genre: {detected_genre.upper()}")
                
                # Show top 3 probabilities for user reference
                print("Genre probabilities:")
                sorted_indices = np.argsort(probabilities)[::-1][:3]
                for i in sorted_indices:
                    print(f"  {params['genres'][i]}: {probabilities[i]*100:.1f}%")
                
                return detected_genre
            else:
                print("Could not detect genre.")
                return None
        else:
            print("Could not load classifier model.")
            return None
    except ImportError:
        print("Audio classifier module not available.")
        print("Make sure you have trained a model using 'audio_classifier_trainer.py'")
        return None
    except Exception as e:
        print(f"Error detecting genre: {str(e)}")
        return None

def main():
    """Main function that runs the image and video generation program"""
    print("=== Image and Video Generator with Stable Diffusion ===")
    
    # Configure PyTorch for better memory management
    torch.cuda.empty_cache()
    
    # Ask what type of content to generate
    print("\nWhat type of content do you want to generate?")
    print("1: Images")
    print("2: Videos (from generated images)")
    
    content_choice = input("\nSelect an option (1-2) [1]: ").strip() or "1"
    generate_video_content = content_choice == "2"
    
    if generate_video_content:
        print("\nTo generate videos, we'll first create an image and then animate it.")
        
        # Show memory warning for videos
        if torch.cuda.is_available():
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\nWARNING! Video generation requires a lot of GPU memory.")
            print(f"Your GPU has {gpu_mem_total:.2f} GB of total memory.")
            print("It's recommended to close other GPU-using applications before continuing.")
            print("Also recommended to use lower resolution images (384x384) for videos.")
    
    # Load image model with Dreamlike Diffusion
    image_pipe, is_cpu = setup_model("image")
    if image_pipe is None:
        print("Program terminated by user.")
        return
    
    # Load video model if necessary
    video_pipe = None
    if generate_video_content:
        print("\nLoading video generation model...")
        video_pipe, video_is_cpu = setup_model("video")
        if video_pipe is None:
            print("Program terminated by user.")
            return
    
    # Ask if you want to use music genre mode
    print("\nWhat mode do you want to use?")
    print("1: Normal mode (descriptive prompt)")
    print("2: Music genre mode (convert musical genre to image/video)")
    
    mode_choice = input("\nSelect a mode (1-2) [1]: ").strip() or "1"
    music_mode = mode_choice == "2"
    
    while True:
        # Free memory before each iteration
        torch.cuda.empty_cache()
        
        if music_mode:
            print("\nWhat method do you want to use?")
            print("1: Only musical genre")
            print("2: Complete song analysis (personalized)")
            
            analysis_choice = input("\nSelect a method (1-2) [1]: ").strip() or "1"
            song_analysis = analysis_choice == "2"
            
            if song_analysis:
                # Request audio file
                file_path = input("\nPath to audio file: ")
                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist.")
                    continue
                    
                # Request genre (or detect automatically if you have the classifier)
                genre = input("\nMusical genre (leave empty to auto-detect): ")
                
                if not genre:
                    # Auto-detect genre using the updated function
                    detected_genre = auto_detect_genre(file_path)
                    if detected_genre:
                        genre = detected_genre
                        
                        # Ask user to confirm or modify the detected genre
                        confirm = input(f"\nUse detected genre '{genre}'? (y/n/edit) [y]: ").strip().lower() or "y"
                        if confirm == 'n':
                            genre = input("Please enter genre manually: ")
                        elif confirm == 'edit':
                            genre = input(f"Edit genre [{genre}]: ").strip() or genre
                    else:
                        genre = input("Could not detect genre. Please enter it manually: ")
                
                # Generate prompt based on complete song analysis
                prompt = song_to_prompt(file_path, genre)
                print(f"\nSong analyzed: {os.path.basename(file_path)}")
                print(f"Genre: {genre}")
                print(f"Generated prompt: {prompt}")
            else:
                # Original method based only on genre
                genre = input("\nEnter a musical genre (or 'exit' to quit): ")
                if genre.lower() in ['exit', 'quit', 'salir']:
                    break
                    
                # Convert genre to prompt
                prompt = music_genre_to_prompt(genre)
                print(f"\nGenre: {genre}")
                print(f"Generated prompt: {prompt}")
        
            # Ask if you want to edit the prompt
            edit_option = input("\nDo you want to edit the prompt? (y/n) [n]: ").strip().lower() or "n"
            if edit_option in ['y', 'yes', 's', 'si', 'sí']:
                prompt = input(f"Edit the prompt: ").strip() or prompt
        else:
            # Normal mode: request prompt directly
            prompt = input("\nDescribe the image you want to generate (or 'exit' to quit): ")
            if prompt.lower() in ['exit', 'quit', 'salir']:
                break
        
        # Configure parameters for the image
        print("\nImage parameter configuration (press Enter to use defaults)")
        
        # Adjust default values for CPU
        default_steps = 25 if is_cpu else 50
        default_width = 384 if is_cpu else 512
        default_height = 384 if is_cpu else 512
        
        steps = input(f"Number of inference steps (1-150) [{default_steps}]: ").strip()
        steps = int(steps) if steps.isdigit() else default_steps
        
        width = input(f"Image width (256-1024) [{default_width}]: ").strip()
        width = int(width) if width.isdigit() else default_width
        
        height = input(f"Image height (256-1024) [{default_height}]: ").strip()
        height = int(height) if height.isdigit() else default_height
        
        num_images = input("Number of images to generate (1-4) [1]: ").strip()
        num_images = int(num_images) if num_images.isdigit() else 1
        
        negative_prompt = input("Negative prompt (features to avoid): ").strip()
        
        # Build parameter dictionary for the image
        image_params = {
            'num_inference_steps': steps,
            'width': width,
            'height': height,
            'num_images_per_prompt': num_images
        }
        
        if negative_prompt:
            image_params['negative_prompt'] = negative_prompt
        
        # Generate images
        images = generate_image(image_pipe, prompt, image_params, is_cpu)
        
        # Ask if you want to save the images
        save_option = input("\nDo you want to save the images? (y/n) [y]: ").strip().lower() or "y"
        if save_option in ['y', 'yes', 's', 'si', 'sí']:
            # If we're in music mode, use the genre as base name
            default_name = genre if music_mode else "generated_image"
            base_name = input(f"Base name for files [{default_name}]: ").strip() or default_name
            save_images(images, base_name)
        
        # If video generation was chosen, proceed with video generation
        if generate_video_content:
            print("\n=== Video Generation ===")
            
            # Select which image to use as base
            image_idx = 0
            if len(images) > 1:
                image_idx_input = input(f"Which image do you want to animate? (1-{len(images)}) [1]: ").strip()
                image_idx = int(image_idx_input) - 1 if image_idx_input.isdigit() and 1 <= int(image_idx_input) <= len(images) else 0
            
            selected_image = images[image_idx]
            
            # Configure video parameters
            print("\nVideo parameter configuration (press Enter to use defaults)")
            
            # Adjust default values to save memory
            default_video_steps = 20 if video_is_cpu else 25  # Reduced to save memory
            default_motion_bucket_id = 127  # Medium value for movement
            default_duration = 1.0  # Default duration in seconds
            
            # Suggest lower resolution for videos
            current_width, current_height = selected_image.size
            if current_width > 512 or current_height > 512:
                print(f"\nWARNING! The selected image is large ({current_width}x{current_height}).")
                print("This may cause memory issues during video generation.")
                resize_option = input("Do you want to reduce resolution to 512x512 to save memory? (y/n) [y]: ").strip().lower() or "y"
                if resize_option in ['y', 'yes', 's', 'si', 'sí']:
                    # Maintain aspect ratio
                    if current_width > current_height:
                        new_width = 512
                        new_height = int(current_height * (512 / current_width))
                    else:
                        new_height = 512
                        new_width = int(current_width * (512 / current_height))
                    
                    selected_image = selected_image.resize((new_width, new_height), Image.LANCZOS)
                    print(f"Image resized to {new_width}x{new_height}")
            
            video_steps = input(f"Number of inference steps (1-50) [{default_video_steps}]: ").strip()
            video_steps = int(video_steps) if video_steps.isdigit() else default_video_steps
            
            motion_bucket_id = input(f"Movement intensity (1-255) [{default_motion_bucket_id}]: ").strip()
            motion_bucket_id = int(motion_bucket_id) if motion_bucket_id.isdigit() else default_motion_bucket_id
            
            # Ask for desired duration
            duration_input = input(f"Desired video duration in seconds (0.5-10) [{default_duration}]: ").strip()
            try:
                duration = float(duration_input) if duration_input else default_duration
                # Limit to reasonable range
                duration = max(0.5, min(10, duration))
            except ValueError:
                duration = default_duration
                print(f"Invalid value, using default duration: {default_duration} seconds")
            
            # Build parameter dictionary for the video
            video_params = {
                'num_inference_steps': video_steps,
                'motion_bucket_id': motion_bucket_id,
                'decode_chunk_size': 1,  # Process in chunks to save memory
            }
            
            # Generate video
            frames = generate_video(video_pipe, selected_image, video_params, video_is_cpu)
            
            # Check if generation was successful
            if frames is not None:
                # Ask if you want to save the video
                save_video_option = input("\nDo you want to save the video? (y/n) [y]: ").strip().lower() or "y"
                if save_video_option in ['y', 'yes', 's', 'si', 'sí']:
                    # If we're in music mode, use the genre as base name
                    default_video_name = f"{genre}_video" if music_mode else "generated_video"
                    video_base_name = input(f"Base name for video [{default_video_name}]: ").strip() or default_video_name
                    
                    # Save the video with specified duration
                    save_video(frames, video_base_name, duration=duration)
            else:
                print("\nCould not generate video due to memory issues.")
                print("Try with a smaller image or fewer inference steps.")
        
        # Ask if you want to generate another image/video
        continue_option = input("\nDo you want to generate another image/video? (y/n) [y]: ").strip().lower() or "y"
        if continue_option not in ['y', 'yes', 's', 'si', 'sí']:
            break

if __name__ == "__main__":
    # Check if necessary libraries are installed
    try:
        import diffusers
        import transformers
        import torch
    except ImportError:
        print("Installing necessary libraries...")
        import subprocess
        subprocess.check_call(["pip", "install", "diffusers", "transformers", "accelerate", "torch"])
        print("Libraries installed correctly.")
    
    # Check if moviepy is installed to save videos
    try:
        import moviepy.editor
    except ImportError:
        print("Note: To save videos in MP4 format, it's recommended to install moviepy:")
        print("pip install moviepy")
        print("For now, videos will be saved only as GIF.")
    
    main()
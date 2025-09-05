# Sound to Something 🎵

A pilot AI-powered system that explores converting audio into visual art by classifying music genres and generating corresponding images and videos using Stable Diffusion models.

## 🌟 Features

- **Music Genre Classification**: Classify audio files into 10 different genres using a trained neural network
- **Text-to-Image Generation**: Generate artistic images based on the detected music genre
- **Text-to-Video Generation**: Create videos from generated images using Stable Video Diffusion
- **Interactive Web Interface**: User-friendly Streamlit application for easy interaction
- **Audio Visualization**: Display waveforms and spectrograms of uploaded audio files
- **Model Training**: Train custom music classification models with the GTZAN dataset

## 🎯 Supported Music Genres

- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## 🏗️ Project Structure

```
Sound_To_Something/
├── app_streamlit.py              # Main Streamlit web application
├── audio_classifier.py          # Music genre classification module
├── audio_classifier_trainer.py  # Model training script
├── text_to_image_generator.py   # Image/video generation module
├── requirements.txt             # Python dependencies
├── genres_original/             # GTZAN dataset (10 genres, ~100 files each)
├── models/                      # Trained classification models
│   ├── v1.0/, v1.1/, ..., v2.1/ # Model versions
│   └── demostracion/            # Demo model
├── results/                     # Generated images and videos
├── temp/                        # Temporary files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- At least 8GB RAM

### Installation

1. **Clone or download the project**
   ```bash
   cd Sound_To_Something
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   For CUDA support (recommended):
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run app_streamlit.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage

### Web Interface (Recommended)

1. **Launch the app**: Run `streamlit run app_streamlit.py`
2. **Upload audio**: Click "Browse files" and select an audio file (.mp3, .wav)
3. **Analyze**: The app will:
   - Display audio waveform and spectrogram
   - Classify the music genre
   - Generate an artistic image based on the genre
   - Optionally create a video from the image
4. **Download results**: Save generated images and videos

### Command Line Interface

#### Classify Audio
```bash
python audio_classifier.py
```
Follow the prompts to select a model and audio file.

#### Generate Images
```bash
python text_to_image_generator.py
```

#### Train New Models
```bash
python audio_classifier_trainer.py
```

## 🧠 How It Works

1. **Audio Processing**: Extract MFCC (Mel-Frequency Cepstral Coefficients) features from audio files
2. **Genre Classification**: Use a trained neural network to predict the music genre
3. **Prompt Generation**: Convert the detected genre into descriptive text prompts
4. **Image Generation**: Use Stable Diffusion to create artistic images from prompts
5. **Video Generation**: Transform static images into dynamic videos using Stable Video Diffusion

## 🎨 Model Information

### Audio Classification Models
- **Architecture**: Dense Neural Network with dropout layers
- **Input**: MFCC features (mean + standard deviation)
- **Output**: 10 genre probabilities
- **Best Model**: v2.1 with 74.5% test accuracy
- **Training Data**: GTZAN dataset (1000 audio tracks, 30 seconds each)

### Image Generation Models
- **Primary**: Dreamlike Diffusion v1.0
- **Technology**: Stable Diffusion
- **Capabilities**: High-quality artistic image generation

### Video Generation Models
- **Technology**: Stable Video Diffusion
- **Input**: Static images
- **Output**: Short video sequences

## 📊 Model Performance

| Model Version | Test Accuracy | Training Date | Epochs |
|---------------|---------------|---------------|---------|
| v1.0          | ~65%          | Early version | 100     |
| v1.4          | ~70%          | Mid version   | 100     |
| v2.1          | 74.5%         | 2025-05-31    | 200     |

## 🛠️ Configuration

### Hyperparameters (Training)
```python
DEFAULT_HYPERPARAMS = {
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "early_stopping_patience": 15,
    "dropout_rate": 0.3,
    "validation_split": 0.2
}
```

### Generation Parameters
- **Image Size**: 512x512 pixels (configurable)
- **Inference Steps**: 20-50 (quality vs speed trade-off)
- **Guidance Scale**: 7.5 (creativity vs prompt adherence)

## 📁 File Formats

### Supported Audio Formats
- MP3
- WAV
- M4A
- FLAC

### Output Formats
- **Images**: PNG
- **Videos**: MP4, GIF

## 🔧 Customization

### Adding New Genres
1. Add audio samples to `genres_original/new_genre/`
2. Update `GENRES` list in `audio_classifier_trainer.py`
3. Retrain the model

### Custom Prompts
Modify the `music_genre_to_prompt_static()` function in `text_to_image_generator.py` to customize how genres are converted to image prompts.

## 📈 Performance Tips

### For Better Speed
- Use CUDA-enabled GPU
- Reduce image resolution
- Lower inference steps
- Use CPU for classification, GPU for generation

### For Better Quality
- Increase inference steps (30-50)
- Use higher resolution (768x768)
- Experiment with different guidance scales
- Fine-tune prompts for specific genres

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Error**
```bash
torch.cuda.OutOfMemoryError
```
- Reduce batch size or image resolution
- Close other GPU-intensive applications

**Model Not Found**
```bash
No trained models found
```
- Run `audio_classifier_trainer.py` first
- Check `models/` directory exists

**Audio Loading Error**
```bash
librosa.util.exceptions.ParameterError
```
- Ensure audio file is valid
- Try converting to WAV format

### Dependencies Issues
```bash
pip install --upgrade torch diffusers transformers librosa
```

## 📄 License

This project is for educational and research purposes. Please ensure you have the right to use any audio files you process.

## 🤝 Contributing

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the code comments
- Create an issue with detailed error information

## 🙏 Acknowledgments

- **GTZAN Dataset**: George Tzanetakis for the music genre dataset
- **Stable Diffusion**: Stability AI for the diffusion models
- **Librosa**: For audio processing capabilities
- **Streamlit**: For the web interface framework

---

**🎵 Turn your music into art! 🎨**

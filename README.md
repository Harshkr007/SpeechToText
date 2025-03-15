# Multilingual Speech-to-Text Transcription

A robust multilingual speech-to-text transcription system based on the Whisper model, fine-tuned for Indian languages.

## Supported Languages
- Hindi (hi)
- Kannada (kn)
- English (en)
- Tamil (ta)
- Sanskrit (sa)

## Project Structure
```
SpeechToText/
├── backend/                     # FastAPI backend service
│   ├── model/                  
│   │   └── transcriber.py      # Audio transcription logic
│   ├── routes/                 
│   │   └── transcribe.py       # API endpoints
│   └── main.py                 # FastAPI application
│
├── training/                    # Model training components
│   ├── config/
│   │   └── training_config.py  # Training configuration
│   ├── data_preprocessing/
│   │   └── prepare_dataset.py  # Dataset preparation
│   └── training/
│       └── train.py            # Training script
│
├── finetuned_multilingual/     # Directory for trained models
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

## Model Details
- Base Model: `openai/whisper-small`
- Training Dataset: `ai4bharat/indicvoices_r`
- Fine-tuning Approach: Multilingual fine-tuning
- Model Type: Sequence-to-Sequence with Attention

### Model Architecture
- Encoder: Convolutional Neural Network + Transformer
- Decoder: Transformer with Cross-Attention
- Parameters: ~244M (small variant)

## Setup and Installation

1. Clone the repository
```bash
git clone <repository-url>
cd SpeechToText
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Ubuntu Setup Guide

1. System Requirements:
```bash
# Update system packages
sudo apt update
sudo apt upgrade

# Install Python and required system packages
sudo apt install python3.8 python3.8-venv python3-pip git ffmpeg
```

2. Project Setup:
```bash
# Navigate to project directory
cd SpeechToText

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install pip tools
pip install --upgrade pip setuptools wheel

# Install project dependencies
pip install -r requirements.txt
```

3. Environment Setup:
```bash
# Create and edit .env file
touch .env
echo "MODEL_PATH=./backend/model/cached_model" >> .env
echo "CUDA_VISIBLE_DEVICES=0" >> .env  # If using GPU
```

4. Model Training (Optional):
```bash
# Navigate to training directory
cd training

# Start training
python3 training/train.py

# Wait for training to complete
# Trained model will be saved in finetuned_multilingual directory
```

5. Run API Server:
```bash
# Navigate to backend directory
cd backend

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

6. Test API:
```bash
# In a new terminal (with venv activated)
# Test server health
curl http://localhost:8000/

# Test transcription (replace audio.wav with your file)
curl -X POST http://localhost:8000/transcribe/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.wav" \
  -F "language=hi"
```

### Troubleshooting

1. CUDA Issues:
```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"

# If CUDA not found, install CUDA drivers:
sudo ubuntu-drivers autoinstall
sudo reboot
```

2. Audio Processing Issues:
```bash
# Install additional audio processing libraries
sudo apt install libsndfile1-dev
```

3. Permission Issues:
```bash
# Fix directory permissions
chmod -R 755 .
```

4. Memory Issues:
```bash
# Add swap space if needed
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Development Tools (Optional)

Install helpful development tools:
```bash
# Install development tools
sudo apt install git-all
sudo apt install tmux
sudo apt install htop

# Install VS Code (if not already installed)
sudo snap install code --classic
```

## Usage

### Training the Model
```bash
cd training
python training/train.py
```

### Running the API Server
```bash
cd backend
uvicorn main:app --reload
```

### API Endpoints
- POST `/transcribe/`
  - Parameters:
    - `file`: WAV audio file (required)
    - `language`: Language code (optional)
  - Returns:
    - `transcription`: Transcribed text
    - `language`: Detected or specified language

## Performance

The model is fine-tuned on the ai4bharat/indicvoices_r dataset with the following configurations:
- Training Epochs: 30
- Batch Size: 16
- Learning Rate: 1e-5
- Evaluation Metric: Word Error Rate (WER)

## Model Caching

The trained model is automatically cached in `backend/model/cached_model/` for faster loading during inference.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Your chosen license]

## Acknowledgments

- OpenAI Whisper team for the base model
- AI4Bharat for the Indian languages dataset
- Hugging Face for the transformers library

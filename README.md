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

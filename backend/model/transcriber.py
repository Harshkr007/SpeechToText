import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import io
import os

# Define model cache directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cached_model")
MODEL_ID = "openai/whisper-small"  # We'll replace this with our fine-tuned model later

def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            print("Loading model from local cache...")
            processor = WhisperProcessor.from_pretrained(MODEL_PATH)
            model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
        else:
            print("Downloading model for the first time...")
            processor = WhisperProcessor.from_pretrained(MODEL_ID)
            model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
            
            os.makedirs(MODEL_PATH, exist_ok=True)
            processor.save_pretrained(MODEL_PATH)
            model.save_pretrained(MODEL_PATH)
            
        return processor, model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

processor, model = load_model()

async def transcribe_audio(file, language=None):
    try:
        contents = await file.read()
        audio_stream = io.BytesIO(contents)
        
        try:
            waveform, sample_rate = torchaudio.load(audio_stream)
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")
        
        try:
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            input_features = processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features
        except Exception as e:
            raise Exception(f"Error preprocessing audio: {str(e)}")
        
        try:
            # Generate tokens with language forcing if specified
            with torch.no_grad():
                forced_decoder_ids = None
                if language:
                    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
                predicted_ids = model.generate(
                    input_features, 
                    forced_decoder_ids=forced_decoder_ids,
                    language=language
                )
            
            # Decode tokens
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        except Exception as e:
            raise Exception(f"Error during transcription: {str(e)}")
        
        return transcription
        
    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")
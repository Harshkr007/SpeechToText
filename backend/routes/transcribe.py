from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
from model.transcriber import transcribe_audio

router = APIRouter()

@router.post("/transcribe/")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = None
):
    # Check if file is provided
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    allowed_types = ["audio/wav", "audio/wave", "audio/x-wav"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Must be WAV file. Got {file.content_type}"
        )
    
    # Validate language code if provided
    valid_languages = ["hi", "kn", "en", "ta", "sa"]
    if language and language not in valid_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language code. Must be one of {valid_languages}"
        )
    
    try:
        text = await transcribe_audio(file, language)
        return {
            "transcription": text,
            "language": language or "auto-detected"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

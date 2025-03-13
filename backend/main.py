from fastapi import FastAPI
from routes.transcribe import router as transcribe_router

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Audio to Text API is running!"}

app.include_router(transcribe_router)

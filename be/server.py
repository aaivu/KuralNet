from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Union, Any
import numpy as np
import librosa
import soundfile as sf
import io
import tempfile
import os
import uuid
from pydantic import BaseModel
import random

app = FastAPI(title="Audio Emotion Recognition API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for responses
class EmotionScores(BaseModel):
    angry: float
    sad: float
    fear: float
    happy: float
    neutral: float

class EmotionSegment(BaseModel):
    start: float
    end: float
    mainEmotion: str
    emotions: EmotionScores

class PredictionResponse(BaseModel):
    segments: List[EmotionSegment]
    overallEmotion: str
    audioLength: float

# Audio chunking utilities
def find_silence_points(audio_data: np.ndarray, sr: int, 
                        min_silence_len: float = 0.3, 
                        silence_thresh: float = -40) -> List[float]:
    """
    Find potential chunking points at silence regions.
    
    Args:
        audio_data: Audio signal
        sr: Sample rate
        min_silence_len: Minimum silence length in seconds
        silence_thresh: Threshold in dB for silence detection
        
    Returns:
        List of timestamps (in seconds) for silence regions
    """
    # Convert to dB
    audio_db = librosa.amplitude_to_db(np.abs(audio_data), ref=np.max)
    
    # Find silence regions
    silence_mask = audio_db < silence_thresh
    
    # Find silence regions of sufficient length
    min_samples = int(min_silence_len * sr)
    silence_regions = []
    
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silence_mask):
        if is_silent and not in_silence:
            in_silence = True
            silence_start = i
        elif not is_silent and in_silence:
            in_silence = False
            silence_duration = i - silence_start
            if silence_duration >= min_samples:
                # Add the middle point of the silence as a possible chunk boundary
                silence_regions.append((silence_start + silence_duration // 2) / sr)
    
    return silence_regions

def chunk_audio(audio_data: np.ndarray, sr: int, max_chunk_len: int = 60, 
                overlap: float = 0.5) -> List[Dict[str, Any]]:
    """
    Chunk audio into segments optimally.
    
    Args:
        audio_data: Audio signal
        sr: Sample rate
        max_chunk_len: Maximum chunk length in seconds
        overlap: Overlap between chunks in seconds
        
    Returns:
        List of chunks with start and end times
    """
    audio_length = len(audio_data) / sr
    
    # If audio is already shorter than max_chunk_len, return as is
    if audio_length <= max_chunk_len:
        return [{"start": 0, "end": audio_length, "audio": audio_data}]
    
    # Find silence points to use as natural chunk boundaries
    silence_points = find_silence_points(audio_data, sr)
    
    chunks = []
    current_pos = 0
    
    while current_pos < audio_length:
        # Calculate the ideal end position for this chunk
        ideal_end = min(current_pos + max_chunk_len, audio_length)
        
        # Look for silence points near the ideal end
        best_end = ideal_end
        
        if silence_points:
            # Find the closest silence point to the ideal end
            closest_silence = min(silence_points, key=lambda x: abs(x - ideal_end))
            
            # Use the silence point if it's reasonably close to the ideal end
            if abs(closest_silence - ideal_end) < max_chunk_len * 0.2:  # Within 20% of chunk length
                best_end = closest_silence
        
        # Extract the chunk
        chunk_start_samples = int(current_pos * sr)
        chunk_end_samples = int(best_end * sr)
        chunk_audio = audio_data[chunk_start_samples:chunk_end_samples]
        
        chunks.append({
            "start": current_pos,
            "end": best_end,
            "audio": chunk_audio
        })
        
        # Move to next chunk with the specified overlap
        current_pos = best_end - overlap if best_end < audio_length else audio_length
    
    return chunks

# ToDo: Yubee
def predict_from_model(audio_chunk: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Mock function to simulate emotion predictions from a model.
    Will be replaced with actual model predictions.
    
    Args:
        audio_chunk: Audio chunk data
        sr: Sample rate
        
    Returns:
        Dictionary with emotion predictions
    """
    # Generate mock emotion scores that sum to 100
    emotions = {
        "angry": random.randint(5, 60),
        "sad": random.randint(5, 60),
        "fear": random.randint(5, 60),
        "happy": random.randint(5, 60),
        "neutral": random.randint(5, 70)
    }
    
    # Normalize to sum to 100
    total = sum(emotions.values())
    for emotion in emotions:
        emotions[emotion] = round((emotions[emotion] / total) * 100)
    
    # Ensure they sum to 100 after rounding
    adjustment = 100 - sum(emotions.values())
    emotions["neutral"] += adjustment
    
    # Determine main emotion
    main_emotion = max(emotions, key=emotions.get)
    
    return {
        "mainEmotion": main_emotion,
        "emotions": emotions
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: Optional[UploadFile] = File(None), 
                 recorded_audio: Optional[str] = Form(None)):
    """
    Endpoint to predict emotions from uploaded or recorded audio.
    
    Args:
        file: Uploaded audio file
        recorded_audio: Base64 encoded audio data from frontend recording
        
    Returns:
        JSON with emotion predictions for each audio segment
    """
    if not file and not recorded_audio:
        raise HTTPException(status_code=400, detail="No audio provided")
    
    try:
        # Process uploaded file
        if file:
            audio_data, sr = librosa.load(io.BytesIO(await file.read()), sr=None)
        # Process recorded audio from frontend
        else:
            import base64
            audio_bytes = base64.b64decode(recorded_audio.split(',')[1] if ',' in recorded_audio else recorded_audio)
            
            # Save to temporary file and read with librosa
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
                
            audio_data, sr = librosa.load(temp_file_path, sr=None)
            os.unlink(temp_file_path)  # Clean up temp file
        
        # Get audio length in seconds
        audio_length = len(audio_data) / sr
        
        # Chunk the audio
        chunks = chunk_audio(audio_data, sr)
        
        # Process each chunk
        segments = []
        for chunk in chunks:
            # Get predictions for this chunk
            prediction = predict_from_model(chunk["audio"], sr)
            
            # Create segment with start/end times and predictions
            segment = {
                "start": chunk["start"],
                "end": chunk["end"],
                "mainEmotion": prediction["mainEmotion"],
                "emotions": prediction["emotions"]
            }
            segments.append(segment)
        
        # Determine overall emotion (most frequent or weighted average)
        emotion_counts = {}
        for segment in segments:
            emotion = segment["mainEmotion"]
            duration = segment["end"] - segment["start"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + duration
        
        overall_emotion = max(emotion_counts, key=emotion_counts.get)
        
        return {
            "segments": segments,
            "overallEmotion": overall_emotion,
            "audioLength": audio_length
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8767)
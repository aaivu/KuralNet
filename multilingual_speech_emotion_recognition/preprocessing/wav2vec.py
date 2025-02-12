import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load Wav2Vec2 processor and model
MODEL_NAME = "facebook/wav2vec2-base"  # Can be replaced with another variant


class Wav2Vec2FeatureExtractor:
    def __init__(self, model_name: str = MODEL_NAME):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extracts last hidden layer features from Wav2Vec2 encoder.

        Args:
            audio (np.ndarray): Input waveform (1D NumPy array).
            sr (int, optional): Sampling rate (default = 16000).

        Returns:
            np.ndarray: Features from the last encoder hidden layer (SeqLen, HiddenDim).
        """
        # Process audio input
        input_values = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_values

        # Forward pass through Wav2Vec2 encoder
        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        # Extract last hidden layer (Tensor -> NumPy)
        last_hidden_state = outputs.hidden_states[-1].squeeze(0).cpu().numpy()  # Shape: (SeqLen, HiddenDim)

        return last_hidden_state  # Expected: (SeqLen, 1024) for wav2vec2-large-960h

# Example Usage:
# extractor = Wav2Vec2FeatureExtractor()
# audio_path = "test_audio.wav"
# audio, sr = librosa.load(audio_path, sr=16000)
# features = extractor.extract_features(audio)

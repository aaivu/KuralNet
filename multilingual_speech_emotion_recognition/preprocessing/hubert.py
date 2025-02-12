import librosa
import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# Define the HuBERT model name
MODEL_NAME = "facebook/hubert-base-ls960"  # You can replace with other HuBERT models


class HuBERTFeatureExtractor:
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initializes the HuBERT model and feature extractor.
        """
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extracts last hidden layer features from HuBERT encoder.

        Args:
            audio (np.ndarray): Input waveform (1D NumPy array).
            sr (int, optional): Sampling rate (default = 16000).

        Returns:
            np.ndarray: Features from the last encoder hidden layer (SeqLen, HiddenDim).
        """
        # Process audio input
        input_values = self.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_values

        # Forward pass through HuBERT encoder
        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        # Extract last hidden layer (Tensor -> NumPy)
        last_hidden_state = outputs.hidden_states[-1].squeeze(0).cpu().numpy()  # Shape: (SeqLen, HiddenDim)

        return last_hidden_state  # Expected: (SeqLen, 768) for hubert-base-ls960


if __name__ == "__main__":
    # Initialize HuBERT feature extractor
    extractor = HuBERTFeatureExtractor()

    # Load audio file
    audio_path = "./tests/test_data/test_audio.wav"
    audio, sr = librosa.load(audio_path, sr=16000)

    # Extract features
    features = extractor.extract_features(audio)

    # Ensure features are extracted
    assert features is not None

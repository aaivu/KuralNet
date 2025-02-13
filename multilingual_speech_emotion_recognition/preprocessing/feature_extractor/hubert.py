import librosa
import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

MODEL_NAME = "facebook/hubert-base-ls960"


class HuBERTFeatureExtractor:
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initializes the HuBERT model and feature extractor.
        """
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name
        )
        self.model = HubertModel.from_pretrained(model_name)
        self.model.eval()

    def extract_features(
        self, audio: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """
        Extracts last hidden layer features from HuBERT encoder.

        Args:
            audio (np.ndarray): Input waveform (1D NumPy array).
            sr (int, optional): Sampling rate (default = 16000).

        Returns:
            np.ndarray: Features from the last encoder hidden layer (SeqLen, HiddenDim).
        """
        input_values = self.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_values

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        last_hidden_states = outputs.last_hidden_state

        return torch.mean(last_hidden_states, dim=1).numpy().squeeze()


if __name__ == "__main__":
    extractor = HuBERTFeatureExtractor()
    audio_path = "./tests/test_data/test_audio.wav"
    audio, sr = librosa.load(audio_path, sr=16000)
    features = extractor.extract_features(audio)

    assert features is not None

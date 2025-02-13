import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

MODEL_NAME = "microsoft/wavlm-base-plus"


class WavLMModelExtractor:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        """
        Initializes the WavLM model and feature extractor.
        """
        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name
        )
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_features(
        self, audio: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """
        Extracts features from the last hidden layer of the WavLM encoder.

        Args:
            audio (np.ndarray): Input waveform (1D NumPy array).
            sr (int, optional): Sampling rate (default is 16000).

        Returns:
            np.ndarray: Array of features with shape (SeqLen, HiddenDim).
        """
        inputs = self.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        last_hidden_states = outputs.last_hidden_state
        return torch.mean(last_hidden_states, dim=1).cpu().numpy().squeeze()


if __name__ == "__main__":
    extractor = WavLMModelExtractor()
    audio_path = "./tests/test_data/test_audio.wav"
    audio, sr = librosa.load(audio_path, sr=16000)
    features = extractor.extract_features(audio)
    assert features is not None

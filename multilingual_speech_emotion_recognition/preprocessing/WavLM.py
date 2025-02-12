import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

MODEL_NAME = "microsoft/wavlm-base-plus"


class WavLMModelExtractor:
    def __init__(self, model_name: str = MODEL_NAME):
        # Use AutoFeatureExtractor for audio-only processing
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name
        )
        self.model = WavLMModel.from_pretrained(model_name)
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
        # Process the audio input using the feature extractor.
        inputs = self.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        )
        input_values = inputs.input_values

        # Forward pass through the model (disabling gradient computation).
        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        # Extract the last hidden layer, remove batch dimension, and convert to NumPy.
        last_hidden_state = outputs.hidden_states[-1].squeeze(0).cpu().numpy()
        return last_hidden_state


# Example Usage:
# extractor = WavLMModelExtractor()
# audio_path = "./tests/test_data/test_audio.wav"
# audio, sr = librosa.load(audio_path, sr=16000)
# features = extractor.extract_features(audio)
# print(features)

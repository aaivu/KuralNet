import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

MODEL_NAME = "facebook/wav2vec2-base"


class Wav2Vec2FeatureExtractor:
    def __init__(self, model_name: str = MODEL_NAME):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()

    def extract_features(
        self, audio: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """
        Extracts last hidden layer features from Wav2Vec2 encoder.

        Args:
            audio (np.ndarray): Input waveform (1D NumPy array).
            sr (int, optional): Sampling rate (default = 16000).

        Returns:
            np.ndarray: Features from the last encoder hidden layer (SeqLen, HiddenDim).
        """

        input_values = self.processor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_values

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        last_hidden_states = outputs.last_hidden_state

        return torch.mean(last_hidden_states, dim=1).numpy().squeeze()


if __name__ == "__main__":
    extractor = Wav2Vec2FeatureExtractor()
    audio_path = "./tests/test_data/test_audio.wav"
    audio, sr = librosa.load(audio_path, sr=16000)
    features = extractor.extract_features(audio)

    print(features)

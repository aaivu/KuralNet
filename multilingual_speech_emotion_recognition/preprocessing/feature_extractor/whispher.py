import librosa
import numpy as np
import torch
from transformers import WhisperModel, WhisperProcessor

from multilingual_speech_emotion_recognition.utils.utils import class_renamer

MODEL_NAME = "openai/whisper-small"


@class_renamer("whisper", MODEL_NAME.split("/")[1].split("-")[1])
class WhisperFeatureExtractor:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        """
        Initializes the WhisperFeatureExtractor with the specified model.

        Args:
            model_name (str, optional): Name of the Whisper model to use (default = "openai/whisper-small").
            device (str, optional): Device to run the model on ("cuda" or "cpu").
            If None, automatically selects based on availability.
        """
        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_features(
        self, audio: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """
        Extracts last hidden layer features from Whisper encoder.

        Args:
            audio (np.ndarray): Input waveform (1D NumPy array).
            sr (int, optional): Sampling rate (default = 16000).

        Returns:
            np.ndarray: Features from the last encoder hidden layer (SeqLen, HiddenDim).
        """
        # Convert audio to log-Mel spectrogram features
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            outputs = self.model.encoder(
                input_features, output_hidden_states=True
            )

        last_hidden_states = outputs.last_hidden_state
        return torch.mean(last_hidden_states, dim=1).cpu().numpy().squeeze()


if __name__ == "__main__":
    extractor = WhisperFeatureExtractor().__class__
    audio_path = "./tests/test_data/test_audio.wav"
    audio, sr = librosa.load(audio_path, sr=16000)
    features = extractor.extract_features(audio)

    assert features is not None

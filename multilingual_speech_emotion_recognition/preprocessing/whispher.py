import librosa
import numpy as np
import torch
from transformers import WhisperModel, WhisperProcessor

MODEL_NAME = "openai/whisper-small"


class WhisperFeatureExtractor:
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initializes the WhisperFeatureExtractor with the specified model.

        Args:
            model_name (str, optional): Name of the Whisper model to use (default = "openai/whisper-small").
        """
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name)
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
        # convert audio to log-Mel spectrogram features
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_features, output_hidden_states=True
            )

        last_hidden_state = (
            encoder_outputs.hidden_states[-1].squeeze(0).cpu().numpy()
        )

        return last_hidden_state


if __name__ == "__main__":
    extractor = WhisperFeatureExtractor()
    audio_path = "./tests/test_data/test_audio.wav"
    audio, sr = librosa.load(audio_path, sr=16000)
    features = extractor.extract_features(audio)

    assert features is not None

import librosa
import numpy as np
import torch
from transformers import WhisperModel, WhisperProcessor


class WhisperFeatureExtractor:
    def __init__(
        self, model_name: str = "openai/whisper-small", device: str = None
    ):
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
        return last_hidden_states.cpu()


class TraditionalFeatureExtractor:
    def __init__(self, sr: int = 16000):
        self.sr = sr

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract traditional features: MFCCs, Chroma, Mel Spectrogram, RMS, and Zero-Crossing Rate (ZCR).
        """
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13).mean(
            axis=1
        )
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr).mean(axis=1)
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sr).mean(axis=1)
        rms = librosa.feature.rms(y=audio).mean(axis=1)
        zcr = librosa.feature.zero_crossing_rate(y=audio).mean(axis=1)

        return np.concatenate((mfccs, chroma, mel, rms, zcr))

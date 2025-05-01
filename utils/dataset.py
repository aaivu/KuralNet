import os

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.encoder import emotion_converter
from src.utils.utils import is_valid_wav


def pad_or_truncate(audio, desired_length=30000):
    if len(audio) > desired_length:
        return audio[:desired_length]
    elif len(audio) < desired_length:
        pad_width = desired_length - len(audio)
        return np.pad(audio, (0, pad_width), mode="constant")
    return audio


class SpeechEmotionDataset(Dataset):
    """
    A custom dataset class for speech emotion recognition.

    This class dynamically loads dataset metadata and extracts necessary attributes
    such as emotion labels and feature columns.

    Args:
        dataset_name (str): The name of the dataset.
        dataset_path (str): Path to the CSV metadata file.

    Attributes:
        dataset_name (str): Name of the dataset.
        emotions (torch.Tensor): Tensor containing emotion labels.
        features (pd.DataFrame): DataFrame containing feature columns.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        language: str,
        sr: int = 16000,
    ) -> None:
        """
        Initializes the dataset by loading metadata and extracting attributes.

        Args:
            dataset_name (str): Dataset name.
            dataset_path (str): Path to the CSV metadata file.
        """
        self.dataset_name = dataset_name
        self.language = language

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Metadata file not found: {dataset_path}")

        # Load dataset metadata
        self.dataset = pd.read_csv(dataset_path)

        # Ensure 'emotion' column exists
        if "emotion" not in self.dataset.columns:
            raise ValueError("Metadata file must contain an 'emotion' column.")

        # Extract features and emotion labels
        self.available_emotions = sorted(self.dataset["emotion"].unique())
        self.EMOTION_MAPPING = {
            emotion: idx for idx, emotion in enumerate(self.available_emotions)
        }

        paths = self.dataset["audio_path"].astype(str).tolist()
        emotions_raw = self.dataset["emotion"].tolist()

        self.audios = []
        self.emotions = []

        for path, emotion in zip(paths, emotions_raw):
            if not is_valid_wav(path):
                print(f"Skipping invalid file: {path}")
                continue
            audio, sr = librosa.load(path, sr=sr)
            audio_fixed = pad_or_truncate(audio, 50000)
            self.audios.append(audio_fixed)

            emotion_encoded = emotion_converter(
                emotion, mode="encode", EMOTION_MAPPING=self.EMOTION_MAPPING
            )[0]
            self.emotions.append(emotion_encoded)

        self.emotions = torch.tensor(self.emotions, dtype=torch.long)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.emotions)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves a sample from the dataset."""

        return {"audio": self.emotions[idx], "labels": self.audios[idx]}

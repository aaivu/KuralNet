import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from multilingual_speech_emotion_recognition.dataset.encoder import (
    emotion_encoder, gender_encoder, language_encoder)


class _SpeechEmotionDataset(Dataset):
    """
    A custom dataset class for speech emotion recognition.

    This class dynamically loads dataset metadata and extracts necessary attributes
    such as gender and emotion labels.

    Args:
        dataset (str): The name of the dataset (e.g., 'EmoDB', 'RAVDESS').
        language (str): The language of the dataset.
        dataset_path (str): The directory where the meta is stored.
        max_length (int, optional): Maximum length of the audio input in samples (default: 16000).

    Attributes:
        audio_paths (List[str]): List of audio file paths.
        emotions (List[int]): Emotion labels.
        gender (List[int]): Gender labels.
        metadata (pd.DataFrame): Loaded dataset metadata.
    """

    def __init__(
        self,
        dataset: str,
        language: str,
        dataset_path: str,
        max_length: int = 16000,
    ) -> None:
        """
        Initializes the dataset by loading metadata and extracting attributes.

        Args:
            dataset (str): Dataset name.
            language (str): Language of the dataset.
            dataset_path (str): The directory where the meta is stored.
            max_length (int): Maximum audio sample length.
        """
        self.dataset = dataset
        self.language = language_encoder(language=language)
        self.dataset_path = dataset_path
        self.max_length = max_length

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Metadata file not found: {dataset_path}")

        # load dataset
        self.metadata = pd.read_csv(dataset_path)

        # encode dataset
        self.metadata["emotion"] = self.metadata["emotion"].apply(
            emotion_encoder
        )
        self.metadata["gender"] = self.metadata["gender"].apply(gender_encoder)

        self.audio_paths = self.metadata["audio_path"].tolist()
        self.emotions = self.metadata["emotion"].tolist()
        self.gender = self.metadata["gender"].tolist()

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.emotions)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves a sample from the dataset."""
        features = {
            "emotion": torch.tensor(self.emotions[idx], dtype=torch.long),
            "gender": torch.tensor(self.gender[idx], dtype=torch.long),
            "language": torch.tensor(self.language, dtype=torch.long),
            "metadata": {
                "dataset": self.dataset,
                "language": self.language,
                "dataset_path": self.dataset_path,
            },
        }
        return features

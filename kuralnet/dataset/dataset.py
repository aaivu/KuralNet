import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from kuralnet.dataset.encoder import emotion_encoder


class _SpeechEmotionDataset(Dataset):
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
        self, dataset_name: str, dataset_path: str, language: str
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
        self.emotions = torch.tensor(
            self.dataset["emotion"].apply(emotion_encoder).values,
            dtype=torch.long,
        )
        self.features = torch.tensor(
            self.dataset.drop(columns=["emotion"]).values.astype(np.float32),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.emotions)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves a sample from the dataset."""

        sample = {
            "emotion": self.emotions[idx],
            "features": self.features[idx],
        }
        return sample

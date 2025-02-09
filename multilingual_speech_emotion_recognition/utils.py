import logging
from typing import List, Optional

import torch
from torch.utils.data import Dataset


class _SpeechEmotionDataset(Dataset):
    """
    A custom dataset class for speech emotion recognition.

    Args:
        audio_paths (Optional[List[str]]): A list of file paths to the audio files.
        emotions (List[int]): A list of emotion labels corresponding to the audio files.
        max_length (int): The maximum length of audio input, typically in number of samples.
        filepath (Optional[str]): Path to the directory where the audio files are stored.
        language (Optional[str]): The language of the audio data.
        dataset (Optional[str]): The name of the dataset (e.g., 'EmoDB', 'RAVDESS').
        gender (Optional[str]): The gender of the speaker in the audio files.

    Attributes:
        audio_paths (Optional[List[str]]): List of file paths for the audio samples.
        emotions (List[int]): Emotion labels for the audio samples.
        max_length (int): Maximum length for audio samples.
        filepath (Optional[str]): Path to the audio files.
        language (Optional[str]): Language in which the audio is spoken.
        dataset (Optional[str]): Dataset name.
        gender (Optional[List[str]]): Gender of the speaker.
    """

    def __init__(
        self,
        audio_paths: Optional[List[str]] = None,
        emotions: List[int] = None,
        gender: Optional[List[str]] = None,
        max_length: int = 16000,
        filepath: Optional[str] = None,
        language: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> None:
        """
        Initializes the _SpeechEmotionDataset class.

        Args:
            audio_paths (Optional[List[str]]): A list of paths to the audio files.
            emotions (List[int]): A list of emotion labels corresponding to the audio files.
            max_length (int): The maximum length of audio input, typically in number of samples.
            filepath (Optional[str]): Path to the directory where audio files are stored.
            language (Optional[str]): The language of the audio data.
            dataset (Optional[str]): The name of the dataset.
            gender (Optional[List[str]]): Gender of the speaker.
        """
        self.audio_paths = audio_paths
        self.emotions = emotions
        self.max_length = max_length
        self.filepath = filepath
        self.language = language
        self.dataset = dataset
        self.gender = gender

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of emotion labels in the dataset.
        """
        return len(self.emotions)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the emotion label and metadata.
        """
        features = {}

        metadata = {
            "filepath": self.filepath,
            "language": self.language,
            "dataset": self.dataset,
        }

        features["emotion"] = torch.tensor(
            self.emotions[idx], dtype=torch.long
        )
        features["gender"] = torch.tensor(self.gender[idx], dtype=torch.long)
        features["metadata"] = metadata
        return features


def _get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger.

    Args:
        name (str): The name of the logger (typically the module name).
        level (int): The logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

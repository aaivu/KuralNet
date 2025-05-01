import logging
import os
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, random_split


# Logging Configuration
def get_logger(
    name: str = __name__, level: int = logging.INFO
) -> logging.Logger:
    """
    Creates and configures a logger with specified name and level.

    Args:
        name (str): Logger name, typically __name__ of the module
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


# File Operations
def load_csv(path: str) -> pd.DataFrame:
    """
    Loads and returns data from a CSV file.

    Args:
        path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded data as DataFrame

    Raises:
        ValueError: If file doesn't exist
    """
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")
    return pd.read_csv(path)


def get_wav_files(root_dir: str, extension: str = ".wav") -> List[str]:
    """
    Recursively finds all WAV files in the specified directory.

    Args:
        root_dir (str): Root directory to search
        extension (str): File extension to look for (default: '.wav')

    Returns:
        List[str]: List of paths to WAV files
    """
    wav_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        wav_files.extend(
            os.path.join(dirpath, f)
            for f in filenames
            if f.endswith(extension)
        )
    return wav_files


def is_valid_wav(file_path: str) -> bool:
    """
    Checks if a WAV file is valid and can be opened.

    Args:
        file_path (str): Path to WAV file

    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        with sf.SoundFile(file_path) as _:
            return True
    except RuntimeError:
        return False


# Audio Processing
def load_audio(path: str, sr: float = 22050) -> Tuple[np.ndarray, float]:
    """
    Loads an audio file at the specified sampling rate.

    Args:
        path (str): Path to audio file
        sr (float): Target sampling rate in Hz

    Returns:
        Tuple[np.ndarray, float]: Audio time series and sampling rate

    Raises:
        ValueError: If file doesn't exist
    """
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")
    return librosa.load(path, sr=sr)


def pad_or_truncate(
    audio: np.ndarray, desired_length: int = 30000
) -> np.ndarray:
    """
    Adjusts audio length by padding or truncating.

    Args:
        audio (np.ndarray): Input audio array
        desired_length (int): Target length in samples

    Returns:
        np.ndarray: Modified audio array of desired length
    """
    if len(audio) > desired_length:
        return audio[:desired_length]
    elif len(audio) < desired_length:
        return np.pad(audio, (0, desired_length - len(audio)), mode="constant")
    return audio


# Data Processing
def stratified_sampling(data: List, max_files: int) -> List:
    """
    Performs stratified sampling on the dataset.

    Args:
        data (List): Input data with emotion labels and paths
        max_files (int): Maximum number of files to sample

    Returns:
        List: Sampled data maintaining class distribution
    """
    if len(data) <= max_files:
        return data

    df = pd.DataFrame(data, columns=["emotion", "path"])
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=max_files, random_state=42
    )

    for train_idx, _ in splitter.split(df, df["emotion"]):
        sampled_df = df.iloc[train_idx]

    return sampled_df.values.tolist()


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    val_split: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, DataLoader]:
    """
    Creates DataLoader objects for train, validation, and test sets.

    Args:
        dataset (Dataset): Input dataset
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the data
        val_split (bool): Whether to create validation set
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation

    Returns:
        Dict[str, DataLoader]: DataLoaders for each split
    """
    dataset_size = len(dataset)

    if val_split:
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        dataloaders = {
            "train": DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle
            ),
            "val": DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            ),
            "test": DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            ),
        }
    else:
        train_size = int(train_ratio * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size]
        )
        dataloaders = {
            "train": DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle
            ),
            "test": DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            ),
        }

    return dataloaders

import logging
import os

import librosa
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedShuffleSplit


def get_logger(
    name: str = __name__, level: int = logging.INFO
) -> logging.Logger:
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

    if not logger.handlers:
        logger.addHandler(ch)

    return logger


def load_csv(path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        ValueError: If the file is not found.
    """
    if not os.path.exists(path):
        raise ValueError(f"{path} not found")

    return pd.read_csv(path)


def load_audio(path: str, sr: float = 22050) -> tuple:
    """
    Loads an audio file using librosa.

    Args:
        path (str): The path to the audio file.
        sr (float, optional): The target sampling rate. Defaults to 22050.

    Returns:
        tuple: A tuple containing the audio time series and the sampling rate.

    Raises:
        ValueError: If the file is not found.
    """
    if not os.path.exists(path):
        raise ValueError(f"{path} not found")

    audio, sample_rate = librosa.load(path, sr=sr)
    return audio, sample_rate


def model_namer(name, version):
    def decorator(cls):
        cls.__name__ = f"{name}_{version}"
        return cls

    return decorator


def get_wav_files(root_dir, extention=".wav"):
    wav_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(extention):
                wav_files.append(os.path.join(dirpath, file))
    return wav_files


def is_valid_wav(file_path):
    try:
        with sf.SoundFile(file_path) as f:
            return True
    except RuntimeError:
        return False


def stratified_sampling(data, max_files):
    if len(data) > max_files:
        df = pd.DataFrame(data, columns=["emotion", "path"])
        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=max_files, random_state=42
        )

        for train_idx, _ in splitter.split(df, df["emotion"]):
            sampled_df = df.iloc[train_idx]

        data = sampled_df.values.tolist()

    return data

import logging
import os

import librosa
import pandas as pd
from pydub import AudioSegment


def _get_logger(
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


import os
import tempfile

import librosa
from pydub import AudioSegment


def load_audio(path: str, sr: float = 22050) -> tuple:
    """
    Loads an audio file using librosa. If the file is not in .wav format, converts it to .wav.

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

    if not path.lower().endswith(".wav"):
        audio = AudioSegment.from_file(path)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp_wav_file:
            wav_path = temp_wav_file.name
            audio.export(wav_path, format="wav")
        path = wav_path

    audio, sample_rate = librosa.load(path, sr=sr)

    if path != wav_path:
        os.remove(path)

    return audio, sample_rate

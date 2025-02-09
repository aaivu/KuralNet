import librosa
import numpy as np


def load_audio(file_path: str, sr: int = 16000):
    """
    Load an audio file and resample to the target sample rate.

    Args:
        file_path (str): Path to the audio file.
        sr (int): Target sample rate.

    Returns:
        np.ndarray: Loaded audio waveform.
    """
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def extract_mfcc(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 13):
    """
    Extract MFCC features from an audio signal.

    Args:
        audio (np.ndarray): Audio waveform.
        sr (int): Sample rate.
        n_mfcc (int): Number of MFCC coefficients.

    Returns:
        np.ndarray: MFCC features.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def preprocess_data(audio_paths: list, max_length: int = 16000):
    """
    Preprocess the dataset by extracting MFCC features for each audio file.

    Args:
        audio_paths (list): List of paths to audio files.
        max_length (int): Maximum length for audio input.

    Returns:
        list: Processed features (MFCC) for each audio file.
    """
    features = []
    for path in audio_paths:
        audio = load_audio(path)
        mfcc = extract_mfcc(audio)
        if mfcc.shape[1] > max_length:
            mfcc = mfcc[:, :max_length]
        features.append(mfcc)
    return features

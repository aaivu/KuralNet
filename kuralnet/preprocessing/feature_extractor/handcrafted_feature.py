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
    mfcc = np.mean(
        librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T, axis=0
    )
    return mfcc


def extract_melspectrogram(audio: np.ndarray, sr: int = 16000):
    """
    Extract Mel-spectrogram features from an audio signal.

    Args:
        audio (np.ndarray): Audio waveform.
        sr (int): Sample rate.

    Returns:
        np.ndarray: Mel-spectrogram features.
    """
    mel_spectrogram = np.mean(
        librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0
    )
    return mel_spectrogram


def extract_chroma_stft(audio: np.ndarray, sr: int = 16000):
    """
    Extract Chroma STFT features from an audio signal.

    Args:
        audio (np.ndarray): Audio waveform.
        sr (int): Sample rate.

    Returns:
        np.ndarray: Chroma STFT features.
    """
    stft = np.abs(librosa.stft(audio))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    return chroma_stft


def extract_rmse(audio: np.ndarray, _sr: int = 16000):
    """
    Extract Root Mean Square Energy (RMSE) from an audio signal.

    Args:
        audio (np.ndarray): Audio waveform.

    Returns:
        np.ndarray: RMSE features.
    """
    rmse = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    return rmse


def extract_zcr(audio: np.ndarray, _sr: int = 16000):
    """
    Extract Zero Crossing Rate (ZCR) from an audio signal.

    Args:
        audio (np.ndarray): Audio waveform.

    Returns:
        np.ndarray: ZCR features.
    """
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    return zcr


def preprocess_data(
    file_path: str,
    sr: int = 16000,
    required_features: list = [
        "mfcc",
        "mel_spectrogram",
        "chroma_stft",
        "rmse",
        "zcr",
    ],
):
    """
    Preprocess an audio file to extract features.

    Args:
        file_path (str): Path to the audio file.
        required_features (list): List of required features.

    Returns:
        np.ndarray: Extracted features.
    """
    result = np.array([])

    audio = load_audio(file_path)

    if "mfcc" in required_features:
        mfcc = extract_mfcc(audio, sr)
        result = np.hstack([result, mfcc])

    if "mel_spectrogram" in required_features:
        mel_spectrogram = extract_melspectrogram(audio, sr)
        result = np.hstack([result, mel_spectrogram])

    if "chroma_stft" in required_features:
        chroma_stft = extract_chroma_stft(audio, sr)
        result = np.hstack([result, chroma_stft])

    if "rmse" in required_features:
        rmse = extract_rmse(audio)
        result = np.hstack([result, rmse])

    if "zcr" in required_features:
        zcr = extract_zcr(audio)
        result = np.hstack([result, zcr])

    return result

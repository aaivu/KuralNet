import librosa
import numpy as np

REQUIRED_FEATURES = [
    "mfcc",
    "mel_spectrogram",
    "chroma_stft",
    "rmse",
    "zcr",
]
SAMPLING_RATE = 16000


class TraditionalFeatureExtractor:

    def __init__(self):
        """
        Initialize the TraditionalFeatureExtractor.
        """
        pass

    def extract_mfcc(
        audio: np.ndarray, sr: int = SAMPLING_RATE, n_mfcc: int = 13
    ):
        mfcc = np.mean(
            librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T, axis=0
        )
        return mfcc

    def extract_melspectrogram(audio: np.ndarray, sr: int = SAMPLING_RATE):
        mel_spectrogram = np.mean(
            librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0
        )
        return mel_spectrogram

    def extract_chroma_stft(audio: np.ndarray, sr: int = SAMPLING_RATE):
        stft = np.abs(librosa.stft(audio))
        chroma_stft = np.mean(
            librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0
        )
        return chroma_stft

    def extract_rmse(audio: np.ndarray, _sr: int = SAMPLING_RATE):
        rmse = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        return rmse

    @staticmethod
    def extract_zcr(audio: np.ndarray, _sr: int = SAMPLING_RATE):
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        return zcr

    def extract_features(
        audio: np.ndarray,
        sr: int = SAMPLING_RATE,
        required_features: list = REQUIRED_FEATURES,
    ):
        result = np.array([])

        if "mfcc" in required_features:
            mfcc = TraditionalFeatureExtractor.extract_mfcc(audio, sr)
            result = np.hstack([result, mfcc])

        if "mel_spectrogram" in required_features:
            mel_spectrogram = (
                TraditionalFeatureExtractor.extract_melspectrogram(audio, sr)
            )
            result = np.hstack([result, mel_spectrogram])

        if "chroma_stft" in required_features:
            chroma_stft = TraditionalFeatureExtractor.extract_chroma_stft(
                audio, sr
            )
            result = np.hstack([result, chroma_stft])

        if "rmse" in required_features:
            rmse = TraditionalFeatureExtractor.extract_rmse(audio)
            result = np.hstack([result, rmse])

        if "zcr" in required_features:
            zcr = TraditionalFeatureExtractor.extract_zcr(audio)
            result = np.hstack([result, zcr])

        return result

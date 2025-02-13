import numpy as np

from multilingual_speech_emotion_recognition.preprocessing.feature_extractor.handcrafted_feature import (
    extract_chroma_stft, extract_melspectrogram, extract_mfcc, extract_rmse,
    extract_zcr, load_audio, preprocess_data)
from tests.conftest import TEST_AUDIO


# extract_whisper_features


def test_load_audio():
    # Test loading an audio file
    sr = 16000
    audio = load_audio(TEST_AUDIO, sr=sr)

    # Check if the audio is loaded correctly
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert audio.dtype == np.float32


def test_extract_mfcc():
    # Test MFCC extraction
    sr = 16000
    audio = load_audio(TEST_AUDIO, sr=sr)
    mfcc = extract_mfcc(audio, sr=sr, n_mfcc=13)

    # Check if MFCC features are extracted correctly
    assert isinstance(mfcc, np.ndarray)
    assert mfcc.shape == (13,)


def test_extract_melspectrogram():
    # Test Mel-spectrogram extraction
    sr = 16000
    audio = load_audio(TEST_AUDIO, sr=sr)
    mel_spectrogram = extract_melspectrogram(audio, sr=sr)

    # Check if Mel-spectrogram features are extracted correctly
    assert isinstance(mel_spectrogram, np.ndarray)
    assert mel_spectrogram.shape[0] > 0


def test_extract_chroma_stft():
    # Test Chroma STFT extraction
    sr = 16000
    audio = load_audio(TEST_AUDIO, sr=sr)
    chroma_stft = extract_chroma_stft(audio, sr=sr)

    # Check if Chroma STFT features are extracted correctly
    assert isinstance(chroma_stft, np.ndarray)
    assert chroma_stft.shape[0] == 12  # Chroma features have 12 bins


def test_extract_rmse():
    # Test RMSE extraction
    audio = load_audio(TEST_AUDIO)
    rmse = extract_rmse(audio)

    # Check if RMSE features are extracted correctly
    assert isinstance(rmse, np.ndarray)
    assert rmse.shape[0] == 1


def test_extract_zcr():
    # Test ZCR extraction
    audio = load_audio(TEST_AUDIO)
    zcr = extract_zcr(audio)

    # Check if ZCR features are extracted correctly
    assert isinstance(zcr, np.ndarray)
    assert zcr.shape[0] == 1


# def test_extract_whisper_features():
#     # Test Whisper feature extraction
#     sr = 16000
#     audio = load_audio(TEST_AUDIO_FILE, sr=sr)
#     whisper_features = extract_whisper_features(audio, sr=sr)
#
#     # Check if Whisper features are extracted correctly
#     assert isinstance(whisper_features, np.ndarray)
#     assert whisper_features.shape[0] > 0  # Check that the features are not empty


def test_preprocess_data():
    # Test preprocessing with all features
    required_features = ["mfcc", "mel_spectrogram", "chroma_stft", "rmse", "zcr"]
    features = preprocess_data(TEST_AUDIO, required_features=required_features)

    # Check if the features are concatenated correctly
    assert isinstance(features, np.ndarray)
    assert features.shape[0] > 0

    # Test preprocessing with a subset of features
    required_features = ["mfcc", "zcr"]
    features = preprocess_data(TEST_AUDIO, required_features=required_features)

    # Check if only the specified features are extracted
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 13 + 1  # 13 MFCC + 1 ZCR

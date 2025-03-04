import torch
import numpy as np

def test_whisper_feature_extractor_init(whisper_extractor):
    """Test if the WhisperFeatureExtractor initializes properly."""
    assert whisper_extractor.device == "cpu"
    assert whisper_extractor.model is not None
    assert whisper_extractor.processor is not None


def test_extract_features(whisper_extractor,sample_dataset):
    """Test if the WhisperFeatureExtractor extracts features correctly."""
    features = whisper_extractor.extract_features(sample_dataset[0]['audio'])
    assert isinstance(features, torch.Tensor)

    assert features.shape[0] == 1
    assert features.shape[1] == 1500
    assert features.shape[2] == 768


def test_traditional_feature_extractor_init(traditional_extractor):
    """Test if the TraditionalFeatureExtractor initializes properly."""
    assert traditional_extractor.sr == 16000


def test_extract_traditional_features(traditional_extractor,sample_dataset):
    """Test if the TraditionalFeatureExtractor extracts features correctly."""
    features = traditional_extractor.extract_features(sample_dataset[0]['audio'])
    assert isinstance(features, np.ndarray)

    # 13 MFCCs + 12 Chroma + 128 Mel + 1 RMS + 1 ZCR
    assert features.shape[0] == 155
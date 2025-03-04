import pytest

from kuralnet.dataset.dataset import SpeechEmotionDataset
from kuralnet.feature.feature_extractor import WhisperFeatureExtractor, TraditionalFeatureExtractor

TEST_CSV = "tests/test_data/en_iemocap.csv"
TEST_AUDIO = "tests/test_data/test_audio.wav"


@pytest.fixture
def sample_dataset():
    """Fixture to create a test dataset instance."""
    return SpeechEmotionDataset(
        dataset_name="IEMOCAP",
        language="English",
        dataset_path=TEST_CSV,
    )

@pytest.fixture
def whisper_extractor():
    """Fixture to create an instance of WhisperFeatureExtractor."""
    return WhisperFeatureExtractor(model_name="openai/whisper-small", device="cpu")

@pytest.fixture
def traditional_extractor():
    """Fixture to create an instance of TraditionalFeatureExtractor."""
    return TraditionalFeatureExtractor(sr=16000)
import pytest

from multilingual_speech_emotion_recognition.dataset.dataset import \
    _SpeechEmotionDataset

TEST_CSV = "tests/test_data/en_iemocap.csv"
TEST_AUDIO = "tests/test_data/test_audio.wav"


@pytest.fixture
def sample_dataset():
    """Fixture to create a test dataset instance."""
    return _SpeechEmotionDataset(
        dataset="IEMOCAP",
        language="English",
        dataset_path=TEST_CSV,
    )

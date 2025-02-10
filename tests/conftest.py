import pytest
import pandas as pd
from multilingual_speech_emotion_recognition.dataset.dataset import _SpeechEmotionDataset

TEST_CSV = "tests/en_iemocap.csv"


@pytest.fixture
def sample_dataset():
    """Fixture to create a test dataset instance."""
    return _SpeechEmotionDataset(
        dataset="IEMOCAP",
        language="English",
        dataset_path=TEST_CSV,
    )
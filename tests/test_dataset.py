import torch

from multilingual_speech_emotion_recognition.dataset.dataset import \
    _SpeechEmotionDataset


def test_dataset_initialization(sample_dataset: _SpeechEmotionDataset):

    assert len(sample_dataset) > 0, "Dataset should not be empty"
    assert isinstance(sample_dataset.audio_paths, list), "Audio paths should be a list"
    assert isinstance(sample_dataset.emotions, list), "Emotions should be a list"
    assert isinstance(sample_dataset.gender, list), "Gender should be a list"
    assert len(sample_dataset.audio_paths) == len(
        sample_dataset.emotions
    ), "Mismatch in audio paths and emotions"


def test_getitem(sample_dataset: _SpeechEmotionDataset):

    sample = sample_dataset[0]
    assert isinstance(sample, dict), "Sample should be a dictionary"
    assert "emotion" in sample, "Sample should contain 'emotion'"
    assert "gender" in sample, "Sample should contain 'gender'"
    assert "language" in sample, "Sample should contain 'language'"
    assert "metadata" in sample, "Sample should contain 'metadata'"
    assert isinstance(sample["emotion"], torch.Tensor), "Emotion should be a tensor"
    assert isinstance(sample["gender"], torch.Tensor), "Gender should be a tensor"
    assert isinstance(sample["language"], torch.Tensor), "Language should be a tensor"


def test_sample_tensors(sample_dataset: _SpeechEmotionDataset):

    sample = sample_dataset[0]
    assert (
        sample["emotion"].dtype == torch.long
    ), "Emotion tensor dtype should be torch.long"
    assert (
        sample["gender"].dtype == torch.long
    ), "Gender tensor dtype should be torch.long"
    assert (
        sample["language"].dtype == torch.long
    ), "Language tensor dtype should be torch.long"

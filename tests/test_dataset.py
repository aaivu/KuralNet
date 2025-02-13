import torch

from multilingual_speech_emotion_recognition.dataset.dataset import \
    _SpeechEmotionDataset


def test_dataset_initialization(sample_dataset: _SpeechEmotionDataset):

    assert len(sample_dataset) > 0, "Dataset should not be empty"
    assert isinstance(sample_dataset.emotions.tolist(), list), "Emotions should be a list"
   


def test_getitem(sample_dataset: _SpeechEmotionDataset):

    sample = sample_dataset[0]
    assert isinstance(sample, dict), "Sample should be a dictionary"
    assert "emotion" in sample, "Sample should contain 'emotion'"
    assert isinstance(sample["emotion"], torch.Tensor), "Emotion should be a tensor"

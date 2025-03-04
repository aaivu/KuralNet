import torch
import numpy as np

def test_dataset_length(sample_dataset):
    """Test if dataset length matches the number of samples."""
    assert len(sample_dataset) > 0


def test_dataset_item_structure(sample_dataset):
    """Test if each dataset item has the correct structure."""
    item = sample_dataset[0]
    assert isinstance(item, dict)
    assert "emotion" in item and "audio" in item
    assert isinstance(item["emotion"], torch.Tensor)
    assert isinstance(item["audio"], np.ndarray)


def test_audio_length(sample_dataset):
    """Test if the loaded audio has the expected length (50000 samples)."""
    item = sample_dataset[0]
    assert len(item["audio"]) == 50000


def test_emotion_tensor_type(sample_dataset):
    """Test if the emotion labels are stored as torch tensors of type long."""
    assert sample_dataset.emotions.dtype == torch.long
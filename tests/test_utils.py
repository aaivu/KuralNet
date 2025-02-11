from multilingual_speech_emotion_recognition.dataset.dataset import _SpeechEmotionDataset
from multilingual_speech_emotion_recognition.utils.dataset_loader import get_dataloader
import pytest

@pytest.mark.parametrize("batch_size, shuffle, val_split", [
    (3, True, False),
    (2, False, True),
    (1, True, True),
    (4, False, False)
])
def test_get_dataloader(batch_size:int,shuffle:bool,val_split:float,sample_dataset:_SpeechEmotionDataset):
    dataloaders = get_dataloader(sample_dataset, batch_size=batch_size, shuffle=shuffle, val_split=val_split)

    assert "train" in dataloaders
    assert "test" in dataloaders
    if val_split:
        assert "val" in dataloaders

    train_size = len(dataloaders["train"].dataset)
    test_size = len(dataloaders["test"].dataset)

    if val_split:
        val_size = len(dataloaders["val"].dataset)
        assert train_size + test_size + val_size == sample_dataset.__len__()
    else:
        assert train_size + test_size == sample_dataset.__len__()

    assert dataloaders["train"].batch_size == batch_size
    assert dataloaders["test"].batch_size == batch_size
    if val_split:
        assert dataloaders["val"].batch_size == batch_size

def test_dataloader_shuffle(sample_dataset:_SpeechEmotionDataset):
    dataloaders = get_dataloader(sample_dataset, batch_size=5, shuffle=True)
    assert dataloaders["train"].dataset.dataset == sample_dataset
    assert dataloaders["test"].dataset.dataset == sample_dataset

def test_dataloader_batch_sizes(sample_dataset:_SpeechEmotionDataset):
    dataloaders = get_dataloader(sample_dataset, batch_size=3, shuffle=True, val_split=True)

    assert dataloaders["train"].batch_size == 3
    assert dataloaders["test"].batch_size == 3
    assert dataloaders["val"].batch_size == 3
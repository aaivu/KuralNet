import logging

import librosa
import pandas as pd
import pytest

from kuralnet.dataset.dataset import \
    _SpeechEmotionDataset
from kuralnet.utils.dataset_loader import \
    get_dataloader
from kuralnet.utils.utils import (_get_logger,
                                                                 load_audio,
                                                                 load_csv)
from tests.conftest import TEST_AUDIO, TEST_CSV


@pytest.mark.parametrize(
    "batch_size, shuffle, val_split",
    [(3, True, False), (2, False, True), (1, True, True), (4, False, False)],
)
def test_get_dataloader(
    batch_size: int,
    shuffle: bool,
    val_split: float,
    sample_dataset: _SpeechEmotionDataset,
):
    dataloaders = get_dataloader(
        sample_dataset, batch_size=batch_size, shuffle=shuffle, val_split=val_split
    )

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


def test_dataloader_shuffle(sample_dataset: _SpeechEmotionDataset):
    dataloaders = get_dataloader(sample_dataset, batch_size=5, shuffle=True)
    assert dataloaders["train"].dataset.dataset == sample_dataset
    assert dataloaders["test"].dataset.dataset == sample_dataset


def test_dataloader_batch_sizes(sample_dataset: _SpeechEmotionDataset):
    dataloaders = get_dataloader(
        sample_dataset, batch_size=3, shuffle=True, val_split=True
    )

    assert dataloaders["train"].batch_size == 3
    assert dataloaders["test"].batch_size == 3
    assert dataloaders["val"].batch_size == 3


def test_get_logger():
    logger = _get_logger(name="test_logger", level=logging.DEBUG)
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1


def test_load_csv_valid():
    result = load_csv(TEST_CSV)
    assert isinstance(result, pd.DataFrame)


def test_load_csv_invalid():
    try:
        load_csv("non_existent_file.csv")
    except ValueError as e:
        assert str(e) == "non_existent_file.csv not found"


def test_load_audio_valid():
    def mock_librosa_load(path, sr=22050):
        return [0.0, 0.0], sr

    librosa.load = mock_librosa_load

    audio, sample_rate = load_audio(TEST_AUDIO)
    assert isinstance(audio, list)
    assert len(audio) == 2
    assert sample_rate == 22050


def test_load_audio_invalid():
    try:
        load_audio("non_existent_audio.wav")
    except ValueError as e:
        assert str(e) == "non_existent_audio.wav not found"

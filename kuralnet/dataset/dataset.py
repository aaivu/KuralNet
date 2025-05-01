import torch
from torch.utils.data import Dataset
from kuralnet.dataset.emotion_mapping import emotion_encoder


class EmotionDataset(Dataset):
    def __init__(self, whisper_features, traditional_features, labels):
        self.emotion_to_idx = emotion_encoder

        self.whisper_features = torch.tensor(
            whisper_features, dtype=torch.float32
        )
        self.traditional_features = torch.tensor(
            traditional_features, dtype=torch.float32
        )
        self.labels = torch.tensor(
            [self.emotion_to_idx(label) for label in labels], dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.whisper_features[idx],
            self.traditional_features[idx],
            self.labels[idx],
        )

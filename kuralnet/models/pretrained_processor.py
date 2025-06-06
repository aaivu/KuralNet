import numpy as np
import torch
import torch.nn as nn


# For Layers after extracting features from Whispher
class WhisperLayersProcessor(nn.Module):
    """
    CNN+MLP block for Whisper embeddings.
    Input: (batch_size, 768) tensor of Whisper features.
    """

    def __init__(self, output_dim=128):
        super(WhisperLayersProcessor, self).__init__()
        # Convolutional layers (act on 768-dim as a 1D sequence)
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=32, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global pool -> length 1
        )
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):  # TODO: Need to Update for handling batch inputs
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.unsqueeze(1)
        x = self.conv(x)  # -> (batch_size, 64, 1)
        x = self.mlp(x)  # -> (batch_size, output_dim)
        return x

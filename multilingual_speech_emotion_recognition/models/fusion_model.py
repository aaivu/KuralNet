import torch
import torch.nn.functional as F
from torch import nn


class FusionModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_emotions=7):
        super(FusionModel, self).__init__()
        self.fusion = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_emotions)
        self.dropout = nn.Dropout(0.3)

    def forward(self, handcraft_features=None, whisper_features=None):
        if handcraft_features is not None and whisper_features is not None:
            combined = torch.cat((handcraft_features, whisper_features), dim=1)
        elif handcraft_features is not None:
            combined = torch.cat(
                (handcraft_features, torch.zeros_like(handcraft_features)),
                dim=1,
            )
        elif whisper_features is not None:
            combined = torch.cat(
                (torch.zeros_like(whisper_features), whisper_features), dim=1
            )
        else:
            raise ValueError("At least one type of features must be provided")

        x = F.relu(self.fusion(combined))
        x = self.dropout(x)
        output = self.classifier(x)
        return output

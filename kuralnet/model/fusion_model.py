import torch
import torch.nn.functional as F
from torch import nn


class FusionModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_emotions=7):
        super(FusionModel, self).__init__()
        self.fusion = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_emotions)
        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        handcraft_predictions=None,
        whisper_predictions=None,
        weights=None,
    ):
        if (
            handcraft_predictions is not None
            and whisper_predictions is not None
            and weights is not None
        ):
            combined = torch.cat(
                (
                    weights[0] * handcraft_predictions,
                    weights[1] * whisper_predictions,
                ),
                dim=1,
            )
        elif handcraft_predictions is not None:
            combined = torch.cat(
                (
                    handcraft_predictions,
                    torch.zeros_like(handcraft_predictions),
                ),
                dim=1,
            )
        elif whisper_predictions is not None:
            combined = torch.cat(
                (torch.zeros_like(whisper_predictions), whisper_predictions),
                dim=1,
            )
        else:
            raise ValueError("At least one type of features must be provided")

        x = F.relu(self.fusion(combined))
        x = self.dropout(x)
        output = self.classifier(x)
        return output

    def find_optimal_weights(
        self, handcraft_preds, whisper_preds, true_labels, grid_size=10
    ):
        best_weights = None
        best_accuracy = 0.0
        device = handcraft_preds.device

        for w1 in torch.linspace(0, 1, grid_size):
            w2 = 1 - w1
            weights = torch.tensor([w1, w2], device=device)

            combined_output = self.forward(
                handcraft_preds, whisper_preds, weights
            )
            predictions = torch.argmax(combined_output, dim=1)
            accuracy = (predictions == true_labels).float().mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights.clone()

        return best_weights

import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Classification head with softmax for 5 emotion classes.
    """

    def __init__(self, in_dim, num_classes=5):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        logits = self.fc(x)  # (batch, num_classes)
        probs = F.softmax(logits, dim=1)  # probability distribution
        return probs

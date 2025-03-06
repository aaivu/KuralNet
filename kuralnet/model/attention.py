import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x) -> torch.Tensor:
        """
        x: [batch_size, time_steps, embedding_dim] -> [1, 1500, 768]
        Returns: [batch_size, embedding_dim] -> [1, 768]
        """
        attention_scores = self.attention_weights(x)
        attention_scores = F.softmax(attention_scores, dim=1)

        attended_representation = torch.sum(x * attention_scores, dim=1)
        return attended_representation

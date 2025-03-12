import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeStepAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(TimeStepAttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)  
    
    def forward(self, x):
        """
        x: [batch_size, time_steps, embedding_dim] -> [1, 1500, 768]
        Returns: [batch_size, embedding_dim] -> [1, 768]
        """
        attention_scores = self.attention_weights(x)
        attention_scores = F.softmax(attention_scores, dim=1)
        
        attended_representation = torch.sum(x * attention_scores, dim=1)
        return attended_representation

class FeatureWiseAttentionPooling(nn.Module):
    def __init__(self, time_steps):
        super(FeatureWiseAttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(time_steps, time_steps)  # Learnable weights per feature over time

    def forward(self, x):
        """
        x: [batch_size, time_steps, embedding_dim] -> [1, 1500, 768]
        Returns: [batch_size, embedding_dim] -> [1, 768]
        """
        x_transposed = x.permute(0, 2, 1) 
        
        attention_scores = self.attention_weights(x_transposed)
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        attended_representation = torch.sum(x_transposed * attention_scores, dim=-1)
        return attended_representation

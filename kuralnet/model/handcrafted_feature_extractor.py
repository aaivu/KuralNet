import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Calculate attention scores
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        attention_weights = self.softmax(attention_scores)  # Apply softmax to get attention weights

        # Apply attention to the values (V)
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, hidden_dim)
        return attention_output


class HandcraftedAcousticEncoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HandcraftedAcousticEncoder, self).__init__()
        
         # Max Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # Define convolution layers
        self.conv_layer_1 = self._conv_block(input_dim, 64)
        self.conv_layer_2 = self._conv_block(64, 64)
        self.conv_layer_3 = self._conv_block(64, 128)
        self.conv_layer_4 = self._conv_block(128, 128)
        self.conv_layer_5 = self._conv_block(128, 256)
        self.conv_layer_6 = self._conv_block(256, 256)
        self.conv_layer_7 = self._conv_block(256, 256)
        self.conv_layer_8 = self._conv_block(256, 256)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)

        # Attention layer
        self.attention = SelfAttention(input_dim=256, hidden_dim=128)

    def _conv_block(self, in_channels, out_channels):
        """Helper function to create a convolutional block with activation and batch normalization."""
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            self.pool,
            self.dropout
        )

    def forward(self, x):
        # Apply convolution blocks
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_5(x)
        x = self.conv_layer_6(x)
        x = self.conv_layer_7(x)
        x = self.conv_layer_8(x)

        # Apply attention mechanism
        x = self.attention(x)

        # Apply global average pooling
        x = self.global_avg_pool(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activations and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

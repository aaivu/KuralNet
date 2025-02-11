import torch.nn as nn


class SpeechEmotionModel(nn.Module):
    """
    Speech Emotion Recognition Model.

    Args:
        input_dim (int): Input dimension for the model (e.g., number of MFCC coefficients).
        hidden_dim (int): Number of hidden units in the LSTM layer.
        output_dim (int): Number of emotion classes.
    """

    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, output_dim: int = 7):
        super(SpeechEmotionModel, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        output = self.softmax(output)
        return output

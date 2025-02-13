from torch import nn
import torch.nn.functional as F


class HandcraftedAcousticEncoder(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(HandcraftedAcousticEncoder, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_shape[0],
            out_channels=64,
            kernel_size=8,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=8, padding="same"
        )
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=8, padding="same"
        )
        self.conv4 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=8, padding="same"
        )
        self.conv5 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=8, padding="same"
        )
        self.conv6 = nn.Conv1d(
            in_channels=256, out_channels=256, kernel_size=8, padding="same"
        )
        self.conv7 = nn.Conv1d(
            in_channels=256, out_channels=256, kernel_size=8, padding="same"
        )
        self.conv8 = nn.Conv1d(
            in_channels=256, out_channels=256, kernel_size=8, padding="same"
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm1d
        self.dropout = nn.Dropout(0.3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.batch_norm(64)(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.batch_norm(128)(x)
        x = self.dropout(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.batch_norm(256)(x)
        x = self.dropout(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)
        x = self.batch_norm(256)(x)
        x = self.dropout(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

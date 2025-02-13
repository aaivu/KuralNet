from torch import nn
import torch


class PretrainedSpeechEncoder(nn.Module):
    pass

    import torch.nn.functional as F

    class PretrainedSpeechEncoder(nn.Module):
        def __init__(self, input_shape, num_classes):
            super(PretrainedSpeechEncoder, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=64, kernel_size=3, padding='same')
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
            self.bn1 = nn.BatchNorm1d(64)
            self.dropout1 = nn.Dropout(0.5)
            self.pool1 = nn.MaxPool1d(kernel_size=2)

            self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
            self.bn2 = nn.BatchNorm1d(128)
            self.dropout2 = nn.Dropout(0.5)
            self.pool2 = nn.MaxPool1d(kernel_size=2)

            self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
            self.bn3 = nn.BatchNorm1d(128)
            self.dropout3 = nn.Dropout(0.4)
            self.pool3 = nn.MaxPool1d(kernel_size=2)

            self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
            self.bn4 = nn.BatchNorm1d(256)
            self.dropout4 = nn.Dropout(0.4)
            self.pool4 = nn.MaxPool1d(kernel_size=2)

            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(256 * (input_shape[1] // 16), 128)
            self.dropout5 = nn.Dropout(0.4)
            self.fc2 = nn.Linear(128, 64)
            self.dropout6 = nn.Dropout(0.4)
            self.fc3 = nn.Linear(64, 32)
            self.dropout7 = nn.Dropout(0.4)
            self.fc4 = nn.Linear(32, num_classes)

        def forward(self, x):
            x = F.elu(self.bn1(self.conv2(self.conv1(x))))
            x = self.dropout1(self.pool1(x))

            x = F.elu(self.bn2(self.conv3(x)))
            x = self.dropout2(self.pool2(x))

            x = F.elu(self.bn3(self.conv4(x)))
            x = self.dropout3(self.pool3(x))

            x = F.elu(self.bn4(self.conv5(x)))
            x = self.dropout4(self.pool4(x))

            x = self.flatten(x)
            x = F.elu(self.fc1(x))
            x = self.dropout5(x)
            x = F.elu(self.fc2(x))
            x = self.dropout6(x)
            x = F.elu(self.fc3(x))
            x = self.dropout7(x)
            x = F.softmax(self.fc4(x), dim=1)

            return x
import os

import torch
from py.dataset import _SpeechEmotionDataset
from py.model import SpeechEmotionModel
from py.utils import _get_logger
from torch.utils.data import DataLoader


class Predictor:
    def __init__(
        self,
        model_path: str,
        test_data: _SpeechEmotionDataset,
        batch_size: int = 32,
        max_length: int = 16000,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.model_path = model_path
        self.test_data = test_data
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.logger = _get_logger(__name__)

        # Load the trained model
        self.model = SpeechEmotionModel().to(device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.logger.info(f"Model loaded from {model_path}")
        else:
            self.logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")

    def predict(self):
        """
        Make predictions on the test dataset.
        """
        test_loader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False
        )
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                labels = batch["emotion"].to(self.device)

                outputs = self.model(features)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Optionally, print classification report
        from sklearn.metrics import classification_report

        print(
            "Classification Report:\n",
            classification_report(all_labels, all_predictions),
        )

        return all_predictions

    def evaluate(self):
        """
        Evaluate the model's performance on the test dataset.
        """
        test_loader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False
        )
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                labels = batch["emotion"].to(self.device)

                outputs = self.model(features)

                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        accuracy = 100 * correct_preds / total_preds
        self.logger.info(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

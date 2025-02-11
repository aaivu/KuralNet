import torch
from py.dataset import _SpeechEmotionDataset
from py.model import SpeechEmotionModel
from py.utils import _get_logger
from sklearn.metrics import classification_report
from torch.optim import Adam
from torch.utils.data import DataLoader


class _ModelTrainer:
    def __init__(
        self,
        model: SpeechEmotionModel,
        train_data: _SpeechEmotionDataset,
        val_data: _SpeechEmotionDataset,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        max_length: int = 16000,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = device
        self.logger = _get_logger(__name__)

    def train(self, epochs: int = 10):
        """
        Train the model.
        """
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )

        for epoch in range(epochs):
            epoch_loss = 0
            correct_preds = 0
            total_preds = 0

            for batch in train_loader:
                features, labels = batch["features"].to(self.device), batch[
                    "emotion"
                ].to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(features)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

            train_accuracy = 100 * correct_preds / total_preds
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {train_accuracy:.2f}%"
            )

            # Validation after every epoch
            val_accuracy = self.evaluate(val_loader)
            self.logger.info(f"Validation Accuracy: {val_accuracy:.2f}%")

    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate the model's performance on the validation dataset.
        """
        self.model.eval()
        correct_preds = 0
        total_preds = 0

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in val_loader:
                features, labels = batch["features"].to(self.device), batch[
                    "emotion"
                ].to(self.device)

                outputs = self.model(features)

                _, predicted = torch.max(outputs, 1)

                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = 100 * correct_preds / total_preds
        # Optionally, print classification report
        print(classification_report(all_labels, all_predictions))

        return accuracy

    def save_model(self, model_path: str):
        """
        Save the trained model to a file.
        """
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")

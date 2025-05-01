import json
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.model.base_models import FeatureExtractorFactory
from src.model.model import SERBenchmarkModel
from src.utils.constant import BASE_MODEL, DATASET
from src.utils.data_loader import get_dataloader
from src.utils.dataset import SpeechEmotionDataset
from src.utils.encoder import emotion_converter
from src.utils.utils import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


# Apply Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
pd.options.mode.chained_assignment = None
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sklearn.utils.check_random_state(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# Hyperparameters
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
EPOCHS = int(os.getenv("EPOCHS", 30))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 5))

os.makedirs("./finetuned_models", exist_ok=True)
os.makedirs("./ui/public/train_val_test_logs", exist_ok=True)


def plot_loss(train_losses, val_losses, path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    parts = path.split("_", 3)
    if len(parts) >= 3:
        plt.title(
            f"Training and Validation Loss of {parts[0]} - {parts[1]} ({parts[2]})"
        )
    else:
        plt.title("Training and Validation Loss")
    plt.savefig(
        f"./ui/public/train_val_test_logs/{path}/{path}_loss_curve.png"
    )
    plt.close()


def plot_confusion_matrix(
    y_true, y_pred, phase, path, classes, EMOTION_MAPPING
):

    y_true = [
        emotion_converter(y, mode="decode", EMOTION_MAPPING=EMOTION_MAPPING)
        for y in y_true
    ]
    y_pred = [
        emotion_converter(y, mode="decode", EMOTION_MAPPING=EMOTION_MAPPING)
        for y in y_pred
    ]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    parts = path.split("_", 3)
    if len(parts) >= 3:
        plt.title(
            f"{phase} Confusion Matrix of {parts[0]} - {parts[1]} ({parts[2]})"
        )
    else:
        plt.title(f"{phase} Confusion Matrix")
    plt.savefig(
        f"./ui/public/train_val_test_logs/{path}/{path}_{phase}_confusion_matrix.png"
    )
    plt.close()


def print_classification_report(y_true, y_pred, phase, path, EMOTION_MAPPING):
    y_true = [
        emotion_converter(y, mode="decode", EMOTION_MAPPING=EMOTION_MAPPING)
        for y in y_true
    ]
    y_pred = [
        emotion_converter(y, mode="decode", EMOTION_MAPPING=EMOTION_MAPPING)
        for y in y_pred
    ]

    report = classification_report(y_true, y_pred)
    with open(
        f"./ui/public/train_val_test_logs/{path}/{path}_{phase}_classification_report.txt",
        "w",
    ) as f:
        f.write(report)


def train(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    device,
    base_path,
    ac_labels,
    EMOTION_MAPPING,
):
    logger.info("Finetuning Started...")

    os.makedirs(f"./finetuned_models/{base_path}", exist_ok=True)
    os.makedirs(f"./ui/public/train_val_test_logs/{base_path}", exist_ok=True)

    model_path = os.path.join(
        "./finetuned_models", base_path, f"{base_path}.pth"
    )
    emotion_map_path = os.path.join(
        "./finetuned_models", base_path, f"{base_path}.json"
    )
    logger.info(f"Save {emotion_map_path}...")

    with open(emotion_map_path, "w") as f:
        json.dump(EMOTION_MAPPING, f)

    best_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    y_true_val, y_pred_val = [], []
    y_true_test, y_pred_test = [], []

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")

        for phase in ["train", "val", "test"]:
            if phase not in dataloaders:
                continue

            model.train() if phase == "train" else model.eval()
            running_loss = 0.0

            for batch in dataloaders[phase]:
                labels, audio = batch["audio"], batch["labels"]
                audio, labels = audio.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(audio)
                    labels = labels.long()
                    loss = criterion(outputs.float(), labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    if phase in ["val", "test"]:
                        if phase == "val":
                            y_true_val.extend(labels.cpu().numpy())
                            y_pred_val.extend(
                                torch.argmax(outputs, dim=1).cpu().numpy()
                            )
                        else:
                            y_true_test.extend(labels.cpu().numpy())
                            y_pred_test.extend(
                                torch.argmax(outputs, dim=1).cpu().numpy()
                            )

                running_loss += loss.item() * audio.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            logger.info(f"{phase} Loss: {epoch_loss:.4f}")

            if phase == "train":
                train_losses.append(epoch_loss)
            else:
                if phase == "val":
                    val_losses.append(epoch_loss)
                    scheduler.step(epoch_loss)

                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        patience_counter = 0
                        # logger.info("Saving best model...")
                        # torch.save(model.state_dict(), model_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= EARLY_STOPPING_PATIENCE:
                            logger.info("Early stopping triggered!")
                            plot_loss(train_losses, val_losses, base_path)
                            plot_confusion_matrix(
                                y_true_val,
                                y_pred_val,
                                phase="val",
                                path=base_path,
                                classes=ac_labels,
                                EMOTION_MAPPING=EMOTION_MAPPING,
                            )
                            print_classification_report(
                                y_true_val,
                                y_pred_val,
                                phase="val",
                                path=base_path,
                                EMOTION_MAPPING=EMOTION_MAPPING,
                            )
                            plot_confusion_matrix(
                                y_true_test,
                                y_pred_test,
                                phase="test",
                                path=base_path,
                                classes=ac_labels,
                                EMOTION_MAPPING=EMOTION_MAPPING,
                            )
                            print_classification_report(
                                y_true_test,
                                y_pred_test,
                                phase="test",
                                path=base_path,
                                EMOTION_MAPPING=EMOTION_MAPPING,
                            )
                            logger.info("Logs Saved...")
                            return

    logger.info("Finetuning Ended...")
    plot_loss(train_losses, val_losses, path=base_path)
    plot_confusion_matrix(
        y_true_val,
        y_pred_val,
        phase="val",
        path=base_path,
        classes=ac_labels,
        EMOTION_MAPPING=EMOTION_MAPPING,
    )
    print_classification_report(
        y_true_val,
        y_pred_val,
        phase="val",
        path=base_path,
        EMOTION_MAPPING=EMOTION_MAPPING,
    )
    plot_confusion_matrix(
        y_true_test,
        y_pred_test,
        phase="test",
        path=base_path,
        classes=ac_labels,
        EMOTION_MAPPING=EMOTION_MAPPING,
    )
    print_classification_report(
        y_true_test,
        y_pred_test,
        phase="test",
        path=base_path,
        EMOTION_MAPPING=EMOTION_MAPPING,
    )
    logger.info("Logs Saved...")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Total CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

    device_id = 0
    device = torch.device(
        f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    )

    # ----------Modify-------------#
    CUR_DATASET = DATASET.KAZAKHEMOTIONALTTS
    CUR_BASE_MODEL = BASE_MODEL.XLS_R_1B.value
    # ----------End-------------#

    logger.info(
        f"Start finetuning {CUR_BASE_MODEL} with {CUR_DATASET.value.name}"
    )
    feature_extractor = FeatureExtractorFactory.get_extractor(
        model_name=CUR_BASE_MODEL, device=device
    )

    dataset = SpeechEmotionDataset(
        dataset_name=CUR_DATASET.value.name,
        dataset_path=f"meta_csvs/{CUR_DATASET.value.language}_{CUR_DATASET.value.name}.csv",
        language=CUR_DATASET.value.language,
    )

    logger.info(f"Emotion Map is Ready: {dataset.EMOTION_MAPPING}")

    dataloaders = get_dataloader(
        dataset, BATCH_SIZE, shuffle=True, val_split=True
    )

    num_of_classes = len(list(dataset.EMOTION_MAPPING.keys()))
    ac_labels = dataset.EMOTION_MAPPING.keys()

    base_model_name = feature_extractor.model_name.split("/")[1]

    model = SERBenchmarkModel(
        feature_extractor=feature_extractor,
        num_classes=num_of_classes,
        device=device,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    train(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        base_path=f"{CUR_DATASET.value.language}_{CUR_DATASET.value.name}_{base_model_name}",
        ac_labels=ac_labels,
        EMOTION_MAPPING=dataset.EMOTION_MAPPING,
    )

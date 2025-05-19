import os
import argparse
import yaml
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from kuralnet.models.traditional_feature_extractor import TraditionalFeatureExtractor
from kuralnet.models.pretrained_feature_extractor import WhisperFeatureExtractor
from kuralnet.dataset.dataset import EmotionDataset
from kuralnet.models.model import KuralNet
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Train emotion classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()

def plot_training_progress(train_losses, val_losses, train_accs, val_accs, save_path='training_progress.png'):
    """
    Plots training and validation loss and accuracy over epochs.

    Parameters:
    - train_losses: List or array of training loss values
    - val_losses: List or array of validation loss values
    - train_accs: List or array of training accuracy values
    - val_accs: List or array of validation accuracy values
    - save_path: File path to save the resulting plot image (default: 'training_progress.png')
    """
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_and_split_from_path(path):
    df = pd.read_csv(path)
    train, temp = train_test_split(df, test_size=0.3, stratify=df['emotion'], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['emotion'], random_state=42)
    return train, val, test


def load_full_dataset(paths):
    train_parts, val_parts, test_parts = [], [], []
    for path in paths:
        train, val, test = load_and_split_from_path(path)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
    return pd.concat(train_parts), pd.concat(val_parts), pd.concat(test_parts)


def extract_features(data_df):
    file_paths = data_df["audio_path"]
    labels = data_df["emotion"].tolist()
    whisper_features, traditional_features = [], []
    whisper_feature_extractor = WhisperFeatureExtractor()
    
    for file_path in tqdm(file_paths, desc="Extracting features"):
        audio, sr = librosa.load(file_path, sr=16000)
        whisper_features.append(whisper_feature_extractor.extract_features(audio))
        traditional_features.append(TraditionalFeatureExtractor.extract_features(audio))

    return np.array(whisper_features), np.array(traditional_features), np.array(labels)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for w, t, y in loader:
        w, t, y = w.to(device), t.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(w, t)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for w, t, y in loader:
            w, t, y = w.to(device), t.to(device), y.to(device)
            out = model(w, t)
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    args = parse_args()

    # Load config
    print(os.getcwd())
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_paths = config["dataset"]["path"]

    train_df, val_df, test_df = load_full_dataset(dataset_paths)
    train_w, train_t, train_y = extract_features(train_df)
    val_w, val_t, val_y = extract_features(val_df)
    test_w, test_t, test_y = extract_features(test_df)

    train_loader = DataLoader(EmotionDataset(train_w, train_t, train_y), batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(EmotionDataset(val_w, val_t, val_y), batch_size=config["training"]["batch_size"])
    test_loader = DataLoader(EmotionDataset(test_w, test_t, test_y), batch_size=config["training"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KuralNet().to(device)

    criterion = getattr(nn, config["loss"]["type"])()
    optimizer_class = getattr(torch.optim, config["optimizer"]["type"])
    optimizer = optimizer_class(model.parameters(), lr=config["optimizer"]["lr"])

    best_val_loss = float("inf")
    best_state = None
    patience = config["training"]["patience"]
    num_epochs = config["training"]["num_epochs"]
    counter = 0
    tr_losses,tr_accs,val_losses,val_accs = [],[],[],[]
    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        tr_losses.append(tr_loss)
        tr_accs.append(tr_acc)
        val_losses.append(val_losses)
        val_accs.append(val_acc)
        
        print(f"[Epoch {epoch+1}] Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)
        plot_training_progress(tr_losses,
    val_losses,
    tr_accs,
    val_accs)
        print("Best model loaded.")


if __name__ == "__main__":
    main()
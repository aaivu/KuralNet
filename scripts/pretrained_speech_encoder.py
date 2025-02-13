from multilingual_speech_emotion_recognition.dataset.dataset import _SpeechEmotionDataset
from multilingual_speech_emotion_recognition.utils.dataset_loader import DataLoader
from multilingual_speech_emotion_recognition.models.handcrafted_feature_extractor import HandcraftedAcousticEncoder
from data.constant import DATASET
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 

TRAIN_SPLIT = 0.8
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
    
dataset = _SpeechEmotionDataset(
    dataset_name=DATASET.SUBESCO.value.name,
    language=DATASET.SUBESCO.value.language,
    dataset_path=DATASET.SUBESCO.value.feature_path)

train_size = int(TRAIN_SPLIT * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


num_classes = len(torch.unique(dataset.emotions))
input_shape = (1, dataset.features.shape[1])
model = HandcraftedAcousticEncoder(input_shape, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(device)

best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        # Move batch to device
        features = batch['features'].to(device)
        labels = batch['emotion'].to(device)
            
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
            
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
            
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
        
    # Validation phase
    model.train(False)
    val_loss = 0.0
    val_preds = []
    val_true = []
        
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            features = batch['features'].to(device)
            labels = batch['emotion'].to(device)
            
            if len(features.shape) == 2:
                features = features.unsqueeze(1)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    val_accuracy = accuracy_score(val_true, val_preds)
        
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
    print(f'Train Loss: {avg_train_loss:.4f}')
    print(f'Val Loss: {avg_val_loss:.4f}')
    print(f'Val Accuracy: {val_accuracy:.4f}')
    print('-' * 50)
        
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= 100:
            print("Early stopping triggered!")
            break

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
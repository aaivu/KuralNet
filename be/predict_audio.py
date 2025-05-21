import joblib
import os
import gdown
from transformers import WhisperProcessor, WhisperModel
import torch
from python_speech_features import logfbank
import librosa
import numpy as np


from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Resource management
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'Final_Model')
GDRIVE_FOLDER_ID = "1Z89B9B7sl1PJktfjliS52i83PaCQujoX"
selected_features = [4, 17, 18, 21, 22, 27, 30, 32, 33, 42, 43, 47, 48, 49, 52, 54, 59, 64, 65, 77, 80, 81, 88, 89, 94, 99, 105, 122, 125, 127, 136, 147, 149, 153, 154, 161, 162, 164, 167, 171, 174, 175, 185, 192, 196, 202, 203, 205, 207, 208, 212, 214, 217, 228, 232, 233, 235, 239, 252, 257, 262, 263, 264, 267, 268, 273, 294, 297, 300, 306, 316, 318, 322, 332, 334, 341, 342, 354, 355, 359, 366, 368, 372, 373, 375, 378, 380, 390, 397, 400, 401, 410, 416, 417, 422, 424, 427, 433, 435, 442, 443, 444, 450, 453, 454, 466, 476, 480, 483, 490, 491, 495, 499, 504, 511, 513, 514, 515, 517, 521, 522, 532, 533, 535, 539, 546, 558, 559, 560, 561, 565, 584, 589, 590, 592, 595, 600, 603, 604, 610, 616, 617, 624, 631, 632, 636, 644, 645, 650, 654, 655, 665, 676, 677, 678, 679, 683, 684, 687, 708, 712, 720, 722, 723, 728, 731, 739, 740, 748, 766, 783, 785, 801, 802, 807, 811, 814, 815, 816, 823, 825, 830, 831, 834, 846, 849, 850, 851, 857, 863, 865, 866, 871, 880, 892, 900, 903, 905, 907, 909, 915, 916, 920, 926, 927, 935, 938, 949, 953, 955, 960, 973, 975, 979, 982, 989, 991, 993, 998, 999, 1021, 1022, 1026, 1034, 1036, 1037, 1040, 1043, 1044, 1046, 1060, 1066, 1068, 1070, 1074, 1079, 1080, 1081, 1082, 1089, 1090, 1097, 1099, 1101, 1102, 1104, 1105, 1115, 1116, 1125, 1133, 1136, 1140, 1144, 1145, 1148, 1149, 1150, 1155, 1160, 1161, 1163, 1169, 1171, 1175, 1184, 1186, 1189, 1190, 1191, 1193, 1194, 1205, 1208, 1209, 1213, 1222, 1228, 1233, 1234, 1244, 1256, 1259, 1260, 1264, 1269, 1276, 1279]

def download_resources():
    """Download required resources from Google Drive"""
    os.makedirs(RESOURCES_DIR, exist_ok=True)
    
    # Download folder from Google Drive
    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}?usp=sharing"
    gdown.download_folder(url=url, output=RESOURCES_DIR, quiet=False)
    
    print(f"Resources downloaded to {RESOURCES_DIR}")

def load_resources():
    """Load required resources or download if not present"""
    if not os.path.exists(RESOURCES_DIR):
        print("Downloading required resources...")
        download_resources()



def process_audio_whisper(audio_path, sr=16000):
    
    model_name = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)

    # Load and preprocess the audio file
    audio, sr = librosa.load(audio_path, sr=sr)  # Whisper expects 16kHz audio
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

    # Pass the inputs through the model
    with torch.no_grad():
        outputs = model.encoder(inputs.input_features)

    # Encoder outputs as embeddings
    embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_length, embedding_dim)

    # Aggregate embeddings to a single vector per audio sample
    aggregated_embeddings = embeddings.mean(dim=1)  # Shape: (batch_size, embedding_dim)
    
    return aggregated_embeddings.cpu().numpy().tolist()

def process_audio_acoustic(audio_chunk: np.ndarray, sr: int) -> np.ndarray:
    
    data, sample_rate = librosa.load(audio_chunk, sr=sr)
    result = np.array([])

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma Feature
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    # Perceptual Linear Predictive (PLP) features
    plp = np.mean(logfbank(data, sample_rate, nfilt=26, nfft=min(512, len(data))), axis=0)
    result = np.hstack((result, plp))

    # Pitch
    pitches, magnitudes = librosa.core.piptrack(y=data, sr=sample_rate)
    pitch = np.mean(pitches[pitches > 0])
    result = np.hstack((result, pitch))

    # Pitch Difference
    pitch_diff = np.diff(pitches[pitches > 0]).mean()
    result = np.hstack((result, pitch_diff))

    # Energy
    energy = np.sum(data**2) / np.float64(len(data))
    result = np.hstack((result, energy))

    # Linear Predictive Coding Coefficients (LPCC)
    lpcc = librosa.lpc(data, order=13)
    result = np.hstack((result, lpcc))

    # Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spectral_centroid))

    # Spectral Bandwidth
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spectral_bandwidth))

    # Spectral Contrast
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spectral_contrast))

    # Spectral Flatness
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=data).T, axis=0)
    result = np.hstack((result, spectral_flatness))

    # Spectral Rolloff
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spectral_rolloff))

    # Tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T, axis=0)
    result = np.hstack((result, tonnetz))

    return result


def extract_features(audio_chunk: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract features from the audio chunk.
    
    Args:
        audio_chunk: Audio chunk data
        sr: Sample rate
        
    Returns:
        Feature vector
    """

    whisper_embeddings = process_audio_whisper(audio_chunk, sr)
    
    whisper_embeddings_array = np.array(whisper_embeddings[0])
    whisper_features = whisper_embeddings_array[selected_features]
    
    acoustic_features = process_audio_acoustic(audio_chunk, sr) 
    
    return whisper_features, acoustic_features
        
def preprocess_input(whisper_embeddings, acoustic_features, whisper_scaler, acoustic_scaler):
    """
    Preprocess input features for the model.
    
    Args:
        whisper_embeddings: whisper embeddings
        acoustic_features: acoustic features
        whisper_scaler: Whisper feature scaler
        acoustic_scaler: Acoustic feature scaler
    Returns:
        Preprocessed feature vector
    """
    whisper_scaled = whisper_scaler.transform(whisper_embeddings.reshape(1, -1))
    acoustic_scaled = acoustic_scaler.transform(acoustic_features.reshape(1, -1))
    
    return whisper_scaled, acoustic_scaled

def load_trained_models():
    """
    Load pre-trained models, scalers, and label encoders.

    Returns:
        Acoustic model, Linguistic model, Whisper scaler, Acoustic scaler, Whisper label encoder, Acoustic label encoder.
    """
    acoustic_model = load_model(os.path.join(RESOURCES_DIR, 'Acoustic/Acoustic_model.h5'))
    lingustic_model = load_model(os.path.join(RESOURCES_DIR, 'Whisper/Lingustic_model.h5'))
    whisper_scaler = joblib.load(os.path.join(RESOURCES_DIR, 'Whisper/scaler.pkl'))
    acoustic_scaler = joblib.load(os.path.join(RESOURCES_DIR, 'Acoustic/Utils/Acoustic_scaler.pkl'))
    whisper_label = joblib.load(os.path.join(RESOURCES_DIR, 'Whisper/label_encoder.pkl'))
    acoustic_label = joblib.load(os.path.join(RESOURCES_DIR, 'Acoustic/Utils/Acoustic_label_encoder.pkl'))
    
    return acoustic_model, lingustic_model, whisper_scaler, acoustic_scaler, whisper_label, acoustic_label

if __name__ == "__main__":
    load_resources()
    acoustic_model, lingustic_model, whisper_scaler, acoustic_scaler, whisper_label, acoustic_label = load_trained_models()
    print("Resources loaded successfully.")
    
    # Example usage
    audio_path = input("Enter the path to the audio file: ")
    
    if not os.path.exists(audio_path):
        print(f"File {audio_path} does not exist.")
    else:
        print(f"Processing audio file: {audio_path}")
        
    sr = 16000  # Sample rate
    whisper_embeddings, acoustic_features = extract_features(audio_path, sr=sr)
    whisper_scaled, acoustic_scaled = preprocess_input(whisper_embeddings, acoustic_features, whisper_scaler, acoustic_scaler)
    acoustic_prediction = acoustic_model.predict(acoustic_scaled)
    lingustic_prediction = lingustic_model.predict(whisper_scaled)
    print("Model A Prediction:", acoustic_prediction[0])
    print("Model B Prediction:", lingustic_prediction[0])
    print("Combined prediction:", (acoustic_prediction[0] + lingustic_prediction[0])/2)
    # Decode the predictions
    acoustic_predicted_label = acoustic_label.inverse_transform(np.argmax(acoustic_prediction, axis=1))
    lingustic_predicted_label = whisper_label.inverse_transform(np.argmax(lingustic_prediction, axis=1))
    print("Model A Predicted Label:", acoustic_predicted_label[0])
    print("Model B Predicted Label:", lingustic_predicted_label[0])
    
    acoustic_probs = acoustic_prediction
    linguistic_probs = lingustic_prediction

    # Average the probabilities
    combined_probs = (acoustic_probs + linguistic_probs) / 2

    # Map the predicted class index to emotion label
    emotion_labels = whisper_label.classes_

    # Create a dictionary with emotion labels and their probabilities
    predicted_emotions = {emotion_labels[i]: float(combined_probs[0][i]) for i in range(len(emotion_labels))}
    
    emotions = {
    "angry": int(predicted_emotions['Anger'] * 100),
    "sad": int(predicted_emotions['Sadness'] * 100),
    "fear": int(predicted_emotions['Fear'] * 100),
    "happy": int(predicted_emotions['Happiness'] * 100),
    "neutral": int(predicted_emotions['Neutral'] * 100)
    }
    
    main_emotion = max(emotions, key=emotions.get)
    
    print(f"Predicted emotion : {main_emotion},\nPredicted values : {emotions}")
    
import os
import logging
from dataset_processor import process_dataset

def process_emota_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for root, _, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith(".wav"):
                file_path = os.path.join(root, file_name)
                parts = file_name.split('.')
                if len(parts) > 2:
                    emo_abb = parts[2].split("-")[0]
                    emotion = emotion_map.get(emo_abb)
                    if emotion is None:
                        logging.warning(f"Emotion {emo_abb} not found in emotion_map, file: {file_path}")
                        continue
                    if emotion in selected_emotions:
                        data.append([emotion, file_path])
    return data

if __name__ == "__main__":
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/kaggle/input/speech-emotion-recognition-hindi"
    emotion_map = {
        "anger": "Anger",
        "disgust": "Disgust",
        "fear": "Fear",
        "happy": "Happiness",
        "neutral": "Neutral",
        "sad": "Sadness",
        "sarcastic": "Sarcastic",
        "surprise": "Surprise"
    }

    process_dataset(
        dataset_path=dataset_path,
        language_code="hi",
        dataset_name="hindi",
        emotion_map=emotion_map,
        selected_emotions=selected_emotions,
        file_processor=process_emota_files
    )
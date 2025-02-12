import os
import logging
from dataset_processor import process_dataset
from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS


HINDI_DATASET = DATASET.HINDI_DATASET.value
EMOTION_MAP = {
    "anger": EMOTION.ANGER.value,
    "disgust": EMOTION.DISGUST.value,
    "fear": EMOTION.FEAR.value,
    "happy": EMOTION.HAPPINESS.value,
    "neutral": EMOTION.NEUTRAL.value,
    "sad": EMOTION.SADNESS.value,
    "sarcastic": EMOTION.SARCASTIC.value,
    "surprise": EMOTION.SURPRISE.value
}

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
    process_dataset(
        dataset_path=HINDI_DATASET.path,
        language_code=HINDI_DATASET.language,
        dataset_name=HINDI_DATASET.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_emota_files
    )
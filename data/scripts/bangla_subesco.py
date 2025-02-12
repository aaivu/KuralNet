import logging
import os

from dataset_processor import process_dataset


def process_subesco_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for directory in os.listdir(dataset_path):
        val = directory.split('_')[5]
        file_path = os.path.join(dataset_path, directory)
        emotion = emotion_map.get(val)
        if not emotion:
            logging.warning(f"Emotion {val} not found in emotion_map, file: {file_path}")
            continue
        if emotion in selected_emotions:
            data.append([emotion, file_path])
    return data

if __name__ == "__main__":
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/kaggle/input/subescobangla-speech-emotion-dataset/SUBESCO/"
    emotion_map = {
        "ANGRY": "Anger",
        "SAD": "Sadness",
        "SURPRISE": "Surprise",
        "HAPPY": "Happiness",
        "FEAR": "Fear",
        "NEUTRAL": "Neutral",
        "DISGUST": "Disgust"
    }

    process_dataset(
        dataset_path=dataset_path,
        language_code="bn",
        dataset_name="subesco",
        emotion_map=emotion_map,
        selected_emotions=selected_emotions,
        file_processor=process_subesco_files
    )


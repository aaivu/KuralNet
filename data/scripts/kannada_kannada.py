import logging
import os

from dataset_processor import process_dataset


def process_kannada_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for directory in os.listdir(dataset_path):
        val = directory.split('.')[0].split('-')[1]
        if int(val) == 3:  # Skip surprise emotion
            continue
        file_path = os.path.join(dataset_path, directory)
        try:
            emotion = emotion_map[int(val)]
        except KeyError:
            logging.warning(f"Emotion {val} not found in emotion_map")
            continue
        if emotion in selected_emotions:
            data.append([emotion, file_path])
    return data

if __name__ == "__main__":
    selected_emotions = ['Fear', 'Sadness', 'Happiness', 'Anger', 'Neutral']
    dataset_path = "/kaggle/input/kannada-emo-speech-dataset/"
    emotion_map = {
        1: "Anger",
        2: "Sadness",
        3: "Surprise",
        4: "Happiness",
        5: "Fear",
        6: "Neutral"
    }

    process_dataset(
        dataset_path=dataset_path,
        language_code="kn",
        dataset_name="kannada",
        emotion_map=emotion_map,
        selected_emotions=selected_emotions,
        file_processor=process_kannada_files
    )


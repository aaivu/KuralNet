import logging
import os

from dataset_processor import process_dataset

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS

RAVDESS = DATASET.RAVDESS.value
EMOTION_MAP = {
    "01": EMOTION.NEUTRAL.value,
    "02": EMOTION.CALM.value,
    "03": EMOTION.HAPPINESS.value,
    "04": EMOTION.SADNESS.value,
    "05": EMOTION.ANGER.value,
    "06": EMOTION.FEAR.value,
    "07": EMOTION.DISGUST.value,
    "08": EMOTION.SURPRISE.value,
}


def process_ravdess_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for directory in os.listdir(dataset_path):
        directory_path = os.path.join(dataset_path, directory)
        if os.path.isdir(directory_path):
            for filename in os.listdir(directory_path):
                if filename.endswith(".wav"):
                    emo_abb = filename.split("-")[2]
                    emotion = emotion_map.get(emo_abb)
                    if not emotion:
                        logging.warning(f"Emotion not found for {filename}")
                        continue
                    if emotion in selected_emotions:
                        file_path = os.path.join(directory_path, filename)
                        data.append([emotion, file_path])
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=RAVDESS.path,
        language_code=RAVDESS.language,
        dataset_name=RAVDESS.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_ravdess_files,
    )

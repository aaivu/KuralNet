import logging
import os

from data.meta_extractors.dataset_processor import process_dataset

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS

URUDU_DATASET = DATASET.URDU_DATASET.value
EMOTION_MAP = {
    "Angry": EMOTION.ANGER.value,
    "Happy": EMOTION.HAPPINESS.value,
    "Neutral": EMOTION.NEUTRAL.value,
    "Sad": EMOTION.SADNESS.value,
}


def process_urudu_files(dataset_path, emotion_map, selected_emotions):
    data = []
    dir_list = os.listdir(dataset_path)  # dir_list: emotion names
    for dir in dir_list:
        emotion = emotion_map.get(dir)
        if not emotion:
            logging.warning(f"Emotion not found for {dir}")
            continue
        if emotion not in selected_emotions:
            continue
        file_list = os.listdir(os.path.join(dataset_path, dir))
        for file in file_list:
            file_path = os.path.join(dataset_path, dir, file)
            data.append([emotion, file_path])
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=URUDU_DATASET.path,
        language_code=URUDU_DATASET.language,
        dataset_name=URUDU_DATASET.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_urudu_files,
    )

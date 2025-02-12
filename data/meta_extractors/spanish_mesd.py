import logging
import os

from data.meta_extractors.dataset_processor import process_dataset

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS

MESD = DATASET.MESD.value
EMOTION_MAP = {
    "Anger: ": EMOTION.ANGER.value,
    "Disgust: ": EMOTION.DISGUST.value,
    "Fear: ": EMOTION.FEAR.value,
    "Happiness: ": EMOTION.HAPPINESS.value,
    "Neutral: ": EMOTION.NEUTRAL.value,
    "Sadness: ": EMOTION.SADNESS.value,
}


def process_mesd_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for file_name in os.listdir(dataset_path):
        emotion = emotion_map.get(file_name.split("_")[0])
        if not emotion:
            logging.warning(f"Emotion not found for {file_name}")
            continue
        if emotion in selected_emotions:
            dir_path = os.path.join(dataset_path, file_name)
            data.append([emotion, dir_path])
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=DATASET.MESD.value.path,
        language_code=DATASET.MESD.value.language,
        dataset_name=DATASET.MESD.value.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_mesd_files,
    )

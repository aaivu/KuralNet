import logging
import os

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from data.meta_extractors.dataset_processor import process_dataset

SUBESCO = DATASET.SUBESCO.value
EMOTION_MAP = {
    "ANGRY": EMOTION.ANGER.value,
    "SAD": EMOTION.SADNESS.value,
    "SURPRISE": EMOTION.SURPRISE.value,
    "HAPPY": EMOTION.HAPPINESS.value,
    "FEAR": EMOTION.FEAR.value,
    "NEUTRAL": EMOTION.NEUTRAL.value,
    "DISGUST": EMOTION.DISGUST.value,
}


def process_subesco_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for directory in os.listdir(dataset_path):
        val = directory.split("_")[5]
        file_path = os.path.join(dataset_path, directory)
        emotion = emotion_map.get(val)
        if not emotion:
            logging.warning(
                f"Emotion {val} not found in emotion_map, file: {file_path}"
            )
            continue
        if emotion in selected_emotions:
            data.append([emotion, file_path])
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=SUBESCO.path,
        language_code=SUBESCO.language,
        dataset_name=SUBESCO.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_subesco_files,
    )

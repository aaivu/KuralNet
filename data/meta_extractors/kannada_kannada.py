import logging
import os

from data.meta_extractors.dataset_processor import process_dataset

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS

KANNADA_DATASET = DATASET.KANNADA_DATASET.value
EMOTION_MAP = {
    1: EMOTION.ANGER.value,
    2: EMOTION.SADNESS.value,
    3: EMOTION.SURPRISE.value,
    4: EMOTION.HAPPINESS.value,
    5: EMOTION.FEAR.value,
    6: EMOTION.NEUTRAL.value,
}


def process_kannada_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for directory in os.listdir(dataset_path):
        val = directory.split(".")[0].split("-")[1]
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
    process_dataset(
        dataset_path=KANNADA_DATASET.path,
        language_code=KANNADA_DATASET.language,
        dataset_name=KANNADA_DATASET.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_kannada_files,
    )

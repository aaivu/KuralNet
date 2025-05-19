import logging
import os

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

AESDD = DATASET.AESDD.value
EMOTION_MAP = {
    "anger": EMOTION.ANGER.value,
    "happiness": EMOTION.HAPPINESS.value,
    "sadness": EMOTION.SADNESS.value,
    "disgust": EMOTION.DISGUST.value,
    "fear": EMOTION.FEAR.value,
}


def process_aesdd_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for emotion in emotion_map.keys():
        emotion_dir = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_dir):
            logging.warning(f"Directory {emotion_dir} not found")
            continue
        if emotion_map[emotion] not in selected_emotions:
            continue
        for file_name in os.listdir(emotion_dir):
            if file_name.endswith(".wav"):
                file_path = os.path.join(emotion_dir, file_name)
                data.append([emotion_map[emotion], file_path])
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=AESDD.path,
        language_code=AESDD.language,
        dataset_name=AESDD.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_aesdd_files,
    )

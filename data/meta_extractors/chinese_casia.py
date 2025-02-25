import logging
import os

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from data.meta_extractors.dataset_processor import process_dataset

CASIA = DATASET.CASIA.value
EMOTION_MAP = {
    "anger": EMOTION.ANGER.value,
    "fear": EMOTION.FEAR.value,
    "happy": EMOTION.HAPPINESS.value,
    "neutral": EMOTION.NEUTRAL.value,
    "sad": EMOTION.SADNESS.value,
    "surprise": EMOTION.SURPRISE.value,
}


def process_casia_files(dataset_path, emotion_map, selected_emotions):
    data = []
    emotions = os.listdir(dataset_path)
    for emotion in emotions:
        mapped_emotion = emotion_map.get(emotion)
        if not mapped_emotion:
            logging.warning(f"Emotion {emotion} not found in emotion_map")
            continue
        if mapped_emotion not in selected_emotions:
            continue
        files = os.listdir(os.path.join(dataset_path, emotion))
        for file in files:
            if file.endswith(".wav"):
                data.append(
                    [mapped_emotion, os.path.join(dataset_path, emotion, file)]
                )
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=CASIA.path,
        language_code=CASIA.language,
        dataset_name=CASIA.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_casia_files,
    )

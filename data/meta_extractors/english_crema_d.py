import logging
import os

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from data.meta_extractors.dataset_processor import process_dataset

CREMA_D = DATASET.CREMA_D.value
EMOTION_MAP = {
    "ANG": EMOTION.ANGER.value,
    "DIS": EMOTION.DISGUST.value,
    "FEA": EMOTION.FEAR.value,
    "HAP": EMOTION.HAPPINESS.value,
    "NEU": EMOTION.NEUTRAL.value,
    "SAD": EMOTION.SADNESS.value,
}


def process_crema_d_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(dataset_path, file_name)
            emo_abb = file_name.split("_")[2].strip()
            emotion = emotion_map.get(emo_abb)
            if not emotion:
                logging.warning(
                    f"Emotion {emo_abb} not found in emotion_map, file: {file_path}"
                )
                continue
            if emotion in selected_emotions:
                data.append([emotion, file_path])
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=CREMA_D.path,
        language_code=CREMA_D.language,
        dataset_name=CREMA_D.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_crema_d_files,
    )

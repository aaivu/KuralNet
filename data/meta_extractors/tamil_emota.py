import logging
import os

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from data.meta_extractors.dataset_processor import process_dataset

EMOTA = DATASET.EMOTA.value
EMOTION_MAP = {
    "ang": EMOTION.ANGER.value,
    "hap": EMOTION.HAPPINESS.value,
    "neu": EMOTION.NEUTRAL.value,
    "fea": EMOTION.FEAR.value,
    "sad": EMOTION.SADNESS.value,
}


def process_emota_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(dataset_path, file_name)
            emo_abb = file_name.split(".")[0].split("_")[2].strip()
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
        dataset_path=EMOTA.path,
        language_code=EMOTA.language,
        dataset_name=EMOTA.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_emota_files,
    )

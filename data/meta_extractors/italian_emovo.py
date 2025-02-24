import logging
import os

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from data.meta_extractors.dataset_processor import process_dataset

EMOVO = DATASET.EMOVO.value
EMOTION_MAP = {
    "dis": EMOTION.DISGUST.value,
    "gio": EMOTION.JOY.value,
    "neu": EMOTION.NEUTRAL.value,
    "pau": EMOTION.FEAR.value,
    "rab": EMOTION.ANGER.value,
    "sor": EMOTION.SURPRISE.value,
    "tri": EMOTION.SADNESS.value,
}


def process_emovo_files(dataset_path, emotion_map, selected_emotions):
    data = []
    for directory in os.listdir(dataset_path):
        if directory == "documents":
            continue
        for filename in os.listdir(os.path.join(dataset_path, directory)):
            if filename.endswith(".wav"):
                emo_abb = filename.split("-")[0]
                emotion = emotion_map.get(emo_abb)
                if not emotion:
                    logging.warning(f"Emotion not found for {filename}")
                    continue
                if emotion in selected_emotions:
                    file_path = os.path.join(dataset_path, directory, filename)
                    data.append([emotion, file_path])
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=EMOVO.path,
        language_code=EMOVO.language,
        dataset_name=EMOVO.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_emovo_files,
    )

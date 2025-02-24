import logging
import os

import pandas as pd

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from data.meta_extractors.dataset_processor import process_dataset

EMOZIONALMENTE = DATASET.EMOZIONALMENTE.value
EMOTION_MAP = {
    "anger": EMOTION.ANGER.value,
    "disgust": EMOTION.DISGUST.value,
    "fear": EMOTION.FEAR.value,
    "joy": EMOTION.JOY.value,
    "neutrality": EMOTION.NEUTRAL.value,
    "sadness": EMOTION.SADNESS.value,
    "surprise": EMOTION.SURPRISE.value,
}


def process_emoz_files(dataset_path, emotion_map, selected_emotions):
    data = []
    meta_csv_file = os.path.join(dataset_path, "metadata/samples.csv")
    meta_df = pd.read_csv(meta_csv_file)
    for index, row in meta_df.iterrows():
        emotion = emotion_map.get(row["emotion_expressed"])
        if not emotion:
            logging.warning(f"Emotion not found for {row['emotion']}")
            continue
        if emotion not in selected_emotions:
            continue
        file_path = os.path.join(dataset_path, "audio/" + row["file_name"])
        data.append([emotion, file_path])
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=EMOZIONALMENTE.path,
        language_code=EMOZIONALMENTE.language,
        dataset_name=EMOZIONALMENTE.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_emoz_files,
    )

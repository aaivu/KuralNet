import logging

import pandas as pd

from src.scripts.meta_extractors.dataset_processor import process_dataset
from src.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from src.utils.utils import get_wav_files

EMOZIONALMENTE = DATASET.EMOZIONALMENTE.value
EMOTION_MAP = {
    "anger": EMOTION.ANGER.value,
    "sadness": EMOTION.SADNESS.value,
    "disgust": EMOTION.DISGUST.value,
    "surprise": EMOTION.SURPRISE.value,
    "fear": EMOTION.FEAR.value,
    "joy": EMOTION.JOY.value,
    "neutrality": EMOTION.NEUTRAL.value,
}


def process_EMOZIONALMENTE_files(dataset_path, emotion_map, selected_emotions):
    df = pd.read_csv(
        "ser_datasets/Emozionalmente/emozionalmente_dataset/metadata/samples.csv"
    )
    data = []
    try:
        wav_files = get_wav_files(dataset_path)
        for file in wav_files:
            emo = emotion_map[
                df.loc[
                    df["file_name"] == file.split("/")[-1], "emotion_expressed"
                ].values[0]
            ]

            if emo not in selected_emotions:
                continue
            data.append([emo, file])

    except Exception as e:
        logging.error(
            f"Error processing {EMOZIONALMENTE.name} files: {str(e)}"
        )

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=EMOZIONALMENTE.path,
        language_code=EMOZIONALMENTE.language,
        dataset_name=EMOZIONALMENTE.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_EMOZIONALMENTE_files,
    )

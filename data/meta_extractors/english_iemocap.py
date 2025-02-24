import logging
import os
import pandas as pd

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from data.meta_extractors.dataset_processor import process_dataset

IEMOCAP = DATASET.IEMOCAP.value
EMOTION_MAP = {
    "ang": EMOTION.ANGER.value,
    "hap": EMOTION.HAPPINESS.value,
    "sad": EMOTION.SADNESS.value,
    "neu": EMOTION.NEUTRAL.value,
    "fru": EMOTION.FRUSTRATION.value,
    "exc": EMOTION.EXCITEMENT.value,
    "fea": EMOTION.FEAR.value,
    "sur": EMOTION.SURPRISE.value,
    "dis": EMOTION.DISGUST.value,
    # "oth": EMOTION.OTHER.value,
    # "xxx": EMOTION.OTHER.value,
}

def process_iemocap_files(dataset_path, emotion_map, selected_emotions):
    data = []
    csv_file = os.path.join(dataset_path, "iemocap_meta_extracted.csv")
    try:
        df = pd.read_csv(csv_file)
        for index, row in df.iterrows():
            emotion = emotion_map.get(row["emotion"])
            if not emotion:
                logging.warning(f"Emotion {row['emotion']} not found for {row['file_path']}")
                continue
            if emotion in selected_emotions:
                data.append([emotion, row["file_path"]])
    except FileNotFoundError:
        logging.error(f"File not found: {csv_file}")
    return data

if __name__ == "__main__":
    process_dataset(
        dataset_path=IEMOCAP.path,
        language_code=IEMOCAP.language,
        dataset_name=IEMOCAP.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_iemocap_files,
    )
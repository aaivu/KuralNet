import logging
import os

import pandas as pd

from src.scripts.meta_extractors.dataset_processor import process_dataset
from src.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

ANAD = DATASET.ANAD.value
EMOTION_MAP = {
    "surprised": EMOTION.SURPRISE.value,
    "happy": EMOTION.HAPPINESS.value,
    "angry": EMOTION.ANGER.value,
}


def process_anad_files(dataset_path, emotion_map, selected_emotions):
    data = []
    metadata = pd.read_csv(os.path.join(dataset_path, "ANAD.csv"))
    metadata["name"] = metadata["name"].str.slice(0, -1)
    meta_available = metadata["name"].values
    audio_dir = [
        "1sec_segmented_part1",
        "1sec_segmented_part2",
        "1sec_segmented_part3",
    ]
    for dir in audio_dir:
        files = os.listdir(os.path.join(dataset_path, dir, dir))
        for file_name in files:
            if file_name.endswith(".wav") and file_name in meta_available:
                file_path = os.path.join(dataset_path, dir, dir, file_name)
                emotion = metadata[metadata["name"] == file_name][
                    "Emotion "
                ].values[0]
                emotion = emotion_map.get(emotion)
                if not emotion:
                    logging.warning(
                        f"Emotion {emotion} not found in emotion_map, file: {file_path}"
                    )
                    continue
                if emotion in selected_emotions:
                    data.append([emotion, file_path])
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=ANAD.path,
        language_code=ANAD.language,
        dataset_name=ANAD.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_anad_files,
    )

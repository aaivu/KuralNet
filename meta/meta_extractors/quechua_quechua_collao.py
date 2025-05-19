import logging
import os

import pandas as pd

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

from kuralnet.utils.utils import stratified_sampling

QUECHUA_COLLAO = DATASET.QUECHUA_COLLAO.value
EMOTION_MAP = {
    "anger": EMOTION.ANGER.value,
    "boredom": EMOTION.BOREDOM.value,
    "happy": EMOTION.HAPPINESS.value,
    "sleepy": EMOTION.BOREDOM.value,
    "sadness": EMOTION.SADNESS.value,
    "calm": EMOTION.CALMNESS.value,
    "fear": EMOTION.FEAR.value,
    "excited": EMOTION.EXCITEMENT.value,
    "neutral": EMOTION.NEUTRAL.value,
    "angry": EMOTION.ANGER.value,
    "bored": EMOTION.BOREDOM.value,
}


def process_quechua_collao_files(dataset_path, emotion_map, selected_emotions):
    data = []
    meta_data = pd.read_excel(
        os.path.join(dataset_path, "Data/Data/Data.xlsx"), sheet_name="map"
    )
    meta_data["Audio"] = meta_data["Audio"].astype(str) + ".wav"
    audio_files = os.listdir(os.path.join(dataset_path, "Audios"))
    for file in audio_files:
        if file.endswith(".wav"):
            file_path = os.path.join(dataset_path, "Audios", file)
            emotion = meta_data[meta_data["Audio"] == file]["Emotion"].values[
                0
            ]
            emotion = emotion_map.get(emotion)
            if not emotion:
                logging.warning(
                    f"Emotion {emotion} not found in emotion_map, file: {file_path}"
                )
                continue
            if emotion in selected_emotions:
                data.append([emotion, file_path])

    data = stratified_sampling(data, 3000)
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=QUECHUA_COLLAO.path,
        language_code=QUECHUA_COLLAO.language,
        dataset_name=QUECHUA_COLLAO.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_quechua_collao_files,
    )

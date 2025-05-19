import logging
import os

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

nEMO = DATASET.NEMO.value
EMOTION_MAP = {
    "fear": EMOTION.FEAR.value,
    "sadness": EMOTION.SADNESS.value,
    "happiness": EMOTION.HAPPINESS.value,
    "anger": EMOTION.ANGER.value,
    "neutral": EMOTION.NEUTRAL.value,
    "surprise": EMOTION.SURPRISE.value,
}


def process_nemo_files(dataset_path, emotion_map, selected_emotions):

    data = []
    try:
        dir_list = os.listdir(dataset_path)

        for file_name in dir_list:
            if not file_name.endswith(".wav"):
                continue

            parts = file_name.split("_")
            if len(parts) < 2:
                logging.warning(
                    f"Skipping file with unexpected format: {file_name}"
                )
                continue

            emotion_key = parts[1]
            emotion = emotion_map.get(emotion_key)

            if emotion not in selected_emotions:
                continue

            file_path = os.path.join(dataset_path, file_name)
            data.append([emotion, file_path])

    except Exception as e:
        logging.error(f"Error processing nEMO files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=nEMO.path,
        language_code=nEMO.language,
        dataset_name=nEMO.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_nemo_files,
    )

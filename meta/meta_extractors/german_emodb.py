import logging
import os

from src.scripts.meta_extractors.dataset_processor import process_dataset
from src.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

EMODB = DATASET.EMODB.value
EMOTION_MAP = {
    "W": EMOTION.ANGER.value,
    "E": EMOTION.DISGUST.value,
    "F": EMOTION.HAPPINESS.value,
    "L": EMOTION.BOREDOM.value,
    "T": EMOTION.SADNESS.value,
    "A": EMOTION.FEAR.value,
    "N": EMOTION.NEUTRAL.value,
}


def process_emodb_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        for file_name in os.listdir(dataset_path):
            try:
                emo_code = file_name[-6:-5]  # Extract emotion code
                emotion = emotion_map.get(emo_code)

                if not emotion:
                    logging.warning(
                        f"Unknown emotion code {emo_code} for file: {file_name}"
                    )
                    continue

                if emotion in selected_emotions:
                    file_path = os.path.join(dataset_path, file_name)
                    data.append([emotion, file_path])

            except Exception as e:
                logging.warning(f"Error processing file {file_name}: {str(e)}")
                continue

    except Exception as e:
        logging.error(f"Error processing EMoDB files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=EMODB.path,
        language_code=EMODB.language,
        dataset_name=EMODB.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_emodb_files,
    )

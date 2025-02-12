import logging
import os
from pathlib import Path
from typing import Dict, List

from dataset_processor import process_dataset

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS

CAFE = DATASET.CAFE.value
EMOTION_MAP = {
    "C": EMOTION.ANGER.value,
    "D": EMOTION.DISGUST.value,
    "J": EMOTION.HAPPINESS.value,
    "N": EMOTION.NEUTRAL.value,
    "P": EMOTION.FEAR.value,
    "S": EMOTION.SURPRISE.value,
    "T": EMOTION.SADNESS.value,
}


def process_cafe_files(
    dataset_path: str, emotion_map: Dict[str, str], selected_emotions: List[str]
) -> List[List[str]]:
    data = []
    dataset_path = Path(dataset_path)

    try:
        dir_list = os.listdir(dataset_path)
        # Remove metadata files
        metadata_files = {"Version-ChangeLog.txt", "License.txt", "Readme.txt"}
        dir_list = [d for d in dir_list if d not in metadata_files]

        for directory in dir_list:
            dir_path = dataset_path / directory
            for subdir in os.listdir(dir_path):
                directory_path = dir_path / subdir
                if not directory_path.is_dir():
                    continue

                for filename in os.listdir(directory_path):
                    try:
                        emo_abb = filename.split("-")[1]
                        emotion = emotion_map[emo_abb]

                        if emotion in selected_emotions:
                            file_path = str(directory_path / filename)
                            data.append([emotion, file_path])
                    except (KeyError, IndexError) as e:
                        logging.warning(f"Error processing file {filename}: {str(e)}")
                        continue

    except Exception as e:
        logging.error(f"Error processing CaFE files: {str(e)}")
        raise

    return data


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    process_dataset(
        dataset_path=CAFE.path,
        language_code=CAFE.language,
        dataset_name=CAFE.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_cafe_files,
    )

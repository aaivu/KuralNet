import logging
import os

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from data.meta_extractors.dataset_processor import process_dataset

ASED = DATASET.ASED.value
EMOTION_MAP = {
    "Fear": EMOTION.FEAR.value,
    "Sadness": EMOTION.SADNESS.value,
    "Happiness": EMOTION.HAPPINESS.value,
    "Anger": EMOTION.ANGER.value,
    "Neutral": EMOTION.NEUTRAL.value,
}


def process_ased_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        dir_list = os.listdir(dataset_path)
        dir_list.remove(".gitattributes")
        dir_list.remove("README.md")

        for directory in dir_list:
            dir_path = os.path.join(dataset_path, directory)
            sub_dir = os.listdir(dir_path)
            emotion = directory[2:]  # Remove 'E_' prefix
            emotion = emotion_map.get(emotion)

            if emotion not in selected_emotions:
                continue

            for file_name in sub_dir:
                file_path = os.path.join(dir_path, file_name)
                data.append([emotion, file_path])

    except Exception as e:
        logging.error(f"Error processing ASED files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=ASED.path,
        language_code=ASED.language,
        dataset_name=ASED.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_ased_files,
    )
    print("here")

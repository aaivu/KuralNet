import logging
import os

from src.scripts.meta_extractors.dataset_processor import process_dataset
from src.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

SITBOSED = DATASET.SITBOSED.value
EMOTION_MAP = {
    "01": EMOTION.ANGER.value,
    "02": EMOTION.DISGUST.value,
    "03": EMOTION.FEAR.value,
    "04": EMOTION.HAPPINESS.value,
    "05": EMOTION.SADNESS.value,
    "06": EMOTION.SURPRISE.value,
}


def process_sitb_osed_files(dataset_path, emotion_map, selected_emotions):

    data = []
    try:
        dir_list = os.listdir(dataset_path)
        for directory in dir_list:
            dir_path = os.path.join(dataset_path, directory)

            if not os.path.isdir(dir_path):
                continue

            files = os.listdir(dir_path)
            for file in files:
                if not file.endswith(".wav"):
                    continue

                parts = file.split(".")[0].split("-")
                if len(parts) != 4:
                    print(
                        f"Skipping file with unexpected format: {os.path.join(dir_path, file)}"
                    )
                    continue

                emotion_code = parts[1]
                emotion = emotion_map.get(emotion_code)

                if emotion not in selected_emotions:
                    continue

                file_path = os.path.join(dir_path, file)
                data.append([emotion, file_path])

    except Exception as e:
        logging.error(f"Error processing SITB-OSED audio files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=SITBOSED.path,
        language_code=SITBOSED.language,
        dataset_name=SITBOSED.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_sitb_osed_files,
    )

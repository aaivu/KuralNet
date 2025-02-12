import logging
import os

from data.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from data.meta_extractors.dataset_processor import process_dataset

ESD_CHINESE = DATASET.ESD_CHINESE.value
EMOTION_MAP = {
    "Angry": EMOTION.ANGER.value,
    "Happy": EMOTION.HAPPINESS.value,
    "Neutral": EMOTION.NEUTRAL.value,
    "Sad": EMOTION.SADNESS.value,
    "Surprise": EMOTION.SURPRISE.value,
}


def process_esd_files(dataset_path, emotion_map, selected_emotions):
    # TODO: Refactor this function according to new format
    data = []
    dir_list = os.listdir(dataset_path)
    if ".DS_Store" in dir_list:
        dir_list.remove(".DS_Store")
    dir_list.sort()
    dir_list = dir_list[:10]  # Limiting to first 10 folders (chinese actors)

    for directory in dir_list:  # directory: actor
        dir_path = os.path.join(
            dataset_path, directory
        )  # dir_path: dataset_path + actor
        sub_dir = os.listdir(dir_path)  # sub_dir: emotions list
        if ".DS_Store" in sub_dir:
            sub_dir.remove(".DS_Store")
        for dir in sub_dir:  # dir: emotion
            if dir.endswith(".txt"):
                continue
            emotion = EMOTION_MAP.get(dir)
            if not emotion:
                logging.warning(f"Emotion not found for {dir}")
                continue
            if emotion not in selected_emotions:
                continue
            path = os.path.join(
                dir_path, dir
            )  # path: dataset_path + actor + emotion
            for filename in os.listdir(path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(path, filename)
                    data.append([emotion, file_path])
    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=ESD_CHINESE.path,
        language_code=ESD_CHINESE.language,
        dataset_name=ESD_CHINESE.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_esd_files,
    )

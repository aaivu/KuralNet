import logging

from src.scripts.meta_extractors.dataset_processor import process_dataset
from src.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from src.utils.utils import get_wav_files

KESDy18 = DATASET.KESDy18.value
EMOTION_MAP = {
    "sad": EMOTION.SADNESS.value,
    "happy": EMOTION.HAPPINESS.value,
    "angry": EMOTION.ANGER.value,
    "neutral": EMOTION.NEUTRAL.value,
}


def process_KESDy18_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        wav_files = get_wav_files(dataset_path)
        for file in wav_files:
            emo = emotion_map[file.split("/")[-2]]
            if emo not in selected_emotions:
                continue
            data.append([emo, file])

    except Exception as e:
        logging.error(f"Error processing KESDy18 files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=KESDy18.path,
        language_code=KESDy18.language,
        dataset_name=KESDy18.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_KESDy18_files,
    )

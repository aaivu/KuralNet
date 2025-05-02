import logging

from src.scripts.meta_extractors.dataset_processor import process_dataset
from src.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from src.utils.utils import get_wav_files

IESC = DATASET.IESC.value
EMOTION_MAP = {
    "Fear": EMOTION.FEAR.value,
    "Sad": EMOTION.SADNESS.value,
    "Happy": EMOTION.HAPPINESS.value,
    "Anger": EMOTION.ANGER.value,
    "Neutral": EMOTION.NEUTRAL.value,
}


def process_ased_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        wav_files = get_wav_files(dataset_path)

        for file in wav_files:
            emo = EMOTION_MAP[file.split("/")[-2]]
            if emo not in selected_emotions:
                continue
            data.append([emo, file])

    except Exception as e:
        logging.error(f"Error processing ASED files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=IESC.path,
        language_code=IESC.language,
        dataset_name=IESC.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_ased_files,
    )

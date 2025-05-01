import logging

from src.scripts.meta_extractors.dataset_processor import process_dataset
from src.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS
from src.utils.utils import get_wav_files

INDOWAVESENTIMENT = DATASET.INDOWAVESENTIMENT.value
EMOTION_MAP = {
    "05": EMOTION.DISAPPOINTMENT.value,
    "02": EMOTION.HAPPINESS.value,
    "04": EMOTION.DISGUST.value,
    "01": EMOTION.NEUTRAL.value,
    "03": EMOTION.SURPRISE.value,
}


def process_INDOWAVESENTIMENT_files(
    dataset_path, emotion_map, selected_emotions
):
    data = []
    try:
        wav_files = get_wav_files(dataset_path)
        for file in wav_files:
            emo = emotion_map[file.split("/")[-1].split("-")[1]]
            if emo not in selected_emotions:
                continue
            data.append([emo, file])

    except Exception as e:
        logging.error(
            f"Error processing {INDOWAVESENTIMENT.name} files: {str(e)}"
        )

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=INDOWAVESENTIMENT.path,
        language_code=INDOWAVESENTIMENT.language,
        dataset_name=INDOWAVESENTIMENT.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_INDOWAVESENTIMENT_files,
    )

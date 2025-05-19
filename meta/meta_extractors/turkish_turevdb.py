import logging

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

from kuralnet.utils.utils import get_wav_files

TUREVDB = DATASET.TUREVDB.value
EMOTION_MAP = {
    "Sad": EMOTION.SADNESS.value,
    "Happy": EMOTION.HAPPINESS.value,
    "Angry": EMOTION.ANGER.value,
    "Calm": EMOTION.CALMNESS.value,
}


def process_TUREVDB_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        wav_files = get_wav_files(dataset_path)
        for file in wav_files:
            emo = emotion_map[file.split("/")[-2]]
            if emo not in selected_emotions:
                continue
            data.append([emo, file])

    except Exception as e:
        logging.error(f"Error processing TUREVDB files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=TUREVDB.path,
        language_code=TUREVDB.language,
        dataset_name=TUREVDB.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_TUREVDB_files,
    )

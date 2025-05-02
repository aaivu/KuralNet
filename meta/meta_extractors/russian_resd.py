import logging

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

from kuralnet.utils.utils import get_wav_files

RESD = DATASET.RESD.value
EMOTION_MAP = {
    "a": EMOTION.ANGER.value,
    "d": EMOTION.DISGUST.value,
    "e": EMOTION.ENTHUSIASM.value,
    "f": EMOTION.FEAR.value,
    "h": EMOTION.HAPPINESS.value,
    "n": EMOTION.NEUTRAL.value,
    "s": EMOTION.SADNESS.value,
}


def process_RESD_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        wav_files = get_wav_files(dataset_path)
        for file in wav_files:
            emo = emotion_map[file.split("/")[-1][-9]]
            if emo not in selected_emotions:
                continue
            data.append([emo, file])

    except Exception as e:
        logging.error(f"Error processing {RESD.name} files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=RESD.path,
        language_code=RESD.language,
        dataset_name=RESD.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_RESD_files,
    )

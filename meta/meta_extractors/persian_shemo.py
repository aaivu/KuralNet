import logging

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

from kuralnet.utils.utils import get_wav_files

SHEMO = DATASET.SHEMO.value
EMOTION_MAP = {
    "F": EMOTION.FEAR.value,
    "S": EMOTION.SADNESS.value,
    "H": EMOTION.HAPPINESS.value,
    "A": EMOTION.ANGER.value,
    "N": EMOTION.NEUTRAL.value,
    "W": EMOTION.SURPRISE.value,
}


def process_ased_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        wav_files = get_wav_files(dataset_path)
        for file in wav_files:
            emo = emotion_map[file.split("/")[-1][3]]
            if emo not in selected_emotions:
                continue
            data.append([emo, file])

    except Exception as e:
        logging.error(f"Error processing ASED files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=SHEMO.path,
        language_code=SHEMO.language,
        dataset_name=SHEMO.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_ased_files,
    )

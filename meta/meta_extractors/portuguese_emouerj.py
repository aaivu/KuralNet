import logging

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

from kuralnet.utils.utils import get_wav_files

EMOUERJ = DATASET.EMOUERJ.value
EMOTION_MAP = {
    "s": EMOTION.SADNESS.value,
    "h": EMOTION.HAPPINESS.value,
    "a": EMOTION.ANGER.value,
    "n": EMOTION.NEUTRAL.value,
}


def process_EMOUERJ_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        wav_files = get_wav_files(dataset_path)
        for file in wav_files:
            emo = emotion_map[file.split("/")[-1][3]]
            if emo not in selected_emotions:
                continue
            data.append([emo, file])

    except Exception as e:
        logging.error(f"Error processing EMOUERJ files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=EMOUERJ.path,
        language_code=EMOUERJ.language,
        dataset_name=EMOUERJ.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_EMOUERJ_files,
    )

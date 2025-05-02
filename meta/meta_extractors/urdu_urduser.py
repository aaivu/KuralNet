import logging

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

from kuralnet.utils.utils import get_wav_files

URDUSER = DATASET.URDUSER.value
EMOTION_MAP = {
    "Angry": EMOTION.ANGER.value,
    "Boredom": EMOTION.BOREDOM.value,
    "Disgust": EMOTION.DISGUST.value,
    "Fear": EMOTION.FEAR.value,
    "Happy": EMOTION.HAPPINESS.value,
    "Neutral": EMOTION.NEUTRAL.value,
    "Sad": EMOTION.SADNESS.value,
}


def process_URDUSER_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        wav_files = get_wav_files(dataset_path)
        for file in wav_files:
            emo = emotion_map[file.split("/")[-2]]
            if emo not in selected_emotions:
                continue
            data.append([emo, file])

    except Exception as e:
        logging.error(f"Error processing {URDUSER.name} files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=URDUSER.path,
        language_code=URDUSER.language,
        dataset_name=URDUSER.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_URDUSER_files,
    )

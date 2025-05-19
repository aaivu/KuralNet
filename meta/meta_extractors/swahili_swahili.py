import logging

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

from kuralnet.utils.utils import get_wav_files

SWAHILI_DATASET = DATASET.SWAHILI_DATASET.value
EMOTION_MAP = {
    "sad": EMOTION.SADNESS.value,
    "happy": EMOTION.HAPPINESS.value,
    "angry": EMOTION.ANGER.value,
    "calm": EMOTION.CALMNESS.value,
    "surprised": EMOTION.SURPRISE.value,
}


def process_SWAHILI_DATASET_files(
    dataset_path, emotion_map, selected_emotions
):
    data = []
    try:
        wav_files = get_wav_files(dataset_path)
        for file in wav_files:
            emo = emotion_map[file.split("/")[-2]]
            if emo not in selected_emotions:
                continue
            data.append([emo, file])

    except Exception as e:
        logging.error(f"Error processing SWAHILI_DATASET files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=SWAHILI_DATASET.path,
        language_code=SWAHILI_DATASET.language,
        dataset_name=SWAHILI_DATASET.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_SWAHILI_DATASET_files,
    )

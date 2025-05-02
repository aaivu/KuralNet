import logging

from meta.meta_extractors.dataset_processor import process_dataset
from kuralnet.utils.constant import DATASET, EMOTION, SELECTED_EMOTIONS

from kuralnet.utils.utils import get_wav_files

JVNV = DATASET.JVNV.value
EMOTION_MAP = {
    "sad": EMOTION.SADNESS.value,
    "happy": EMOTION.HAPPINESS.value,
    "anger": EMOTION.ANGER.value,
    "surprise": EMOTION.SURPRISE.value,
    "disgust": EMOTION.DISGUST.value,
    "fear": EMOTION.FEAR.value,
}


def process_JVNV_files(dataset_path, emotion_map, selected_emotions):
    data = []
    try:
        wav_files = get_wav_files(dataset_path)
        for file in wav_files:
            emo = emotion_map[file.split("/")[-1].split("_")[1]]
            if emo not in selected_emotions:
                continue
            data.append([emo, file])

    except Exception as e:
        logging.error(f"Error processing JVNV files: {str(e)}")

    return data


if __name__ == "__main__":
    process_dataset(
        dataset_path=JVNV.path,
        language_code=JVNV.language,
        dataset_name=JVNV.name,
        emotion_map=EMOTION_MAP,
        selected_emotions=SELECTED_EMOTIONS,
        file_processor=process_JVNV_files,
    )
